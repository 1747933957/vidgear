# -*- coding: utf-8 -*-
"""
slow_module.py
--------------
A Tooth-like Slow-module customized for your system.

Inputs (per sample):
  - rate_hist: [H]      # 历史丢包率序列（0..1）
  - dur_hist : [H]      # 历史冗余持续轮次序列（>=0 的整数；若缺失可用 0 或均值填充）
  - recentN  : [N]      # 过去 N 个包接收(1)/丢失(0)的 0-1 向量

Outputs (per sample):
  - next_loss_rate in [0,1]
  - next_duration   >= 0  (训练用回归；推理时 round 到整数)

训练数据来源：
  - 使用 receiver.py 输出的 TSV 文件（第1行表头；第 n 帧写到第 n+2 行）：
    frame_id \t loss_rate_at_send \t redundancy_rounds \t [0/1,...]

Author: Viduce-SlowModule (custom)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import ast
import math
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Model
# --------------------------
class CNN1D(nn.Module):
    def __init__(self, in_ch: int = 1, N: int = 100, c: int = 32):
        super().__init__()
        # 1D-CNN 提取过去 N 包的空间模式
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=c, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=5, padding=2)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c

    def forward(self, x):  # x: [B, 1, N]
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)  # [B, C]
        return h

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)

class SlowModuleNet(nn.Module):
    def __init__(self, H: int, N: int, c_cnn: int = 32, d_mlp: int = 64):
        super().__init__()
        self.H, self.N = H, N
        self.cnn = CNN1D(in_ch=1, N=N, c=c_cnn)
        self.mlp_rate = MLP(in_dim=H, hidden=d_mlp, out_dim=d_mlp)
        self.mlp_dur  = MLP(in_dim=H, hidden=d_mlp, out_dim=d_mlp)
        fuse_in = self.cnn.out_dim + d_mlp + d_mlp
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, 128),
            nn.ReLU(),
        )
        # 双头
        self.head_loss = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()  # [0,1]
        )
        self.head_dur = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU()     # >=0
        )

    def forward(self, rate_hist, dur_hist, recentN):
        """
        rate_hist: [B, H]
        dur_hist : [B, H]
        recentN  : [B, N]
        """
        x_rate = self.mlp_rate(rate_hist.float())
        x_dur  = self.mlp_dur(dur_hist.float())
        x_vec  = self.cnn(recentN.unsqueeze(1).float())  # -> [B, C]
        x = torch.cat([x_rate, x_dur, x_vec], dim=-1)
        z = self.fuse(x)
        pred_loss = self.head_loss(z).squeeze(-1)
        pred_dur  = self.head_dur(z).squeeze(-1)
        return pred_loss, pred_dur

# --------------------------
# Dataset from receiver TSV
# --------------------------
@dataclass
class Sample:
    rate_hist: List[float]
    dur_hist : List[float]
    recentN  : List[int]
    tgt_loss : float
    tgt_dur  : float

class SlowTSVDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path: Path, H: int, N: int):
        self.H, self.N = H, N
        self.items: List[Sample] = []
        self._load(tsv_path)

    @staticmethod
    def _parse_vec(s: str) -> List[int]:
        # s 是类似 "[0,1,1,...]"
        try:
            v = ast.literal_eval(s.strip())
            if isinstance(v, list):
                return [int(x) for x in v]
        except Exception:
            pass
        return []

    def _load(self, path: Path):
        rows = path.read_text(encoding="utf-8").splitlines()
        # rows[0] 是表头；从第2行开始是第0帧
        records = []  # [(fid, rate, dur, vec)]
        for i in range(1, len(rows)):
            parts = rows[i].split("\t")
            if len(parts) < 4:
                continue
            try:
                fid = int(parts[0])
            except Exception:
                continue
            rate = None if parts[1] == "NA" else float(parts[1])
            dur  = None if parts[2] == "NA" else float(parts[2])
            vec  = self._parse_vec(parts[3])
            records.append((fid, rate, dur, vec))

        # 构造监督数据：用第 i 帧的历史 -> 预测第 i+1 帧
        # 对于历史序列，不足 H 的在头部用最近可用值或 0 填充；对于缺失 dur，使用 0 作为占位
        def get_hist(arr, idx, H, default=0.0):
            seq = []
            for t in range(idx-H+1, idx+1):
                if t < 0:
                    seq.append(default)
                else:
                    seq.append(arr[t] if arr[t] is not None else default)
            return seq

        rates = [r[1] for r in records]  # 可能包含 None（NA）
        durs  = [r[2] for r in records]
        vecs  = [r[3] for r in records]

        for i in range(0, len(records)-1):
            # 目标是 i+1 帧：需要其标签都可用（尤其 loss_rate 必须有；dur 没有也可以用 0 训练）
            _, tgt_rate, tgt_dur, _ = records[i+1]
            if tgt_rate is None:
                continue
            tgt_dur_val = 0.0 if (tgt_dur is None) else float(tgt_dur)
            rate_hist = get_hist(rates, i, self.H, default=0.0)
            dur_hist  = get_hist(durs,  i, self.H, default=0.0)
            recentN   = vecs[i][-self.N:] if len(vecs[i]) >= self.N else ([0] * (self.N - len(vecs[i])) + vecs[i])
            self.items.append(Sample(rate_hist, dur_hist, recentN, float(tgt_rate), tgt_dur_val))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        s = self.items[idx]
        return (
            torch.tensor(s.rate_hist, dtype=torch.float32),
            torch.tensor(s.dur_hist,  dtype=torch.float32),
            torch.tensor(s.recentN,   dtype=torch.float32),
            torch.tensor(s.tgt_loss,  dtype=torch.float32),
            torch.tensor(s.tgt_dur,   dtype=torch.float32),
        )

# --------------------------
# Trainer / API
# --------------------------
class SlowModule:
    def __init__(self, H: int = 5, N: int = 100, device: Optional[str] = None):
        self.H, self.N = H, N
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SlowModuleNet(H=H, N=N).to(self.device)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "H": self.H, "N": self.N,
            "state": self.net.state_dict(),
        }, str(path))

    @staticmethod
    def load(path: Path, device: Optional[str] = None) -> "SlowModule":
        ckpt = torch.load(str(path), map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sm = SlowModule(H=ckpt["H"], N=ckpt["N"], device=device)
        sm.net.load_state_dict(ckpt["state"])
        return sm

    def fit_from_tsv(self, tsv_path: Path, epochs: int = 20, bs: int = 64, lr: float = 1e-3, val_split: float = 0.1):
        ds = SlowTSVDataset(tsv_path, H=self.H, N=self.N)
        n_total = len(ds)
        if n_total == 0:
            raise RuntimeError("Dataset empty. Ensure your recv_stats.txt has enough rows.")
        # 简单划分
        n_val = max(1, int(n_total * val_split))
        idx = list(range(n_total))
        random.shuffle(idx)
        val_idx = set(idx[:n_val])
        train_items = torch.utils.data.Subset(ds, [i for i in range(n_total) if i not in val_idx])
        val_items   = torch.utils.data.Subset(ds, [i for i in range(n_total) if i in val_idx])

        train_loader = torch.utils.data.DataLoader(train_items, batch_size=bs, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_items, batch_size=bs, shuffle=False)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        # loss: 下一帧丢包率 -> BCE-like (MSE 到 [0,1])；持续轮次 -> L1 回归（低估罚重可在此加权）
        for ep in range(epochs):
            self.net.train()
            sum_l = sum_d = sum_b = 0.0
            for rate_hist, dur_hist, recentN, tgt_loss, tgt_dur in train_loader:
                rate_hist = rate_hist.to(self.device)
                dur_hist  = dur_hist.to(self.device)
                recentN   = recentN.to(self.device)
                tgt_loss  = tgt_loss.to(self.device)
                tgt_dur   = tgt_dur.to(self.device)
                pred_loss, pred_dur = self.net(rate_hist, dur_hist, recentN)
                # 低估惩罚：对 pred_loss < tgt_loss 的样本加权
                w = torch.where(pred_loss < tgt_loss, torch.tensor(2.0, device=self.device), torch.tensor(1.0, device=self.device))
                loss_loss = torch.mean(w * (pred_loss - tgt_loss) ** 2)
                loss_dur  = F.l1_loss(pred_dur, tgt_dur)
                loss = loss_loss + 0.3 * loss_dur
                opt.zero_grad()
                loss.backward()
                opt.step()
                sum_l += float(loss_loss.item()); sum_d += float(loss_dur.item()); sum_b += 1.0
            # 验证
            self.net.eval()
            with torch.no_grad():
                vl_l = vl_d = vb = 0.0
                for rate_hist, dur_hist, recentN, tgt_loss, tgt_dur in val_loader:
                    rate_hist = rate_hist.to(self.device)
                    dur_hist  = dur_hist.to(self.device)
                    recentN   = recentN.to(self.device)
                    tgt_loss  = tgt_loss.to(self.device)
                    tgt_dur   = tgt_dur.to(self.device)
                    pred_loss, pred_dur = self.net(rate_hist, dur_hist, recentN)
                    w = torch.where(pred_loss < tgt_loss, torch.tensor(2.0, device=self.device), torch.tensor(1.0, device=self.device))
                    loss_loss = torch.mean(w * (pred_loss - tgt_loss) ** 2)
                    loss_dur  = F.l1_loss(pred_dur, tgt_dur)
                    vl_l += float(loss_loss.item()); vl_d += float(loss_dur.item()); vb += 1.0
            print(f"[Ep {ep+1:02d}] train: loss={sum_l/sum_b:.4f} dur={sum_d/sum_b:.4f} | val: loss={vl_l/max(1,vb):.4f} dur={vl_d/max(1,vb):.4f}")

    @torch.no_grad()
    def predict(self, rate_hist: List[float], dur_hist: List[float], recentN: List[int]) -> Tuple[float, int]:
        """
        给 sender 使用的在线接口：
          输入三路（长度分别为 H、H、N），输出：下一帧丢包率（float）、下一帧持续轮次（int）
        """
        assert len(rate_hist) == self.H and len(dur_hist) == self.H and len(recentN) == self.N
        self.net.eval()
        r = torch.tensor([rate_hist], dtype=torch.float32, device=self.device)
        d = torch.tensor([dur_hist],  dtype=torch.float32, device=self.device)
        v = torch.tensor([recentN],   dtype=torch.float32, device=self.device)
        pred_loss, pred_dur = self.net(r, d, v)
        loss = float(pred_loss.item())
        dur  = max(0, int(round(float(pred_dur.item()))))
        return loss, dur

# 便捷命令行（可选）：训练并保存
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, default= "/home/wxk/workspace/nsdi/Viduce/net/vidgear/recv_stats.txt", help="Path to recv_stats.txt produced by receiver.")
    ap.add_argument("--out", type=str, default= "slow_module.pt", help="Path to save model .pt")
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    sm = SlowModule(H=args.H, N=args.N)
    sm.fit_from_tsv(Path(args.tsv), epochs=args.epochs, bs=args.bs, lr=args.lr)
    sm.save(Path(args.out))
