# -*- coding: utf-8 -*-
"""
slow_module.py
==============
严格按照论文 Tooth 的 slow-module 结构实现（仅 slow-module；fast-module 由上层使用 RF 实现）。

功能概述
--------
- 在线推理：在每个 RTCP 周期被调用一次，输入过去 n 个周期的 (P_lr, P_la) 与拼接的逐包位图向量 l_s，
  经过 1D-CNN(ks=100,4)+pooling(size=2) -> AdaptiveAvgPool1d(12) + FC(64->32->16)，
  回归得到下一周期 (lr_f, la_f)。
- 离线训练：读取 JSONL 日志（每行一个 RTCP 周期），按滑动窗口构造样本并用式(3) 训练网络；
  训练完成后保存权重(JSON)，运行时可加载。

论文对齐（关键点）
------------------
- 输入：P_lr(10×1), P_la(10×1), P_ls(由过去 n 个 RTCP 周期逐包 0/1 到达向量 l_s 经 1D-CNN+pooling 得到)
- CNN：两层 1D 卷积，kernel_size 分别为 100 和 4；每层后接 ReLU + Pooling(kernel_size=2)
- FC：三层全连接，单元数 64 -> 32 -> 16，输出 2 维 (lr_f, la_f)
- 输出：下一周期 lr_f, la_f
- 损失：式(3) 不对称放大低估损失的 MSE 变体
- 触发频率：每个 RTCP 周期一次（非逐帧）

日志格式（训练数据）
--------------------
每行 JSON，至少包含：
  {
    "ts_ms": <窗口开始或结束时间戳，毫秒>,
    "lr": <该窗口 loss rate, 0..1>,
    "la": <该窗口 loss aggregation>,
    "pls_bits_b64": "<base64 编码的 0/1 位图，表示过去一个窗口的逐包到达/丢失>",
    "pls_slot_ms": <每个 bit 的时隙长度，毫秒>,
    "pls_epoch_ms": <该位图覆盖的起始时间，毫秒>,
    "window_ms": <该窗口时长，毫秒>
  }
上层落盘时保证顺序或可按 ts_ms 排序。

命令行用法
----------
训练:
  python slow_module.py train --log_dir ./slow_logs --epochs 5 --batch_size 64 \
      --out ./slow_module_weights.json

评估（仅前向，不改变权重）:
  python slow_module.py eval --log_dir ./slow_logs --weights ./slow_module_weights.json
"""

from __future__ import annotations
import os, sys

def _avoid_asyncio_shadow():
    """避免本地 vidgear/gears/asyncio 阴影掉标准库 asyncio。"""
    new_path = []
    shadow = os.path.join("vidgear", "vidgear", "gears")
    for p in sys.path:
        # 跳过当前工作目录 '' 以及指向 …/vidgear/vidgear/gears 的项
        if p == "" or (isinstance(p, str) and p.endswith(shadow)):
            continue
        new_path.append(p)
    # 把被跳过的项放到末尾，标准库路径优先
    for p in sys.path:
        if p == "" or (isinstance(p, str) and p.endswith(shadow)):
            new_path.append(p)
    sys.path[:] = new_path

_avoid_asyncio_shadow()

import json, base64, argparse, math, random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# 常量
# =========================
PLS_OUT_DIM = 12   # 论文里的 P_ls 输出维度（自适应池化）
N_HIST = 10        # 历史窗口数（论文用 10）

# =========================
# 模型定义（论文结构）
# =========================

class SlowModuleCNN(nn.Module):
    """
    1D-CNN 分支：对 l_s（逐包 0/1 到达/丢失向量）做两层卷积 + ReLU + 池化，
    kernel_size 固定为 100 与 4；池化核为 2；最后自适应池化到 12 维。
    """
    def __init__(self, in_ch: int = 1, c_mid: int = 16):
        """
        参数:
          - in_ch: 输入通道数（l_s 是单通道 => 1）
          - c_mid: 中间通道数（论文未指定通道数，这里取一个极小值 16 以“压缩”，
                   不改变论文中 kernel/pooling/输出维度的设定）
        """
        super().__init__()
        # 第一层卷积：kernel_size = 100
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=c_mid, kernel_size=100, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # 第二层卷积：kernel_size = 4
        self.conv2 = nn.Conv1d(in_channels=c_mid, out_channels=c_mid, kernel_size=4, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # 自适应到 12 维，得到论文里的 P_ls(12×1)
        self.adapt = nn.AdaptiveAvgPool1d(output_size=PLS_OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
          - x: [B, 1, Ls] 逐包 0/1 向量（拼接过去 n 个窗口的位图）
        输出:
          - feat: [B, 12] 论文中的 P_ls(12×1)
        """
        h = self.conv1(x)             # [B, C, L1]
        h = F.relu(h)
        h = self.pool1(h)             # [B, C, L1/2]
        h = self.conv2(h)             # [B, C, L2]
        h = F.relu(h)
        h = self.pool2(h)             # [B, C, L2/2]
        h = self.adapt(h)             # [B, C, 12]
        # 论文写的是 12×1，我们把通道维做均值得到 12 维
        h = h.mean(dim=1)             # [B, 12]
        return h


class SlowModuleFC(nn.Module):
    """
    FC 干路：输入拼接 [P_lr(10), P_la(10), P_ls(12)] = 32 维，
    经过三层线性层，单元数 64 -> 32 -> 16，逐层 ReLU，最后回归出 2 个标量 (lr_f, la_f)。
    """
    def __init__(self, in_dim: int = (10 + 10 + 12)):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)  # 输出 [lr_f, la_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.out(h)  # [B, 2]
        return y


class ToothSlowModule(nn.Module):
    """
    论文 slow-module 的组合：CNN(l_s)->P_ls(12) 与历史 P_lr(10), P_la(10) 拼接，过 FC 得到 (lr_f, la_f)。
    """
    def __init__(self):
        super().__init__()
        self.cnn = SlowModuleCNN(in_ch=1, c_mid=16)
        self.fc = SlowModuleFC(in_dim=(10 + 10 + 12))

    @staticmethod
    def _as_tensor_1d(x: List[float], target_len: int) -> torch.Tensor:
        """
        把 Python 列表变成长度为 target_len 的 1D Tensor，不足则用边界值延展（与实际工程一致性好）。
        - 用“末值延展”而不是 0 填充，避免训练/推理时引入非自然的 0。
        """
        if len(x) == 0:
            arr = [0.0] * target_len
        elif len(x) >= target_len:
            arr = x[-target_len:]
        else:
            arr = x[:] + [x[-1]] * (target_len - len(x))
        return torch.tensor(arr, dtype=torch.float32)

    def forward(self, P_lr: List[float], P_la: List[float], l_s_bits: List[int]) -> torch.Tensor:
        """
        前向推理（单样本）：
          - P_lr: 长度<=10 的历史 lr 序列
          - P_la: 长度<=10 的历史 la 序列
          - l_s_bits: 逐包 0/1 向量（把过去 n 个窗口位图按时间拼起来）
        返回:
          - y: Tensor[2] = [lr_f, la_f]  （未裁剪，交由上层按需要裁剪/约束）
        """
        # 准备 10×1 的 P_lr / P_la
        t_lr = self._as_tensor_1d(P_lr, 10)  # [10]
        t_la = self._as_tensor_1d(P_la, 10)  # [10]
        # 准备 l_s：转为 float 并加 batch/通道维
        if len(l_s_bits) == 0:
            l_s_bits = [0]
        t_ls = torch.tensor(l_s_bits, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,Ls]
        # CNN 提取 P_ls(12)
        feat_ls = self.cnn(t_ls)  # [1,12]
        # 拼接成为 32 维
        feat = torch.cat([t_lr.unsqueeze(0), t_la.unsqueeze(0), feat_ls], dim=1)  # [1,32]
        # FC 输出
        y = self.fc(feat).squeeze(0)  # [2]
        return y  # [lr_f, la_f]


# =========================
# 损失函数（论文式 3）
# =========================

class ToothAsymmetricLoss(nn.Module):
    """
    论文式(3) 的不对称损失：低估 (lr_f, la_f) 时惩罚更大。
    L = ((lr_f - lr_hat)^2 + (la_f - la_hat)^2)/2 * exp((lr_hat - lr_f + la_hat - la_f) / (|…| + beta))
    其中 beta=1.
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B,2]  -> [lr_f, la_f]
        target: [B,2] -> [lr_hat, la_hat]
        """
        diff = pred - target                        # [B,2]
        mse = 0.5 * torch.sum(diff * diff, dim=1)  # [B]
        delta = (target[:, 0] - pred[:, 0]) + (target[:, 1] - pred[:, 1])
        amp = torch.exp(delta) / (torch.abs(delta) + self.beta)
        loss = mse * amp
        return loss.mean()


# =========================
# 数据集与读取
# =========================

def _decode_bits_from_b64(b64: str) -> List[int]:
    """
    把 base64 编码的 0/1 字节流还原成 bit 列表（每比特 0/1）。
    落盘时按 LSB-first 打包（slot0 在最低有效位），训练时也按 LSB-first 还原。
    """
    raw = base64.b64decode(b64.encode("ascii"))
    bits: List[int] = []
    for byte in raw:
        for k in range(8):
            bits.append((byte >> k) & 1)   # ✅ LSB-first，与 netgear_udp 写入一致
    return bits


@dataclass
class Sample:
    P_lr: List[float]      # 长度<=10
    P_la: List[float]      # 长度<=10
    l_s_bits: List[int]    # 逐包 0/1 拼接向量（过去 n 个窗口）
    y: Tuple[float, float] # (lr_hat_next, la_hat_next)


class RTCPJsonlDataset(Dataset):
    """
    从 JSONL 日志构造“下一步预测”数据集：
      用第 i..i+9 周期的 (P_lr, P_la) & 拼接 i..i+9 周期的位图，去预测第 i+10 周期的 (lr, la)。
    """
    def __init__(self, paths: List[str], n_hist: int = N_HIST, max_samples: Optional[int] = None):
        super().__init__()
        self.samples: List[Sample] = []
        # 读并合并
        rows: List[Dict[str, Any]] = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        # 按时间排序（若需要）
        rows.sort(key=lambda r: float(r.get("ts_ms", 0.0)))

        # 滑动窗口构造样本
        self.n_hist = int(n_hist)
        if len(rows) < (self.n_hist + 1):
            return

        for i in range(0, max(0, len(rows) - (n_hist))):
            # 输入窗口：i .. i+n_hist-1
            win = rows[i:i + n_hist]
            # 监督目标：第 i+n_hist 条（下一周期）
            if i + n_hist >= len(rows):
                break
            nxt = rows[i + n_hist]

            P_lr = [float(w.get("lr", 0.0)) for w in win]
            P_la = [float(w.get("la", 0.0)) for w in win]

            # 拼接位图（严格遵循论文：把过去 n 个周期的逐包 0/1 合并成单个 l_s）
            l_s_bits: List[int] = []
            for w in win:
                b64 = w.get("pls_bits_b64", "")
                if b64:
                    l_s_bits.extend(_decode_bits_from_b64(b64))
                else:
                    # 若缺失，则用全 0 占位（长度=window_ms/pls_slot_ms）
                    window_ms = int(w.get("window_ms", 100))
                    slots = max(1, int(window_ms / int(w.get("pls_slot_ms", 1)) ))
                    l_s_bits.extend([0] * slots)

            y = (float(nxt.get("lr", 0.0)), float(nxt.get("la", 0.0)))
            self.samples.append(Sample(P_lr=P_lr, P_la=P_la, l_s_bits=l_s_bits, y=y))

            if (max_samples is not None) and (len(self.samples) >= max_samples):
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        # 10×1 的 P_lr / P_la
        t_lr = ToothSlowModule._as_tensor_1d(s.P_lr, 10)  # [10]
        t_la = ToothSlowModule._as_tensor_1d(s.P_la, 10)  # [10]
        # l_s_bits -> [1, Ls]（不再在 Loader 里额外 unsqueeze 通道维）
        if len(s.l_s_bits) == 0:
            s.l_s_bits = [0]
        t_ls = torch.tensor(s.l_s_bits, dtype=torch.float32).unsqueeze(0)  # [1,Ls]
        # 标签
        y = torch.tensor([s.y[0], s.y[1]], dtype=torch.float32)           # [2]
        return t_lr, t_la, t_ls, y


# =========================
# 训练与评估
# =========================

def train_slow_module(log_dir: str, out_path: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3, seed: int = 42) -> None:
    """
    使用 JSONL 日志离线训练 slow-module：
      - 数据：log_dir 下所有 .jsonl
      - 优化：Adam + 论文式(3) 损失
    """
    random.seed(seed); torch.manual_seed(seed)
    # 收集文件
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    if len(files) == 0:
        raise RuntimeError(f"在 {log_dir} 未找到 .jsonl 日志")
    ds = RTCPJsonlDataset(files, n_hist=N_HIST)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ToothSlowModule()
    criterion = ToothAsymmetricLoss(beta=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        steps = 0
        for t_lr, t_la, t_ls, y in dl:
            # 前向（逐样本循环；t_ls 已是 [B,1,Ls]，不要再 unsqueeze）
            preds: List[torch.Tensor] = []
            B = t_lr.shape[0]
            for b in range(B):
                yb = model(
                    P_lr=t_lr[b].tolist(),
                    P_la=t_la[b].tolist(),
                    l_s_bits=t_ls[b,0].tolist(),
                )
                preds.append(yb.unsqueeze(0))
            pred = torch.cat(preds, dim=0)  # [B,2]

            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(1, steps)
        print(f"[epoch {ep}] loss={avg:.6f}")

    # 保存权重（JSON）
    save_weights(model, out_path)
    print(f"[save] weights -> {out_path}")


def evaluate_slow_module(log_dir: str, weight_path: str, max_batches: int = 10, batch_size: int = 32) -> None:
    """
    简单评估：前向跑若干批，打印预测均值与标签均值（非指标，仅 sanity check）。
    """
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    ds = RTCPJsonlDataset(files, n_hist=N_HIST)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = ToothSlowModule()
    load_weights(model, weight_path)
    model.eval()

    n = 0
    p_sum = torch.zeros(2)
    t_sum = torch.zeros(2)
    batches = 0
    for t_lr, t_la, t_ls, y in dl:
        # 前向（逐样本循环；t_ls 已是 [B,1,Ls]，不要再 unsqueeze）
        preds: List[torch.Tensor] = []
        B = t_lr.shape[0]
        for b in range(B):
            yb = model(
                P_lr=t_lr[b].tolist(),
                P_la=t_la[b].tolist(),
                l_s_bits=t_ls[b,0].tolist(),
            )
            preds.append(yb.unsqueeze(0))
        pred = torch.cat(preds, dim=0)  # [B,2]

        p_sum += pred.sum(dim=0)
        t_sum += y.sum(dim=0)
        n += B
        batches += 1
        if batches >= max_batches:
            break

    p_avg = (p_sum / max(1, n)).tolist()
    t_avg = (t_sum / max(1, n)).tolist()
    print(f"[eval] pred_avg(lr_f,la_f)={p_avg}, target_avg={t_avg}, n={n}")


# =========================
# 权重保存/加载（JSON）
# =========================

def save_weights(model: ToothSlowModule, path: str) -> None:
    """
    为了便于跨语言/版本加载，采用 JSON 形式保存（state_dict 的浮点列表）。
    """
    sd = model.state_dict()
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[save] json weights -> {path}")

def load_weights(model: ToothSlowModule, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sd = model.state_dict()
    for k, w in data.items():
        if k in sd:
            sd[k] = torch.tensor(w, dtype=sd[k].dtype)
    model.load_state_dict(sd, strict=False)
    print(f"[load] json weights <- {path}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--log_dir", type=str, required=True)
    ap_train.add_argument("--epochs", type=int, default=5)
    ap_train.add_argument("--batch_size", type=int, default=64)
    ap_train.add_argument("--out", type=str, required=True)
    ap_train.add_argument("--lr", type=float, default=1e-3)

    ap_eval = sub.add_parser("eval")
    ap_eval.add_argument("--log_dir", type=str, required=True)
    ap_eval.add_argument("--weights", type=str, required=True)
    ap_eval.add_argument("--max_batches", type=int, default=10)

    args = ap.parse_args()
    if args.cmd == "train":
        train_slow_module(
            log_dir=args.log_dir,
            out_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.cmd == "eval":
        evaluate_slow_module(
            log_dir=args.log_dir,
            weight_path=args.weights,
            max_batches=args.max_batches,
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
