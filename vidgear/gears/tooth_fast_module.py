# -*- coding: utf-8 -*-
"""
fast_module.py
==============
Tooth fast-module 的严格实现（Random Forest 回归；三路子模型 fast-A/B/C 平均）.

论文要点对齐：
- 输入: (lr_f, la_f, fl)  -> 输出: 每帧冗余决策（这里以 fec_ratio 表示冗余比）
- 模型: Random Forest (MSE), 多决策树平均
- 数据处理: 按 (lr, la, fl) 做均衡切分；三路子特征集 fast-A(la,fl), fast-B(lr,fl), fast-C(lr,la,fl)
- 推理: 三路模型的平均作为最终冗余决策
References:
  §3.4 Fast-module: Determining Per-frame Redundancy; Fig.14; fast-A/B/C; MSE criterion; RF averaging.
"""

from __future__ import annotations
import os, json, math, argparse, warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.utils import shuffle as sk_shuffle
    import joblib
    _SK_AVAILABLE = True
except Exception:
    _SK_AVAILABLE = False
    RandomForestRegressor = object  # type: ignore
    joblib = None  # type: ignore

# -----------------------
# 数据读写
# -----------------------
@dataclass
class FrameRecord:
    lr: float
    la: float
    fl_pkts: int
    label_ratio: Optional[float]  # fec_ratio ∈ [0, +)
    label_pkts: Optional[float]   # redundant_pkts (将根据 fl_pkts 转换为 ratio)

def _load_jsonl(dir_or_file: str) -> List[Dict[str, Any]]:
    files: List[str] = []
    if os.path.isdir(dir_or_file):
        for f in os.listdir(dir_or_file):
            if f.endswith(".jsonl"):
                files.append(os.path.join(dir_or_file, f))
    else:
        files = [dir_or_file]
    rows: List[Dict[str, Any]] = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows.sort(key=lambda x: x.get("ts_ms", 0))
    return rows

def _bytes_to_pkts(fl_bytes: int, max_payload: int) -> int:
    return max(1, math.ceil(float(fl_bytes) / float(max_payload)))

def _parse_frames(rows: List[Dict[str, Any]], default_payload: int = 1200) -> List[FrameRecord]:
    out: List[FrameRecord] = []
    for r in rows:
        try:
            lr = float(r.get("lr", r.get("lr_f", 0.0)))
            la = float(r.get("la", r.get("la_f", 0.0)))
            if "fl_pkts" in r:
                fl_pkts = int(r["fl_pkts"])
            elif "fl_bytes" in r:
                fl_pkts = _bytes_to_pkts(int(r["fl_bytes"]), default_payload)
            else:
                continue
            label_ratio = r.get("fec_ratio", None)
            label_pkts  = r.get("redundant_pkts", None)
            out.append(FrameRecord(lr=lr, la=la, fl_pkts=fl_pkts,
                                   label_ratio=(None if label_ratio is None else float(label_ratio)),
                                   label_pkts=(None if label_pkts is None else float(label_pkts))))
        except Exception:
            continue
    return out

# -----------------------
# 简单分箱均衡：把 (lr, la, fl_pkts) 按分位数切到若干 bin，并等量采样
# -----------------------
def _quantile_bins(x: List[float], q: List[float]) -> List[float]:
    xs = sorted([float(v) for v in x])
    res = []
    for p in q:
        k = max(0, min(len(xs)-1, int(round(p*(len(xs)-1)))))
        res.append(xs[k])
    return res

def _assign_bin(val: float, edges: List[float]) -> int:
    # edges 是递增分位阈值；返回所在区间编号
    for i, e in enumerate(edges):
        if val <= e:
            return i
    return len(edges)

def _balanced_indices(records: List[FrameRecord], bins: int = 4, max_per_bin: int = 2000) -> List[int]:
    if not records:
        return []
    lr_list = [r.lr for r in records]
    la_list = [r.la for r in records]
    fl_list = [float(r.fl_pkts) for r in records]

    q_edges = [0.25, 0.5, 0.75]  # 四分位
    lr_edges = _quantile_bins(lr_list, q_edges)
    la_edges = _quantile_bins(la_list, q_edges)
    fl_edges = _quantile_bins(fl_list, q_edges)

    buckets: Dict[Tuple[int,int,int], List[int]] = {}
    for idx, r in enumerate(records):
        b = (_assign_bin(r.lr, lr_edges),
             _assign_bin(r.la, la_edges),
             _assign_bin(float(r.fl_pkts), fl_edges))
        buckets.setdefault(b, []).append(idx)

    out_idx: List[int] = []
    for b, idxs in buckets.items():
        if len(idxs) > max_per_bin:
            out_idx.extend(idxs[:max_per_bin])
        else:
            out_idx.extend(idxs)
    return out_idx

# -----------------------
# RF 子模型容器
# -----------------------
class _RFWrap:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42):
        if not _SK_AVAILABLE:
            raise RuntimeError("scikit-learn 未安装，无法训练/加载 fast-module。")
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            criterion="squared_error",  # MSE
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.model.fit(X, y)

    def predict(self, X: List[List[float]]) -> List[float]:
        return list(self.model.predict(X))

# -----------------------
# 对外类：ToothFastModuleRF
# -----------------------
class ToothFastModuleRF:
    """
    三路子模型 fast-A/B/C：
      - fast-A: x=(la, fl_pkts)
      - fast-B: x=(lr, fl_pkts)
      - fast-C: x=(lr, la, fl_pkts)
    训练阶段：对原始样本做简单的分箱均衡，然后分别训练三路 RF。
    推理阶段：三路预测的简单平均（论文描述为“多树输出平均”，此处推广到“多子模型平均”以模拟 Fig.25 的设置）。
    """
    def __init__(self) -> None:
        self.A: Optional[_RFWrap] = None
        self.B: Optional[_RFWrap] = None
        self.C: Optional[_RFWrap] = None
        self.max_payload: int = 1200  # 仅用于把 fl_bytes 转 pkts

    # ---------- 训练 ----------
    def train_from_jsonl(self, dir_or_file: str,
                         n_estimators: int = 200,
                         max_depth: Optional[int] = None,
                         random_state: int = 42) -> None:
        rows = _load_jsonl(dir_or_file)
        recs = _parse_frames(rows, default_payload=self.max_payload)
        # 过滤掉没有监督信号的条目
        data: List[Tuple[List[float], List[float], List[float], float]] = []
        for r in recs:
            # 统一监督到 fec_ratio
            if r.label_ratio is not None:
                y = float(r.label_ratio)
            elif r.label_pkts is not None:
                y = float(r.label_pkts) / max(1, float(r.fl_pkts))
            else:
                continue
            xa = [float(r.la), float(r.fl_pkts)]
            xb = [float(r.lr), float(r.fl_pkts)]
            xc = [float(r.lr), float(r.la), float(r.fl_pkts)]
            data.append((xa, xb, xc, y))

        if not data:
            raise RuntimeError("训练集中没有有效的监督标签（fec_ratio / redundant_pkts）")

        # 简单均衡采样（按 (lr, la, fl) 分箱）
        keep_idx = _balanced_indices(recs, bins=4, max_per_bin=2000)
        if keep_idx and len(keep_idx) < len(recs):
            data = [data[i] for i in keep_idx if i < len(data)]

        Xa = [d[0] for d in data]; Xb = [d[1] for d in data]; Xc = [d[2] for d in data]
        y  = [d[3] for d in data]

        # 打乱
        if _SK_AVAILABLE:
            Xa, Xb, Xc, y = sk_shuffle(Xa, Xb, Xc, y, random_state=random_state)

        # 训练三路 RF
        self.A = _RFWrap(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.B = _RFWrap(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state+1)
        self.C = _RFWrap(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state+2)
        self.A.fit(Xa, y); self.B.fit(Xb, y); self.C.fit(Xc, y)

    # ---------- 推理（返回 fec_ratio） ----------
    def predict(self, lr_f: float, la_f: float, fl_bytes: Optional[int] = None, fl_pkts: Optional[int] = None) -> float:
        if self.A is None or self.B is None or self.C is None:
            warnings.warn("fast-module 未加载权重，返回 0.0", RuntimeWarning)
            return 0.0
        if fl_pkts is None:
            if fl_bytes is None:
                raise ValueError("predict 需要提供 fl_pkts 或 fl_bytes")
            fl_pkts = _bytes_to_pkts(int(fl_bytes), self.max_payload)
        xa = [[float(la_f), float(fl_pkts)]]
        xb = [[float(lr_f), float(fl_pkts)]]
        xc = [[float(lr_f), float(la_f), float(fl_pkts)]]
        ya = self.A.predict(xa)[0]; yb = self.B.predict(xb)[0]; yc = self.C.predict(xc)[0]
        y = (ya + yb + yc) / 3.0
        # 上层通常希望一个合理区间；这里不裁剪到具体上限，由上层映射到编码参数
        return float(max(0.0, y))

    # ---------- 存取 ----------
    def save(self, path: str) -> None:
        if not _SK_AVAILABLE:
            raise RuntimeError("scikit-learn 未安装，无法保存 fast-module。")
        obj = {
            "A": None if self.A is None else self.A.model,
            "B": None if self.B is None else self.B.model,
            "C": None if self.C is None else self.C.model,
            "max_payload": int(self.max_payload),
        }
        joblib.dump(obj, path)

    @staticmethod
    def load(path: str) -> "ToothFastModuleRF":
        if not _SK_AVAILABLE:
            raise RuntimeError("scikit-learn 未安装，无法加载 fast-module。")
        obj = joblib.load(path)
        mod = ToothFastModuleRF()
        mod.A = _RFWrap(); mod.B = _RFWrap(); mod.C = _RFWrap()
        mod.A.model = obj["A"]; mod.B.model = obj["B"]; mod.C.model = obj["C"]
        mod.max_payload = int(obj.get("max_payload", 1200))
        return mod

# -----------------------
# CLI
# -----------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="从逐帧 JSONL 样本训练 fast-module")
    p_tr.add_argument("--data", type=str, required=True, help="JSONL 文件或目录（逐帧样本）")
    p_tr.add_argument("--out", type=str, required=True, help="模型输出 .joblib 路径")
    p_tr.add_argument("--n_estimators", type=int, default=200)
    p_tr.add_argument("--max_depth", type=int, default=None)
    p_tr.add_argument("--random_state", type=int, default=42)

    p_pr = sub.add_parser("predict", help="加载模型对单样本做预测")
    p_pr.add_argument("--model", type=str, required=True)
    p_pr.add_argument("--lr_f", type=float, required=True)
    p_pr.add_argument("--la_f", type=float, required=True)
    p_pr.add_argument("--fl_bytes", type=int, default=None)
    p_pr.add_argument("--fl_pkts", type=int, default=None)

    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if args.cmd == "train":
        mod = ToothFastModuleRF()
        mod.train_from_jsonl(args.data, n_estimators=args.n_estimators,
                             max_depth=None if args.max_depth in (None, 0) else int(args.max_depth),
                             random_state=int(args.random_state))
        mod.save(args.out)
        print(f"[save] fast-module -> {args.out}")
    elif args.cmd == "predict":
        mod = ToothFastModuleRF.load(args.model)
        y = mod.predict(lr_f=float(args.lr_f), la_f=float(args.la_f),
                        fl_bytes=None if args.fl_bytes in (None, 0) else int(args.fl_bytes),
                        fl_pkts=None if args.fl_pkts in (None, 0) else int(args.fl_pkts))
        print(f"{y:.6f}")
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
