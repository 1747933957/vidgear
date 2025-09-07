#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
webrtc_interface.py（精简版）
----------------------------
仅保留 compute_webrtc_fec_rate_strict() 所需的最小实现：
  - WebRtcLossFilter：10s 长窗取“最大”平均 + 1s 短窗平均
  - _load_webrtc_fec_table：解析 webrtc-fec-array.h 拿到整型表
  - _WebRtcFecLookup：与 C 实现一致的三维索引（loss × group × bitrate）
  - Star 调制：线性/二次/开方（默认线性）
  - WebRtcPolicy：对外 compute(...) 与静态 compute_fec(...)

注意：
  - 若找不到 webrtc-fec-array.h，本模块会在构造查表器时报错；
    上层 compute_webrtc_fec_rate_strict() 已用 try/except 返回 0.0 作兜底。
"""

import os
import re
import time
import math
import threading
from collections import deque
from typing import List, Optional, Tuple

# =========================
# 常量：窗口长度（秒）
# =========================
_LONG_WIN_SEC = 10.0   # 长窗口：10s（用于“最大值”统计）
_SHORT_WIN_SEC = 1.0   # 短窗口：1s（用于“平均”写入长窗）

# ==========================================================
# 1) WebRtcLossFilter —— 复刻 WebRTC/NS-3 的窗口滤波逻辑
# ==========================================================
class WebRtcLossFilter:
    """
    丢包滤波器：
      - 短窗(1s)累积样本后求平均，周期性写入长窗(10s)
      - 返回长窗内“平均值的最大值”（WebRTC 的做法）
    新增成员:
      - _long: deque[(loss_avg, ts)] 长窗平均序列
      - _short: deque[(loss, ts)]    短窗原始样本
      - _long_win/_short_win: 窗口时长（秒）
      - _lock: 线程互斥锁
    """
    def __init__(self, long_win: float = _LONG_WIN_SEC, short_win: float = _SHORT_WIN_SEC) -> None:
        self._long: deque[Tuple[float, float]] = deque()   # 长窗平均序列
        self._short: deque[Tuple[float, float]] = deque()  # 短窗原始样本
        self._long_win = float(long_win)                   # 长窗时长（秒）
        self._short_win = float(short_win)                 # 短窗时长（秒）
        self._lock = threading.Lock()                      # 互斥锁，保护并发更新

    def update_and_get(self, loss: float, now: Optional[float] = None) -> float:
        """
        参数:
          - loss(float): 当前瞬时丢包率，建议范围 [0, 1]
          - now(float): 时间戳（秒），默认 time.monotonic()
        返回:
          - 过去 long_win 内的“平均丢包的最大值”
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            # 1) 清理短窗中过期样本
            t_cut_short = now - self._short_win
            while self._short and self._short[0][1] < t_cut_short:
                self._short.popleft()
            self._short.append((float(loss), now))

            # 2) 每 ~1s 将短窗平均写入长窗；并清空短窗
            t_cut_long = now - self._long_win
            while self._long and self._long[0][1] < t_cut_long:
                self._long.popleft()
            if (not self._long) or (self._long[-1][1] < t_cut_short):
                avg_loss = sum(x for x, _ in self._short) / max(1, len(self._short))
                self._long.append((avg_loss, now))
                self._short.clear()

            # 3) 长窗内取“最大平均值”
            if not self._long:
                return float(loss)
            return max(x for x, _ in self._long)

# ==========================================================
# 2) 解析 webrtc-fec-array.h，构造 Python 数组
# ==========================================================
def _load_webrtc_fec_table(h_path: Optional[str] = None) -> List[int]:
    """
    从 webrtc-fec-array.h 解析出 fec_array_webrtc[...] 常量数组，返回 int 列表。
    查找路径顺序：显式传入 -> 与本文件同级 -> 上级目录 -> 当前工作目录。
    """
    candidates: List[str] = []
    if h_path:
        candidates.append(h_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(here, "webrtc-fec-array.h"),
        os.path.join(os.path.dirname(here), "webrtc-fec-array.h"),
        os.path.join(os.getcwd(), "webrtc-fec-array.h"),
    ])

    content: Optional[str] = None
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            break
    if content is None:
        raise FileNotFoundError(
            "无法找到 webrtc-fec-array.h，请将该文件放在与 webrtc_interface.py 同级或上一级目录。"
        )

    # 用正则提取大括号内的逗号分隔整数
    m = re.search(r"fec_array_webrtc\s*\[\s*\d+\s*\]\s*=\s*\{([^}]*)\}", content, re.S)
    if not m:
        raise ValueError("未能在 webrtc-fec-array.h 中找到 fec_array_webrtc 初始化数组。")

    nums = re.findall(r"(-?\d+)", m.group(1))
    arr = [int(x) for x in nums]
    if len(arr) == 0:
        raise ValueError("解析 fec_array_webrtc 失败：未解析出任何整数。")
    return arr

# ==========================================================
# 3) 与 C 一致的查表器：get_fec_rate_webrtc(...)
# ==========================================================
class _WebRtcFecLookup:
    """
    复刻 C 侧 get_fec_rate_webrtc() 的索引规则：
      - loss ∈ [0, 0.5]，步长 0.01
      - group_size ∈ {5,10,...,55}
      - bitrate_mbps ∈ {2,3,...,30}
    返回值：fec_rate = fec_array_webrtc[index] / group_size
    """
    def __init__(self, table_path: Optional[str] = None) -> None:
        if table_path is None:
            table_path = "webrtc-fec-array.h"
        # 新属性：查表数组
        self._arr: List[int] = _load_webrtc_fec_table(table_path)

        # 常量（与 C 实现一致）
        self._loss_start = 0.0
        self._loss_step = 0.01
        self._grp_min = 5
        self._grp_max = 55
        self._grp_step = 5
        self._br_min = 2.0
        self._br_max = 30.0
        self._br_step = 1.0

        # 维度与扁平索引（loss:51 × group:11 × bitrate:29 = 16269）
        self._br_dim = 29
        self._grp_dim = 11
        self._loss_dim = 51

    def get_rate(self, loss: float, group_size: int, bitrate_mbps: float) -> float:
        """
        参数:
          - loss: 丢包率（0~1），内部会 clamp 到 [0, 0.5]
          - group_size: 组大小（会量化到 {5,10,...,55}）
          - bitrate_mbps: 码率（会量化到 2..30）
        返回:
          - 基础冗余率 beta（尚未做 Star 调制）
        """
        # 1) 量化 loss -> 索引
        loss = max(self._loss_start, min(0.5, float(loss)))
        loss_idx = int(round((loss - self._loss_start) / self._loss_step))

        # 2) 量化 group_size -> 5 的倍数
        g = float(group_size)
        g = max(self._grp_min, min(self._grp_max, g))
        g = round((g - self._grp_min) / self._grp_step) * self._grp_step + self._grp_min
        grp_idx = int(round((g - self._grp_min) / self._grp_step))
        g_int = int(g)

        # 3) 量化 bitrate
        br = max(self._br_min, min(self._br_max, float(bitrate_mbps)))
        br_idx = int(round((br - self._br_min) / self._br_step))

        # 4) 计算扁平索引并取值
        index = loss_idx * (self._grp_dim * self._br_dim) + grp_idx * self._br_dim + br_idx
        val = self._arr[index]
        return float(val) / float(g_int)

# ==========================================================
# 4) Star 调制（DDL 相关）
# ==========================================================
def _star_linear(beta: float, ddl_left_ms: int, rtt_ms: int, coeff: float = 1.0) -> float:
    """线性一阶：min(coeff * beta * (rtt/ddl_left), 1.0)"""
    if ddl_left_ms <= 0:
        return min(1.0, coeff * beta)
    return min(1.0, coeff * beta * (float(rtt_ms) / float(ddl_left_ms)))

def _star_quadratic(beta: float, ddl_left_ms: int, rtt_ms: int) -> float:
    """二次：4*beta*(rtt/ddl)^2"""
    if ddl_left_ms <= 0:
        return min(1.0, 4.0 * beta)
    x = float(rtt_ms) / float(ddl_left_ms)
    return min(1.0, 4.0 * beta * x * x)

def _star_sqrt(beta: float, ddl_left_ms: int, rtt_ms: int) -> float:
    """开方：beta*sqrt(2*rtt/ddl)"""
    if ddl_left_ms <= 0:
        return min(1.0, beta)
    x = float(rtt_ms) / float(ddl_left_ms)
    return min(1.0, beta * math.sqrt(2.0 * x))

# ==========================================================
# 5) 对外策略类：WebRtcPolicy
# ==========================================================
class WebRtcPolicy:
    """
    WebRTC 风格 FEC 策略（精简版）：
      - loss 侧：WebRtcLossFilter.update_and_get()（10s 长窗最大）
      - fec 基础率：_WebRtcFecLookup.get_rate()
      - Star：根据 DDL 与 RTT 做调制（默认线性一阶）
    新增成员:
      - _loss_filter: 丢包滤波器
      - _lookup: FEC 查表器
      - _use_star/_star_order/_star_coeff: Star 策略控制
      - _max_group: 组大小上限（默认 48，与 C 侧常见设置一致）
    """
    def __init__(self, use_star: bool = True, star_order: int = 1, star_coeff: float = 1.0,
                 fec_table_path: Optional[str] = None) -> None:
        self._loss_filter = WebRtcLossFilter()                     # 丢包滤波
        self._lookup = _WebRtcFecLookup(fec_table_path)            # 查表器
        self._use_star = bool(use_star)                            # 是否启用 Star
        self._star_order = int(star_order)                         # Star 阶数：1=线性，2=二次，0=开方
        self._star_coeff = float(star_coeff)                       # Star 系数（线性）
        self._max_group = 48                                       # 组大小硬上限

    def compute(self, cur_loss: float, bitrate_mbps: float,
                ddl_left_ms: int, rtt_ms: int,
                is_rtx: bool = False,
                max_group_size: int = 48) -> Tuple[int, float]:
        """
        返回 (group_size, fec_rate)：
          - group_size(int): 实际使用的组大小（<=max_group_size<=48）
          - fec_rate(float): 建议冗余率（0..1）
        """
        gmax = min(int(max_group_size), self._max_group)
        if is_rtx:
            # RTX（二次发送）通常不再增加冗余
            return gmax, 0.0

        # 1) WebRTC 风格损失：10s 长窗的“最大平均值”
        loss_w = self._loss_filter.update_and_get(cur_loss, now=time.monotonic())

        # 2) 基础冗余率（查表得到）
        beta = self._lookup.get_rate(loss_w, gmax, bitrate_mbps)
        beta = min(1.0, max(0.0, beta))

        # 3) Star 调制（与 DDL/RTT 相关），默认线性
        if self._use_star:
            if self._star_order == 1:
                beta = _star_linear(beta, ddl_left_ms, rtt_ms, coeff=self._star_coeff)
            elif self._star_order == 2:
                beta = _star_quadratic(beta, ddl_left_ms, rtt_ms)
            elif self._star_order == 0:
                beta = _star_sqrt(beta, ddl_left_ms, rtt_ms)
            else:
                # 未支持更高阶，保持基础 beta
                pass

        return gmax, min(1.0, max(0.0, beta))

    # ===== 便捷入口：给 compute_webrtc_fec_rate_strict() 用 =====
    @staticmethod
    def compute_fec(loss_rate: float, rtt_ms: float, fps: float, bitrate_mbps: float,
                    fec_table_path: Optional[str] = None) -> float:
        """
        直接返回冗余率（不关心组大小）。参数含义：
          - loss_rate: 当前丢包率（0..1）
          - rtt_ms: RTT（毫秒）
          - fps: 帧率（用于估计每帧 DDL）
          - bitrate_mbps: 名义码率（Mb/s），查表用
          - fec_table_path: webrtc-fec-array.h 的位置（可选）
        约定：
          - DDL 剩余时间 ≈ round(1000/fps) 毫秒（每帧时间预算）
        """
        # === 新变量：ddl_left_ms（每帧时间预算，整数毫秒） ===
        ddl_left_ms = int(max(1.0, round(1000.0 / max(1e-3, float(fps)))))

        # 策略实例（默认启用线性 Star）
        pol = WebRtcPolicy(use_star=True, star_order=1, star_coeff=1.0, fec_table_path=fec_table_path)
        _, beta = pol.compute(cur_loss=float(loss_rate),
                              bitrate_mbps=float(bitrate_mbps),
                              ddl_left_ms=int(ddl_left_ms),
                              rtt_ms=int(round(float(rtt_ms))),
                              is_rtx=False,
                              max_group_size=48)
        return float(beta)
