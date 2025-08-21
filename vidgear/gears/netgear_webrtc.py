#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
netgear_webrtc.py
-----------------
在不修改 netgear_udp.py 的前提下，提供一个“WebRTC 风格”的自适应 FEC 包装器：
- 按 ns-3 C++ 版本的 webrtc 策略（webrtc-policy.*）与查表法（webrtc-fec-array.*）实现：
  * 10s 窗口最大丢包 + 1s 窗口平均的 WebRtcLossFilter
  * get_fec_rate_webrtc(loss, group_size, bitrate) 的索引规则与 C 一致（从 webrtc-fec-array.h 解析加载数组）
- 仅调用 NetGearUDP 进行底层收发与分片，不重写 UDP
- 基于组的简易 XOR FEC（分块条带化），每个子块最多 1 个丢失可恢复

参考：
- webrtc-fec-array.cc/.h（表与索引）  -> Python 解析实现
- webrtc-policy.cc/.h（策略结构与窗口滤波） -> Python 等价实现
"""

import os
import re
import time
import math
import threading
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# 直接复用同目录下 UDP 实现（不重写 UDP）：
from .netgear_udp import NetGearUDP, PKT_DATA  # 只调用，不修改底层 UDP 代码

# =========================
# 1) WebRTC 策略相关 (Python 版)
# =========================

# ---- 新增常量：窗口长度（秒） ----
_LONG_WIN_SEC = 10.0   # 长窗口长度：10s（用于“最大值”）
_SHORT_WIN_SEC = 1.0   # 短窗口长度：1s（用于“平均”）

# ---- 新增类：WebRtcLossFilter（复刻 C++） ----
class WebRtcLossFilter:
    """
    复刻 ns-3 的丢包滤波器（webrtc-policy.cc）：
    - 短窗(1s)内累积样本后求平均，周期性地写入长窗(10s)
    - 返回长窗内平均值的“最大值”（WebRTC 做法）
    """
    def __init__(self, long_win: float = _LONG_WIN_SEC, short_win: float = _SHORT_WIN_SEC) -> None:
        # 新属性：双端队列，存 (loss, ts)   # long: 长窗口采样； short: 短窗口原始样本
        self._long: deque[Tuple[float, float]] = deque()
        self._short: deque[Tuple[float, float]] = deque()
        self._long_win = float(long_win)
        self._short_win = float(short_win)
        # 互斥：在多线程统计更新时保护
        self._lock = threading.Lock()

    def update_and_get(self, loss: float, now: Optional[float] = None) -> float:
        """
        参数:
          - loss(float): 当前瞬时丢包率，范围建议 [0, 1]
          - now(float): 时间戳（秒），默认 time.monotonic()
        返回:
          - 过去 long_win 内的平均丢包“最大值”
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            # 清理短窗中过期样本
            t_cut_short = now - self._short_win
            while self._short and self._short[0][1] < t_cut_short:
                self._short.popleft()
            self._short.append((float(loss), now))

            # 将短窗平均周期性写入长窗（每 ~1s）
            t_cut_long = now - self._long_win
            while self._long and self._long[0][1] < t_cut_long:
                self._long.popleft()
            if (not self._long) or (self._long[-1][1] < t_cut_short):
                # 计算短窗平均
                avg_loss = sum(x for x, _ in self._short) / max(1, len(self._short))
                self._long.append((avg_loss, now))
                self._short.clear()

            # 取长窗“最大”平均值
            if not self._long:
                return float(loss)
            return max(x for x, _ in self._long)


# ---- 新增：解析 webrtc-fec-array.h，构造 Python 数组 ----
def _load_webrtc_fec_table(h_path: Optional[str] = None) -> List[int]:
    """
    从 webrtc-fec-array.h 解析出 fec_array_webrtc[16269] 常量数组，返回 int 列表。
    优先使用与本文件同目录/上级目录的相对路径进行查找。
    """
    # 搜索候选路径（同级、上级、当前工作目录）
    candidates = []
    if h_path:
        candidates.append(h_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(here, "webrtc-fec-array.h"),
        os.path.join(os.path.dirname(here), "webrtc-fec-array.h"),
        os.path.join(os.getcwd(), "webrtc-fec-array.h"),
    ])

    content = None
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            break
    if content is None:
        raise FileNotFoundError(
            "无法找到 webrtc-fec-array.h，请将该文件放在与 netgear_webrtc.py 同级或上一级目录。"
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


# ---- 新增：与 C 一致的查表函数 ----
class _WebRtcFecLookup:
    """
    负责在 Python 中复刻 C 侧的 get_fec_rate_webrtc()：
    - 索引规则完全一致（参见 webrtc-fec-array.cc）
    - fec_rate = fec_array_webrtc[index] / group_size，并最终在上层 clamp 到 1
    """
    def __init__(self, table_path: Optional[str] = None) -> None:
        if table_path is None:
            table_path = 'webrtc-fec-array.h'
        self._arr: List[int] = _load_webrtc_fec_table(table_path)
        # 常量（来自 C 实现）：
        self._loss_start = 0.0
        self._loss_step = 0.01
        self._grp_min = 5
        self._grp_max = 55
        self._grp_step = 5
        self._br_min = 2.0
        self._br_max = 30.0
        self._br_step = 1.0
        # 维度：loss(0..50)=51, group(5..55 step5)=11, bitrate(2..30 step1)=29
        # index = loss_index * 319 + group_index * 29 + bitrate_index
        self._br_dim = 29
        self._grp_dim = 11
        self._loss_dim = 51

    def get_rate(self, loss: float, group_size: int, bitrate_mbps: float) -> float:
        # clamp & 量化（与 C 侧一致）
        loss = max(self._loss_start, min(0.5, float(loss)))
        loss_idx = int(round((loss - self._loss_start) / self._loss_step))

        g = float(group_size)
        g = max(self._grp_min, min(self._grp_max, g))
        # 四舍五入到 5 的倍数
        g = round((g - self._grp_min) / self._grp_step) * self._grp_step + self._grp_min
        grp_idx = int(round((g - self._grp_min) / self._grp_step))
        g_int = int(g)

        br = max(self._br_min, min(self._br_max, float(bitrate_mbps)))
        br_idx = int(round((br - self._br_min) / self._br_step))

        index = loss_idx * (self._grp_dim * self._br_dim) + grp_idx * self._br_dim + br_idx
        val = self._arr[index]
        return float(val) / float(g_int)


# ---- 新增：Star 策略（只实现线性一阶，可选开二阶/开方）----
def _star_linear(beta: float, ddl_left_ms: int, rtt_ms: int, coeff: float = 1.0) -> float:
    # 对齐 C 逻辑：min(coeff * beta * (rtt/ddl_left), 1.0)
    if ddl_left_ms <= 0:
        return min(1.0, coeff * beta)
    return min(1.0, coeff * beta * (float(rtt_ms) / float(ddl_left_ms)))


def _star_quadratic(beta: float, ddl_left_ms: int, rtt_ms: int) -> float:
    if ddl_left_ms <= 0:
        return min(1.0, 4.0 * beta)
    x = float(rtt_ms) / float(ddl_left_ms)
    return 4.0 * beta * x * x


def _star_sqrt(beta: float, ddl_left_ms: int, rtt_ms: int) -> float:
    if ddl_left_ms <= 0:
        return min(1.0, beta)
    x = float(rtt_ms) / float(ddl_left_ms)
    return beta * math.sqrt(2.0 * x)


# ---- 新增：策略包装 ----
class WebRtcPolicy:
    """
    Python 版 webrtc 策略：
    - loss 使用 WebRtcLossFilter(update_and_get)
    - fec_rate 使用 _WebRtcFecLookup 查表
    - 额外可选 Star 变体（线性/二次/开方），默认启用一阶线性（与论文中 DDL 相关调制一致）
    """
    def __init__(self, use_star: bool = True, star_order: int = 1, star_coeff: float = 1.0,
                 fec_table_path: Optional[str] = None) -> None:
        # 新属性：窗口滤波
        self._loss_filter = WebRtcLossFilter()
        # 新属性：查表器
        self._lookup = _WebRtcFecLookup(fec_table_path)
        # 新属性：Star 策略参数
        self._use_star = bool(use_star)
        self._star_order = int(star_order)
        self._star_coeff = float(star_coeff)
        # 新属性：最大组大小（对齐 C 代码：min(maxGroupSize, 48)）
        self._max_group = 48

    def compute(self, cur_loss: float, bitrate_mbps: float,
                ddl_left_ms: int, rtt_ms: int,
                is_rtx: bool = False,
                max_group_size: int = 48) -> Tuple[int, float]:
        """
        返回: (group_size, fec_rate)
        - group_size(int): 实际使用的组大小（<=48）
        - fec_rate(float): 建议的冗余率（会在上层 clamp 到 1.0）
        """
        gmax = min(int(max_group_size), self._max_group)
        if is_rtx:
            # 首次发送之外的补发，这里沿 C 代码约定：不再增长 fec
            return gmax, 0.0

        # 计算“WebRTC 风格”损失（10s 窗最大）
        loss_w = self._loss_filter.update_and_get(cur_loss, now=time.monotonic())
        # 基础 fec
        beta = self._lookup.get_rate(loss_w, gmax, bitrate_mbps)
        beta = min(1.0, max(0.0, beta))

        # DDL 相关的 Star 加成（可选）
        if self._use_star:
            if self._star_order == 1:
                beta = _star_linear(beta, ddl_left_ms, rtt_ms, coeff=self._star_coeff)
            elif self._star_order == 2:
                beta = min(1.0, _star_quadratic(beta, ddl_left_ms, rtt_ms))
            elif self._star_order == 0:
                beta = min(1.0, _star_sqrt(beta, ddl_left_ms, rtt_ms))
            else:
                # 未支持更高阶，退回基础 beta
                pass

        return gmax, min(1.0, max(0.0, beta))


# =========================
# 2) WebRTC-over-UDP 发送/接收（仅调用 NetGearUDP）
# =========================

# ---- 新增常量：FEC 内部头 ----
# 为不改动 netgear_udp 的 payload 结构，这里在 payload 最前面再塞一个极小“子头”标识 FEC 语义：
#   MAGIC(4s) | ver(u8) | kind(u8) | group_id(u32) | group_size(u16) | index(u16) | stride(u16)
# 含义：
#   - MAGIC: b"WFEC" 表示 FEC 相关；b"WDAT" 表示普通数据帧（包含组信息）
#   - ver:   版本号，=1
#   - kind:  0=DATA（普通数据），1=PARITY（校验帧）
#   - group_id: 组id（自增），同组的数据与校验共享该值
#   - group_size: 本组数据帧个数（不含校验）
#   - index: DATA 时是帧在组内的序号；PARITY 时是校验的序号
#   - stride: PARITY 条带化的“模数”，即有多少个 parity 子块（条带数）
#             我们将数据按 index%stride = parity_index 条带分组做 XOR，满足每条带最多 1 个丢失可恢复
import struct
_FEC_HDR_FMT = "!4sBBIHHH"
_FEC_HDR_SIZE = struct.calcsize(_FEC_HDR_FMT)
_MAGIC_DATA = b"WDAT"
_MAGIC_FEC  = b"WFEC"
_KIND_DATA  = 0
_KIND_FEC   = 1

class NetGearWebRTC:
    """
    在 NetGearUDP 之上做“webrtc 风格”的 FEC 编排与自适应控制。
    - 发送端：按组聚合 -> 生成 parity -> 逐帧调用 udp.send() 发出
    - 接收端：收集同组帧 -> 若每条带最多 1 个丢失，尝试 XOR 恢复 -> 以原始 DATA 顺序吐出帧
    - 底层 RTP/分片/RTCP/统计等，全部由 NetGearUDP 完成
    """

    def __init__(self,
                 address: str = "0.0.0.0",
                 port: Union[int, str] = 5556,
                 protocol: str = "webrtc",   # 新增：保持对外一致写法
                 receive_mode: bool = False,
                 logging: bool = True,
                 # 新增：策略 & 发送参数
                 fps: float = 30.0,
                 bitrate_mbps: float = 10.0,
                 ddl_left_ms: int = 100,     # 新增：DDL 剩余时间估计（毫秒），用于 Star 调制
                 star_order: int = 1,        # 新增：Star 一阶（=1），可选 0(sqrt)/2(quadratic)
                 star_coeff: float = 1.0,    # 新增：Star 系数
                 max_group_size: int = 48,   # 新增：组上限（与 C 对齐）
                 fec_table_path: Optional[str] = None,
                 **options: Any) -> None:

        # 新属性：底层 UDP（只把 protocol 改为 'udp'，其余参数透传）
        self._udp = NetGearUDP(address=address, port=port, protocol="udp",
                               receive_mode=receive_mode, logging=logging,
                               fps=fps, **options)

        # 新属性：策略
        self._policy = WebRtcPolicy(use_star=True, star_order=star_order,
                                    star_coeff=star_coeff, fec_table_path=fec_table_path)
        self._bitrate_mbps = float(bitrate_mbps)     # 发送码率估计（Mb/s），用于查表
        self._ddl_left_ms  = int(ddl_left_ms)        # 任务/帧的剩余 DDL 估计（ms）

        # 新属性：组装状态（发送侧）
        self._group_id = 0                           # 递增的 group id
        self._cur_group: List[bytes] = []            # 当前未 flush 的数据帧缓冲
        self._cur_group_limit = int(max_group_size)  # 当前允许的最大组大小（策略控制）
        self._fec_last_rate = 0.0                    # 最近一次策略给的 fec_rate
        self._fec_last_stride = 0                    # 最近一次条带数（== parity 个数）
        self._fps = float(fps)                       # 仅用于简单的时间节奏控制

        # 新属性：接收侧重组缓存
        # key = group_id -> {
        #   'g': group_size,           # 该组应有的数据帧数
        #   'stride': stride,          # 条带数（parity 个数）
        #   'data': dict{idx: bytes},  # 已收数据帧
        #   'par': dict{pidx: bytes},  # 已收校验帧
        #   't0': time.monotonic(),    # 组创建时间
        # }
        self._rx_groups: Dict[int, Dict[str, Any]] = {}

        # 互斥
        self._lock = threading.Lock()

        # 新属性：发送节奏（可选用 fps 驱动 flush）
        self._last_send_ts = time.monotonic()

    # ------------- 工具：构造/解析 FEC 子头 -------------
    @staticmethod
    def _pack_fec_hdr(magic: bytes, kind: int, gid: int, gsize: int, index: int, stride: int) -> bytes:
        return struct.pack(_FEC_HDR_FMT, magic, 1, int(kind) & 0xFF,
                           gid & 0xFFFFFFFF, gsize & 0xFFFF, index & 0xFFFF, stride & 0xFFFF)

    @staticmethod
    def _unpack_fec_hdr(buf: bytes) -> Tuple[bytes, int, int, int, int, int]:
        if len(buf) < _FEC_HDR_SIZE:
            raise ValueError("FEC header too short")
        magic, ver, kind, gid, gsize, index, stride = struct.unpack(_FEC_HDR_FMT, buf[:_FEC_HDR_SIZE])
        if ver != 1:
            raise ValueError("Unsupported FEC header version")
        return magic, kind, gid, gsize, index, stride

    # ------------- 发送端：对外接口 -------------
    def send(self, data: bytes, frame_id: Optional[int] = None, pkt_type: int = PKT_DATA) -> None:
        """
        对外保持与 NetGearUDP.send() 一致的签名（frame_id 可选）。
        实际行为：先缓存在“当前组”，当组满或到时间就 flush（含 parity）。
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("NetGearWebRTC.send 仅支持 bytes-like 对象。")

        with self._lock:
            self._cur_group.append(bytes(data))

            # 简单节奏：尽量按策略组大小聚合，且不超过策略允许的最大组
            now = time.monotonic()
            flush_by_size = len(self._cur_group) >= max(1, self._cur_group_limit)
            flush_by_time = (now - self._last_send_ts) >= max(0.0, 1.0 / max(1.0, self._fps))
            if flush_by_size or flush_by_time:
                self._flush_group()

    def _flush_group(self) -> None:
        """将当前组打包：先发 DATA，再发 PARITY。"""
        if not self._cur_group:
            return

        gid = self._group_id
        data_list = self._cur_group
        gsize = len(data_list)

        # 从底层 UDP 统计推测当前 RTT 与丢包（尽量复用 udp 的统计）：
        stats = self._udp.get_stats()
        # 新变量：cur_loss, rtt（用于策略）
        recv = stats.get("total_packets_received", 0)
        lost = stats.get("total_packets_lost", 0)
        total = max(1, recv + lost)
        cur_loss = float(lost) / float(total)
        rtt_ms = int(max(1.0, stats.get("rtt_ms", 50.0)))  # 如果测不到，就给个兜底

        # 通过策略得到 (group_size, fec_rate)
        gmax, beta = self._policy.compute(cur_loss, self._bitrate_mbps, self._ddl_left_ms, rtt_ms,
                                          is_rtx=False, max_group_size=self._cur_group_limit)
        # 实际 group_size 就用当前凑齐的 gsize（<= gmax）
        group_size = min(gsize, gmax)
        fec_cnt = int(round(beta * group_size))  # parity 个数（条带数）
        fec_cnt = max(0, min( min(group_size, 16), fec_cnt))  # 上限 16，防止极端开销

        self._fec_last_rate = beta
        self._fec_last_stride = fec_cnt

        # 1) 先发 DATA（每帧带 WDAT 子头）
        for i in range(group_size):
            subhdr = self._pack_fec_hdr(_MAGIC_DATA, _KIND_DATA, gid, group_size, i, fec_cnt)
            payload = subhdr + data_list[i]
            # 直接调用底层 UDP（不重写 UDP）
            self._udp.send(payload, frame_id=None, pkt_type=PKT_DATA)

        # 2) 生成 & 发送 PARITY（条带化）
        if fec_cnt > 0:
            # 以条带数 fec_cnt 为模，将 i%fec_cnt == k 的所有 data XOR 得到第 k 个 parity
            stripes: List[bytearray] = []
            max_len = 0
            for _ in range(fec_cnt):
                stripes.append(bytearray())
            for i in range(group_size):
                k = i % fec_cnt
                di = data_list[i]
                max_len = max(max_len, len(di))
                if len(stripes[k]) < len(di):
                    stripes[k].extend(b"\x00" * (len(di) - len(stripes[k])))
                # XOR
                for p in range(len(di)):
                    stripes[k][p] ^= di[p]
            # 发送 parity
            for k in range(fec_cnt):
                subhdr = self._pack_fec_hdr(_MAGIC_FEC, _KIND_FEC, gid, group_size, k, fec_cnt)
                payload = subhdr + bytes(stripes[k])
                self._udp.send(payload, frame_id=None, pkt_type=PKT_DATA)

        # 组完成，自增 gid，清空缓存
        self._group_id = (self._group_id + 1) & 0xFFFFFFFF
        self._cur_group.clear()
        self._last_send_ts = time.monotonic()

    # ------------- 接收端：对外接口 -------------
    def recv(self) -> Optional[Dict[str, Any]]:
        """
        返回与 NetGearUDP.recv() 相同风格的 dict:
          {"pkt_type": int, "frame_id": int, "data": bytes}
        但 data 为用户原始数据（不含 WDAT/WFEC 子头）。
        - 普通数据帧到齐直接吐出
        - 若遇到丢包且每条带最多 1 个丢失，尝试在超时时间内用 parity 恢复
        """
        item = self._udp.recv()
        if item is None:
            return None
        raw: bytes = item["data"]
        try:
            magic, kind, gid, gsize, idx, stride = self._unpack_fec_hdr(raw)
        except Exception:
            # 兼容非 webrtc-webrtc 对接时的原始数据
            return item

        body = raw[_FEC_HDR_SIZE:]

        # 更新组缓存
        g = self._rx_groups.get(gid)
        if g is None:
            g = {"g": gsize, "stride": stride, "data": {}, "par": {}, "t0": time.monotonic()}
            self._rx_groups[gid] = g

        if magic == _MAGIC_DATA and kind == _KIND_DATA:
            g["data"][idx] = body
        elif magic == _MAGIC_FEC and kind == _KIND_FEC:
            g["par"][idx] = body
        else:
            # 不识别，直接透传
            return item

        # 尝试立即恢复并吐出“就绪”的最小索引数据
        out = self._try_recover_and_pop(gid)
        if out is not None:
            return {"pkt_type": PKT_DATA, "frame_id": 0, "data": out}
        return None

    def _try_recover_and_pop(self, gid: int) -> Optional[bytes]:
        """在组 gid 中，如果 index=next_idx 的数据已经可用，则弹出；如缺失且可由 parity 恢复，则恢复后弹出。"""
        g = self._rx_groups.get(gid)
        if g is None:
            return None
        gsize: int = g["g"]
        stride: int = g["stride"]
        data: Dict[int, bytes] = g["data"]
        par: Dict[int, bytes] = g["par"]

        # 按序输出：寻找当前最小未输出的 index
        # 为简化，这里逐个 index 从 0..gsize-1 查找，找到就输出一个并删除；一次只吐一个
        for idx in range(gsize):
            if idx in data:
                # 弹出一个
                ret = data.pop(idx)
                if not data and len(par) == 0:
                    # 垃圾回收
                    self._rx_groups.pop(gid, None)
                return ret

            # 若缺失，看看所在条带能否恢复（每条带<=1个缺失）
            if stride > 0:
                band = idx % stride
                # 收集该 band 所有 data 与 parity
                has_parity = (band in par)
                if not has_parity:
                    continue
                # XOR 恢复：parity ^ (band 中其余 data) -> 缺失 data
                rec = bytearray(par[band])
                recoverable = True
                for j in range(idx % stride, gsize, stride):
                    if j == idx:
                        continue
                    dj = data.get(j)
                    if dj is None:
                        # 此条带有另一个缺失，无法恢复
                        recoverable = False
                        break
                    # 补齐长度
                    if len(rec) < len(dj):
                        rec.extend(b"\x00" * (len(dj) - len(rec)))
                    for p in range(len(dj)):
                        rec[p] ^= dj[p]
                if recoverable:
                    # 恢复成功，写回并删除 parity（避免重复恢复）
                    data[idx] = bytes(rec)
                    par.pop(band, None)
                    ret = data.pop(idx)
                    if not data and len(par) == 0:
                        self._rx_groups.pop(gid, None)
                    return ret

        # 超时清理：避免缓存爆炸
        now = time.monotonic()
        if now - g["t0"] > 5.0:  # 5s 超时
            self._rx_groups.pop(gid, None)
        return None

    # ------------- 资源回收 -------------
    def close(self) -> None:
        # 先尝试 flush 发送侧残留组
        with self._lock:
            self._flush_group()
        self._udp.close()

    # ------------- 统计（透传底层并附加策略快照） -------------
    def get_stats(self) -> Dict[str, Any]:
        s = self._udp.get_stats()
        s.update({
            "fec_last_rate": self._fec_last_rate,      # 新增：最近一次策略给出的冗余率
            "fec_last_stride": self._fec_last_stride,  # 新增：最近一次 parity 数
            "webrtc_group_limit": self._cur_group_limit,
        })
        return s
