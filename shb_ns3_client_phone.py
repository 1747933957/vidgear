#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shb_ns3_client.py / phone 变体 —— Metrics 回放 + 多方法估计发送/传播时延（择优）

功能总览：
1) 从 metrics_serial.csv 读取第二列 size（单位：字节），每隔 interval_ms（默认 30ms）发送一帧。
2) 帧内容随机生成（os.urandom），长度严格等于 size。
3) 通过你实现的 UDP NetGear（unified_netgear.NetGearLike 封装的 netgear_udp）发送：
   - 每帧发送前先发送一个 PING 包（结构: "!id" = [frame_id:int32, ping_ts:double]）
   - 服务器 shb_ns3_server_receiver.py 收到 PING 立即 PONG 回显，客户端据此计算 RTT 与 RTT/2 传播近似。
   - 服务器在“帧重组完成”时对该帧回一个 ACK（结构: "!id" = [frame_id:int32, server_send_ts:double]）。
4) 客户端维护每帧的统计表，计算多种发送/传播时延，并基于残差最小原则选出“最终”组合：
   - 发送时延（send_ms）候选：
       A) send_ser_measured_ms = (t_send_end - t_send_start) * 1000
       B) send_ser_bw_nominal_ms = bytes_total * 8 / (NS3_DEFAULT_BITRATE_MBPS*1e6) * 1000
     同时计算 goodput_mbps = bytes_total * 8 / (t_send_end - t_send_start) / 1e6
   - 传播时延（prop_ms）候选：
       A) prop_ping_ms = PING_RTT/2
       B) prop_from_ack_minus_send_ms = max(0, total_ms - send_ser_measured_ms)
   - 选择 (send, prop) 使 | total_ms - send - prop | 最小，且均非负。
5) 将所有方法的值 + 最终选型写入 client_times.csv，并在写入后删除该帧的内存条目，避免增长。

CSV 输出列（按顺序）：
    frame_id, bytes_total, total_ms,
    ping_rtt_ms, ping_prop_ms,
    send_ser_measured_ms, send_ser_bw_nominal_ms, send_goodput_mbps,
    prop_from_ack_minus_send_ms,
    final_send_ms, final_prop_ms, method_pair
"""

import argparse
import csv
import json
import math
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ========== 1) UDP 入口 ==========
# 说明：优先使用你封装的 NetGearLike；若不可用则给出最小兜底（不建议长期使用）
try:
    from vidgear.gears.unified_netgear import NetGearLike as NetGear  # 你的 UDP 封装
except Exception:
    NetGear = None  # type: ignore

# ========== 2) pkt_type 常量 ==========
# 来自你定制的 netgear_udp；若导入失败，定义兜底值（务必与服务端一致）
from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_FRAME, PKT_ACK_PACKET, PKT_PING, PKT_PONG

# ========== 3) FEC 策略与 ns3 分包：若环境具备则调用真实实现 ==========
# WebRTC 策略（可选，用于计算 fec_rate）
try:
    from vidgear.gears.netgear_webrtc import WebRtcPolicy  # 你之前的 Python 版策略封装
except Exception:
    WebRtcPolicy = None  # type: ignore

# ns3 分包函数（可选，若不可用则退化为本地分片）
from ns3.sender import sendFrame, _ack_packet, _report_stats

# ========== 4) 运行参数（按你先前的口径） ==========
UDP_BIND_ADDR: str = "0.0.0.0"       # 本地 UDP 绑定地址（收 ACK/PONG）
UDP_BIND_PORT: int = 5559            # 本地 UDP 端口
UDP_DST_ADDR: str = "114.212.86.152" # 服务器地址（同你之前）
UDP_DST_PORT: int = 5558             # 服务器端口（同你之前）

DEFAULT_MTU: int = 1500              # 以太网 MTU
MAX_PAYLOAD: int = 1400              # 应用层单包最大负载（避免 IP 分片；真实以你的实现为准）
APP_HEADER_BYTES: int = 29           # 【关键】你的应用层自定义头部长度；统计 bytes_total 时要包含
NS3_DEFAULT_BITRATE_MBPS: float = 20.0  # 名义链路带宽（用于理论发送时延的估计）

# ========== 5) ACK/RES/PING-PONG 负载格式 ==========
# ACK: "!id" = [frame_id:int32, server_send_ts:double秒]
# PING/PONG: "!id" = [frame_id:int32, ping_ts:double秒]
_ACK_FMT = "!id"
_ACK_SIZE = struct.calcsize(_ACK_FMT)
_PING_FMT = "!id"
_PING_SIZE = struct.calcsize(_PING_FMT)

# ========== 6) 全局 UDP 与后台接收线程 ==========
_GLOBAL_UDP = None
_UDP_RX_THREAD: Optional[threading.Thread] = None
_UDP_RX_TERMINATE = False

# 最近一次 RTT（毫秒）：保留你的口径，用于参考（非逐帧）
_RTT_MS_LAST: float = float("nan")
_RTT_LOCK = threading.Lock()

# ========== 7) 每帧统计表 ==========
# 结构：{fid: {t_send_start, t_send_end, bytes_total, total_ms, ping_rtt_ms, ping_prop_ms}}
_PER_FRAME: Dict[int, Dict[str, Any]] = {}
_PER_LOCK = threading.Lock()

# ========== 8) 输出 CSV 初始化 ==========
_CLIENT_LOG = Path("./client_times.csv")
with _CLIENT_LOG.open("w", newline="") as f:
    w = csv.writer(f)
    # 列顺序（精简/重排）：
    # frame_id, bytes_total, total_ms, final_send_ms, final_prop_ms, res_elapsed_ms,
    # ping_rtt_ms, send_ser_measured_ms, send_ser_bw_nominal_ms, send_goodput_mbps,
    # prop_from_ack_minus_send_ms
    w.writerow([
        "frame_id","bytes_total","total_ms","final_send_ms","final_prop_ms","res_elapsed_ms",
        "ping_rtt_ms","send_ser_measured_ms","send_ser_bw_nominal_ms","send_goodput_mbps",
        "prop_from_ack_minus_send_ms"
    ])

# ------------------------------------------------------------
# 工具函数：更新/尝试落盘
# ------------------------------------------------------------
def _pf_update(fid: int, **kv: Any) -> None:
    """线程安全更新某帧的统计条目。"""
    with _PER_LOCK:
        rec = _PER_FRAME.setdefault(int(fid), {})
        rec.update(kv)

def _try_flush(fid: int) -> None:
    """
    当必需字段齐备时，计算多方法并写入 CSV，然后删除该帧条目：
      必需：bytes_total, total_ms, t_send_start, t_send_end, t_res_recv
      选填：ping_rtt_ms/ping_prop_ms
    """
    with _PER_LOCK:
        rec = _PER_FRAME.get(int(fid), {})
        if not rec:
            return

        bytes_total = rec.get("bytes_total")
        total_ms    = rec.get("total_ms")
        t0          = rec.get("t_send_start")
        t1          = rec.get("t_send_end")
        t_res_recv = rec.get("t_res_recv")
        ping_rtt_ms = rec.get("ping_rtt_ms", None)
        ping_prop_ms= rec.get("ping_prop_ms", None)

        # 仍缺少关键字段则不落盘
        if (bytes_total is None) or (total_ms is None) or (t0 is None) or (t1 is None) or (t_res_recv is None):
            return

        # ===== 发送时延（多方法）=====
        # 2.1 实测串行发送时延
        send_ser_measured_ms = max(0.0, (float(t1) - float(t0)) * 1000.0)

        # 2.2 名义带宽推导的发送时延
        send_ser_bw_nominal_ms = (int(bytes_total) * 8.0) / (NS3_DEFAULT_BITRATE_MBPS * 1e6) * 1000.0

        # 2.3 实测发送窗口的有效吞吐
        dt_send = max(1e-9, float(t1) - float(t0))
        send_goodput_mbps = (int(bytes_total) * 8.0) / dt_send / 1e6

        # ===== 传播时延（多方法）=====
        prop_ping_ms = float(ping_prop_ms) if (ping_prop_ms is not None) else None
        prop_from_ack_minus_send_ms = max(0.0, float(total_ms) - float(send_ser_measured_ms))

        # ===== 组合择优（残差最小）=====
        send_candidates: List[Tuple[str, float]] = [
            ("send_measured", send_ser_measured_ms),
            ("send_bw_nominal", send_ser_bw_nominal_ms),
        ]
        prop_candidates: List[Tuple[str, float]] = [("prop_ack_minus_send", prop_from_ack_minus_send_ms)]
        if prop_ping_ms is not None:
            prop_candidates.append(("prop_ping_half_rtt", float(prop_ping_ms)))

        best = None
        for _, s_val in send_candidates:
            if s_val < 0:
                continue
            for _, p_val in prop_candidates:
                if p_val < 0:
                    continue
                residual = abs(float(total_ms) - float(s_val) - float(p_val))
                if (best is None) or (residual < best[0]):
                    best = (residual, float(s_val), float(p_val))
        if best is None:
            best = (0.0, float(send_ser_measured_ms), float(prop_from_ack_minus_send_ms))

        _, final_send_ms, final_prop_ms = best

        # 计算 RES 口径的耗时（已在 RES 分支内尽力计算，这里再次兜底避免 None）
        res_elapsed_ms = rec.get("res_elapsed_ms")
        if res_elapsed_ms is None:
            res_elapsed_ms = max(0.0, (float(t_res_recv) - float(t0)) * 1000.0)

        _, final_send_ms, final_prop_ms = best

        # ===== 写 CSV =====
        with _CLIENT_LOG.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                int(fid),
                int(bytes_total),
                int(round(float(total_ms))),
                round(float(final_send_ms), 3),
                round(float(final_prop_ms), 3),
                round(float(res_elapsed_ms), 3),
                ("" if ping_rtt_ms is None else round(float(ping_rtt_ms), 3)),
                round(float(send_ser_measured_ms), 3),
                round(float(send_ser_bw_nominal_ms), 3),
                round(float(send_goodput_mbps), 6),
                round(float(prop_from_ack_minus_send_ms), 3),
            ])


        # 删除该帧条目，避免内存增长
        del _PER_FRAME[int(fid)]

# ------------------------------------------------------------
# RTT 读写（仅作近似参考）
# ------------------------------------------------------------
def get_last_rtt_ms() -> float:
    """返回最近一次 ACK 或 PONG 推导的 RTT（毫秒），仅作参考。"""
    with _RTT_LOCK:
        return float(_RTT_MS_LAST)

# ------------------------------------------------------------
# FEC 策略：严格模式（若缺失策略实现则返回 0）
# ------------------------------------------------------------
def compute_webrtc_fec_rate_strict(loss_rate: float, rtt_ms: float, fps: float, bitrate_mbps: float) -> float:
    """
    从 WebRtcPolicy 计算 FEC，若策略不可用则返回 0.0。
    参数:
      - loss_rate: 丢包率（0~1）
      - rtt_ms:    RTT 毫秒
      - fps:       当前发送帧率
      - bitrate_mbps: 名义码率（用于策略，若需要）
    """
    if WebRtcPolicy is None:
        return 0.0
    try:
        return float(WebRtcPolicy.compute_fec(loss_rate, rtt_ms, fps, bitrate_mbps))
    except Exception:
        return 0.0

# ------------------------------------------------------------
# 本地分片（兜底）：若 ns3.sender.sendFrame 不可用，则按 MAX_PAYLOAD 直接切片
# ------------------------------------------------------------
def _fallback_sendFrame(frame_data: bytes, max_pay_load: int) -> List[bytes]:
    """仅用于兜底：把 frame_data 按 max_pay_load 切片返回列表。"""
    pkts: List[bytes] = []
    n = len(frame_data)
    off = 0
    while off < n:
        pkts.append(frame_data[off: off + max_pay_load])
        off += max_pay_load
    return pkts

# ------------------------------------------------------------
# 全局 UDP 初始化与后台接收线程
# ------------------------------------------------------------
def ensure_global_udp() -> Any:
    """
    创建/复用全局 UDP 会话（你的 NetGearLike）：
    - receive_mode=True：同一个 socket 用来收 ACK/RES
    - 显式设置对端为 (UDP_DST_ADDR, UDP_DST_PORT)：用于发送
    """
    global _GLOBAL_UDP, _UDP_RX_THREAD, _UDP_RX_TERMINATE
    if _GLOBAL_UDP is not None:
        return _GLOBAL_UDP

    if NetGear is None:
        raise RuntimeError("未找到 unified_netgear.NetGearLike，请确认 vidgear 环境。")

    _GLOBAL_UDP = NetGear(
        address=UDP_BIND_ADDR,
        port=UDP_BIND_PORT,
        protocol="udp",
        receive_mode=True,   # 需要接收 ACK/PONG/RES
        logging=True,
        mtu=DEFAULT_MTU,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )
    # 显式指定对端（不同实现提供不同方式，这里做两种兼容尝试）
    try:
        setattr(_GLOBAL_UDP, "_peer_addr", (UDP_DST_ADDR, UDP_DST_PORT))
    except Exception:
        pass
    try:
        if hasattr(_GLOBAL_UDP, "set_peer"):
            _GLOBAL_UDP.set_peer(UDP_DST_ADDR, UDP_DST_PORT)  # type: ignore
    except Exception:
        pass

    # 启动后台接收线程
    _UDP_RX_TERMINATE = False
    _UDP_RX_THREAD = threading.Thread(target=_ack_res_rx_loop, args=(_GLOBAL_UDP,), daemon=True)
    _UDP_RX_THREAD.start()
    return _GLOBAL_UDP

# ------------------------------------------------------------
# 后台接收线程：处理 PONG 与 ACK，并驱动 _PER_FRAME 的落盘
# ------------------------------------------------------------
def _ack_res_rx_loop(net: Any) -> None:
    """后台接收线程：解析 PONG/ACK/RES。"""
    global _RTT_MS_LAST
    while not _UDP_RX_TERMINATE:
        try:
            item = net.recv()
        except Exception:
            item = None

        if not item:
            time.sleep(0.001)
            continue

        pkt_type = int(item.get("pkt_type", -1))
        data: bytes = item.get("data", b"")

        # ---- PONG：记录 RTT 与 RTT/2 ----
        if pkt_type == PKT_PONG:
            try:
                fid, ping_ts = struct.unpack(_PING_FMT, data[:_PING_SIZE])
                now = time.time()
                ping_rtt_ms = max(0.0, (now - float(ping_ts)) * 1000.0)
                ping_prop_ms = ping_rtt_ms / 2.0
                _pf_update(int(fid), ping_rtt_ms=float(ping_rtt_ms), ping_prop_ms=float(ping_prop_ms))
                _try_flush(int(fid))
                with _RTT_LOCK:
                    _RTT_MS_LAST = ping_rtt_ms
            except Exception:
                pass
            continue

        # ---- ACK：每帧一次；由此计算 total_ms（ACK 到达时刻 - t_send_start）----
        if pkt_type == PKT_ACK_FRAME:
            if len(data) >= _ACK_SIZE:
                try:
                    fid, server_send_ts = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
                    now = time.time()
                    # 记录 ACK 到达时间；若已有 t_send_start，则可以得到 total_ms
                    _pf_update(int(fid), t_ack_recv=float(now))
                    rec = _PER_FRAME.get(int(fid), {})
                    t0 = rec.get("t_send_start")
                    if t0 is not None:
                        total_ms = max(0.0, (now - float(t0)) * 1000.0)
                        _pf_update(int(fid), total_ms=float(total_ms))
                    _try_flush(int(fid))
                except Exception:
                    pass
            continue
        if pkt_type == PKT_ACK_PACKET:
            if len(data) >= _ACK_SIZE:
                try:
                    fid, server_send_ts = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
                    print(f"[ACK] {fid}")
                    now = time.time()
                    #TODO: 调用接口
                    _ack_packet(int(fid))  # 来自 ns3.sender
                    # _report_stats()
                    # 维持最近一次 RTT（与 PING 不同口径，只作参考）
                    with _RTT_LOCK:
                        _RTT_MS_LAST = max(0.0, (now - float(server_send_ts)) * 1000.0)
                except Exception:
                    pass
            continue

        # ---- 其他：RES 文本等 ----
        if pkt_type == PKT_RES:
            try:
                if len(data) >= 8 and data[:4] == b'RSID':
                    _, fid = struct.unpack('!4sI', data[:8])  # 稳定获得 frame_id
                    now = time.time()
                    _pf_update(int(fid), t_res_recv=float(now))
                    # 计算 res_elapsed_ms = (t_res_recv - t_send_start) * 1000
                    rec = _PER_FRAME.get(int(fid), {})
                    t0 = rec.get('t_send_start')
                    if t0 is not None:
                        res_ms = max(0.0, (now - float(t0)) * 1000.0)
                        _pf_update(int(fid), res_elapsed_ms=float(res_ms))
                    # 尝试落盘（你之前的要求）
                    _try_flush(int(fid))

                    # 【可选】剩余部分是 JSON，解析仅用于日志/调试，不影响统计稳健性
                    try:
                        text = data[8:].decode('utf-8', errors='ignore')
                        # 这里不强制 json.loads，避免错误再次打断流程
                        print(f"[Client] PKT_RES(fid={fid}) JSON(len={len(text)}): {text[:120]}...")
                    except Exception as e:
                        print(f"[ERROR]{e}")
                        continue
            except Exception as e:
                print(f"[Client] PKT_RES parse failed (framed): {e}; len={len(data)}")
                continue

# ------------------------------------------------------------
# 读取 CSV 第二列 size，并产生 (frame_id, size) 列表
# ------------------------------------------------------------
def _load_sizes_from_csv(csv_path: str) -> List[Tuple[int, int]]:
    """
    读取 metrics_serial.csv，返回 [(frame_id, size), ...]
    - 优先用第一列 frame_idx 作为 frame_id；若第一列无法解析为整数，则回退为顺序编号。
    - 仅提取“第二列 size”（必须为整数），其他列忽略。
    """
    pairs: List[Tuple[int, int]] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics csv 不存在: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        for i, row in enumerate(rdr):
            if not row:
                continue
            # 跳过表头（第二列不是整数时）
            try:
                size = int(row[1])
            except Exception:
                continue
            # 解析 frame_id
            try:
                fid = int(row[0])
            except Exception:
                fid = len(pairs)  # 回退为顺序编号
            if size < 0:
                continue
            pairs.append((fid, size))
    if not pairs:
        raise RuntimeError(f"未能在 {csv_path} 解析出任何有效的 size")
    return pairs

# ------------------------------------------------------------
# 随机生成帧负载（长度严格等于 size）
# ------------------------------------------------------------
def _random_payload(nbytes: int) -> bytes:
    """
    随机生成指定大小的字节串：
    - 使用 os.urandom（速度快、无需额外依赖）
    """
    return os.urandom(int(nbytes))

# ------------------------------------------------------------
# 发送一帧（含 PING、ns3 分包、本地 bytes_total 统计与时间戳维护）
# ------------------------------------------------------------
def send_file_via_netgear(frame_data: bytes, frame_id: int, fps: float) -> None:
    """
    严格版发送函数（保持你原有流程）+【新增】按帧统计：
      - 发送前发送 PING（用于近似传播 RTT/2）
      - 记录 t_send_start / t_send_end
      - 统计 bytes_total（∑(len(pkt)+APP_HEADER_BYTES)）
      - ACK/PONG 线程驱动 total_ms / ping_* 的生成
    """
    net = ensure_global_udp()

    # 1) 丢包率、RTT 等（若实现提供 get_stats，可拿来参考）
    try:
        stats = net.get_stats() or {}
    except Exception:
        stats = {}
    # 丢包率估计：此处仅占位（若你在 net 中维护了更精确的口径，可替换）
    sent_pkts = int(stats.get("sent_packets", 0))
    recv_acks = int(stats.get("total_packets_received", 0))
    loss_rate = 0.0
    if sent_pkts > 0 and recv_acks >= 0:
        # 这里的“ACK 收到包数”不是数据包数，不能直接算丢包率；仅保留接口兼容
        loss_rate = 0.0
    rtt_ms = get_last_rtt_ms()
    if math.isnan(rtt_ms):
        rtt_ms = 0.0

    # 2) 计算 WebRTC 冗余率（保留口径，缺失实现则返回 0）
    fec_rate = compute_webrtc_fec_rate_strict(
        loss_rate=loss_rate, rtt_ms=rtt_ms, fps=float(fps), bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS
    )
    fec_rate = 0.2

    # 3) 分包：优先用 ns3.sender.sendFrame；否则使用本地兜底切片
    if sendFrame is not None:
        try:
            pkts: List[bytes] = sendFrame(frame_data, loss_rate=loss_rate, rtt_ms=rtt_ms, fec_rate=fec_rate, max_pay_load=MAX_PAYLOAD)  # type: ignore
        except Exception as e:
            print(f"[Send] send frame 失败: {e}")
            pkts = _fallback_sendFrame(frame_data, MAX_PAYLOAD)
    else:
        pkts = _fallback_sendFrame(frame_data, MAX_PAYLOAD)

    # 4) 发送 PING（含 frame_id 与 ping_ts），用于近似传播时延
    ping_ts = time.time()
    try:
        net.send(struct.pack(_PING_FMT, int(frame_id), float(ping_ts)), pkt_type=PKT_PING)
    except Exception as e:
        print(f"[Send] PING 失败: {e}")

    # 5) 记录“开始发送该帧数据”的时刻（PING 不计入总时延起点）
    t0 = time.time()
    _pf_update(int(frame_id), t_send_start=float(t0))

    # 6) 逐包发送并统计 bytes_total = ∑(len(pkt)+APP_HEADER_BYTES)
    bytes_sum = 0
    for raw in pkts:
        net.send(raw)  # 实际的帧数据分片（pkt_type 走你实现的默认 DATA）
        bytes_sum += (len(raw) + APP_HEADER_BYTES)

    # 7) 记录“结束发送该帧数据”的时刻
    t1 = time.time()
    _pf_update(int(frame_id), t_send_end=float(t1), bytes_total=int(bytes_sum))

    # 8) 尝试落盘（若 ACK/PONG 已经到达将触发写入并清理）
    _try_flush(int(frame_id))

    print(f"[Sender] frame_id={frame_id} bytes={len(frame_data)} fec={fec_rate:.4f} sent_pkts={len(pkts)}")

# ------------------------------------------------------------
# 回放：按 metrics_serial.csv 的第二列 size 逐帧发送（默认 30ms 一帧）
# ------------------------------------------------------------
def replay_from_metrics(csv_path: str, interval_ms: float = 30.0) -> None:
    """
    按 CSV 的第二列 size 回放发送：
    - 每隔 interval_ms（默认 30ms）发送一帧
    - 帧内容为随机字节，长度= size
    - 帧编号使用 CSV 第一列（或顺序回退）
    - FEC 策略求值使用 fps = 1000/interval_ms
    """
    ensure_global_udp()  # 初始化 UDP & 后台接收线程
    sizes = _load_sizes_from_csv(csv_path)
    fps = 1000.0 / float(interval_ms)
    next_ts = time.time()
    for (fid, size) in sizes:
        payload = _random_payload(size)
        send_file_via_netgear(payload, frame_id=int(fid), fps=fps)
        # 节拍控制
        next_ts += interval_ms / 1000.0
        sleep_s = next_ts - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # 赶不上节拍时，纠偏到当前时刻（避免漂移累积）
            next_ts = time.time()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="metrics_serial.csv",
                        help="含有 size(第二列) 的 CSV 路径")
    parser.add_argument("--interval_ms", type=float, default=30.0,
                        help="两帧之间的发送间隔（毫秒），默认 30ms")
    args = parser.parse_args()

    try:
        replay_from_metrics(csv_path=args.metrics_csv, interval_ms=float(args.interval_ms))
    except KeyboardInterrupt:
        pass
    finally:
        # 等待所有帧写入完成：只有在 _PER_FRAME 为空时才退出
        while True:
            with _PER_LOCK:
                empty = (len(_PER_FRAME) == 0)
            if empty:
                break
            time.sleep(0.05)
        # 可选：优雅结束后台线程
        global _UDP_RX_TERMINATE
        _UDP_RX_TERMINATE = True
        time.sleep(0.05)

if __name__ == "__main__":
    main()
