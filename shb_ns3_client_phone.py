#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shb_ns3_client_phone.py —— 使用 unified_netgear 的 NetGearLike 封装发送数据
本版更新要点：
  - send_file_via_netgear：loss_rate 从 RTCP 获取；rtt_ms 从底层 PING/PONG 获取；
  - 'tooth' 分支：调用 net.get_slow_prediction() 使用慢模块最新建议的 fec_suggest；
  - 其它逻辑沿用你之前版本；不直接导入 PKT_*，通过 net.is_* 判断类型。
"""

import argparse
import csv
import math
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

# 统一封装（内部应路由到 NetGearUDP）
try:
    from vidgear.gears.unified_netgear import NetGearLike as NetGear
    from vidgear.gears.tooth_fast_module import ToothFastModuleRF
except Exception:
    NetGear = None  # type: ignore

try:
    from vidgear.gears.webrtc_interface import WebRtcPolicy
except Exception:
    WebRtcPolicy = None  # type: ignore

from ns3.sender import sendFrame  # 分片（可选）

UDP_BIND_ADDR: str = "0.0.0.0"
UDP_BIND_PORT: int = 5559
UDP_DST_ADDR: str = "114.212.86.152" # 127.0.0.1
UDP_DST_PORT: int = 5558

DEFAULT_MTU: int = 1500
MAX_PAYLOAD: int = 1400
APP_HEADER_BYTES: int = 29
NS3_DEFAULT_BITRATE_MBPS: float = 20.0

_ACK_FMT = "!id"
_ACK_SIZE = struct.calcsize(_ACK_FMT)

_GLOBAL_UDP = None
_UDP_RX_THREAD: Optional[threading.Thread] = None
_UDP_RX_TERMINATE = False

# === 新增：ACK/RTCP 统计（限频打印，避免 I/O 成为瓶颈）===
_ACK_FRAME_RECV = 0       # 帧级 ACK 计数
_ACK_PACKET_RECV = 0      # 包级 ACK 计数
_RTCP_RECV = 0            # RTCP 报告计数
_PONG_RECV = 0            # PONG 计数
_PRINT_EVERY = 50         # 每收到多少个再打印一次

_PER_FRAME: Dict[int, Dict[str, Any]] = {}
_PER_LOCK = threading.Lock()

_CLIENT_LOG = Path("./client_times.csv")
with _CLIENT_LOG.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "frame_id","bytes_total","total_ms","final_send_ms","final_prop_ms","res_elapsed_ms",
        "ping_rtt_ms","send_ser_measured_ms","send_ser_bw_nominal_ms","send_goodput_mbps",
        "prop_from_ack_minus_send_ms"
    ])

def _pf_update(fid: int, **kv: Any) -> None:
    with _PER_LOCK:
        rec = _PER_FRAME.setdefault(int(fid), {})
        rec.update(kv)

def _try_flush(fid: int) -> None:
    with _PER_LOCK:
        rec = _PER_FRAME.get(int(fid), {})
        if not rec:
            return
        bytes_total = rec.get("bytes_total")
        total_ms    = rec.get("total_ms")
        t0          = rec.get("t_send_start")
        t1          = rec.get("t_send_end")
        t_res_recv  = rec.get("t_res_recv")
        ping_rtt_ms = rec.get("ping_rtt_ms", None)
        ping_prop_ms= rec.get("ping_prop_ms", None)
        if (bytes_total is None) or (total_ms is None) or (t0 is None) or (t1 is None) or (t_res_recv is None):
            return

        # 发送/带宽估计
        send_ser_measured_ms = max(0.0, (float(t1) - float(t0)) * 1000.0)
        send_ser_bw_nominal_ms = (int(bytes_total) * 8.0) / (NS3_DEFAULT_BITRATE_MBPS * 1e6) * 1000.0
        dt_send = max(1e-9, float(t1) - float(t0))
        send_goodput_mbps = (int(bytes_total) * 8.0) / dt_send / 1e6

        # 传播估计（ACK - send）
        prop_from_ack_minus_send_ms = max(0.0, float(total_ms) - float(send_ser_measured_ms))

        # 组合（残差最小）
        send_candidates: List[Tuple[str, float]] = [
            ("send_measured", send_ser_measured_ms),
            ("send_bw_nominal", send_ser_bw_nominal_ms),
        ]
        prop_candidates: List[Tuple[str, float]] = [("prop_ack_minus_send", prop_from_ack_minus_send_ms)]
        if ping_prop_ms is not None:
            prop_candidates.append(("prop_ping_half_rtt", float(ping_prop_ms)))

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

        res_elapsed_ms = rec.get("res_elapsed_ms")
        if res_elapsed_ms is None:
            res_elapsed_ms = max(0.0, (float(rec.get("t_res_recv")) - float(t0)) * 1000.0)

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
        del _PER_FRAME[int(fid)]

# -------------------- UDP 入口 --------------------
def ensure_global_udp() -> Any:
    """创建/复用全局 UDP 会话（你的 NetGearLike 封装）。"""
    global _GLOBAL_UDP, _UDP_RX_THREAD, _UDP_RX_TERMINATE
    if _GLOBAL_UDP is not None:
        return _GLOBAL_UDP
    if NetGear is None:
        raise RuntimeError("未找到 unified_netgear.NetGearLike，请确认 vidgear 环境。")

    _GLOBAL_UDP = NetGear(
        address=UDP_DST_ADDR,
        port=UDP_DST_PORT,
        protocol="udp",
        receive_mode=False,
        local_bind_port = UDP_BIND_PORT,
        logging=True,
        mtu=DEFAULT_MTU,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
        slow_log_dir="./tmp/slow_logs_sender",            # 发送端日志
        slow_weights_path="./tmp/slow_module_weights.json",
        rtcp_interval_ms=100,
    )
    # 指定对端
    try:
        setattr(_GLOBAL_UDP, "_peer_addr", (UDP_DST_ADDR, UDP_DST_PORT))
    except Exception:
        pass
    try:
        if hasattr(_GLOBAL_UDP, "set_peer"):
            _GLOBAL_UDP.set_peer(UDP_DST_ADDR, UDP_DST_PORT)  # type: ignore
    except Exception:
        pass

    # 后台接收线程（专用、轻量、仅做分发与状态更新）
    _UDP_RX_TERMINATE = False
    _UDP_RX_THREAD = threading.Thread(target=_client_rx_loop, args=(_GLOBAL_UDP,), daemon=True)
    _UDP_RX_THREAD.start()
    return _GLOBAL_UDP

def get_last_rtt_ms() -> float:
    """返回最近 RTT（ms）：优先 median；退化用 ema。"""
    try:
        net = ensure_global_udp()
        st = net.get_rtt_stats() or {}
        v = st.get("median_ms") or st.get("ema_ms")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")

# -------------------- RTCP 拉取（用于发送函数中取最近一条） --------------------
def _pump_all_rtcp_and_get_latest(net: Any) -> Dict[str, Any]:
    """抽干 RTCP 队列并返回最近一条；若无则返回空 dict。"""
    last = None
    while True:
        try:
            rep = net.get_rtcp_report()
        except Exception:
            rep = None
        if rep is None:
            break
        last = rep
    return {} if last is None else dict(last)

# -------------------- WebRTC 策略 --------------------
def compute_webrtc_fec_rate_strict(loss_rate: float, rtt_ms: float, fps: float, bitrate_mbps: float) -> float:
    if WebRtcPolicy is None:
        return 0.0
    try:
        return float(WebRtcPolicy.compute_fec(loss_rate, rtt_ms, fps, bitrate_mbps))
    except Exception:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 0.0

# -------------------- 轻量事件处理：ACK/RTCP/RES/PONG --------------------
def _on_ack_frame(net: Any, data: bytes) -> None:
    """帧级 ACK：仅做轻处理与统计，禁止重 I/O。"""
    global _ACK_FRAME_RECV
    if len(data) < _ACK_SIZE:
        return
    try:
        fid, server_send_ts = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
        now = time.time()
        _pf_update(int(fid), t_ack_recv=float(now))
        rec = _PER_FRAME.get(int(fid), {})
        t0 = rec.get("t_send_start")
        if t0 is not None:
            total_ms = max(0.0, (now - float(t0)) * 1000.0)
            _pf_update(int(fid), total_ms=float(total_ms))
        # RTT（轻量获取）
        try:
            st = net.get_rtt_stats() or {}
            rtt_ms = st.get("median_ms") or st.get("ema_ms")
            if rtt_ms is not None:
                rtt_ms = float(rtt_ms)
                _pf_update(int(fid), ping_rtt_ms=rtt_ms, ping_prop_ms=rtt_ms/2.0)
        except Exception:
            pass
        _try_flush(int(fid))

        _ACK_FRAME_RECV += 1
        if (_ACK_FRAME_RECV % 1) == 0:
            # 限频打印，避免 I/O 阻塞
            print(f"[Client] ACK_FRAME received = {_ACK_FRAME_RECV}")
    except Exception:
        pass

def _on_ack_packet(net: Any, data: bytes) -> None:
    """包级 ACK：推进 P_ls 位图（轻量）"""
    global _ACK_PACKET_RECV
    if len(data) < _ACK_SIZE:
        return
    try:
        packet_id, server_send_ts = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
        try:
            net.udp_ack_packet(int(packet_id))
        except Exception:
            pass
        _ACK_PACKET_RECV += 1
    except Exception:
        pass

def _on_res(net: Any, data: bytes) -> None:
    """RES：按需解析（此处保持轻处理，避免 I/O）"""
    # 你可以在此设置标志位，让业务线程去异步解析更大的 JSON/图像
    return

def _on_rtcp(net: Any) -> None:
    """RTCP：不需要做事，netgear_udp 内部已经更新慢模块；这里只做计数。"""
    global _RTCP_RECV
    _RTCP_RECV += 1
    if (_RTCP_RECV % (_PRINT_EVERY * 2)) == 0:
        print(f"[Client] RTCP reports received = {_RTCP_RECV}")

def _on_pong() -> None:
    """PONG：仅计数"""
    global _PONG_RECV
    _PONG_RECV += 1

# -------------------- 专用接收线程（轻量抽水） --------------------
def _client_rx_loop(net: Any) -> None:
    """
    发送端的专用接收线程：持续抽干 net.recv()，避免 ACK/RTCP 堵在缓冲里。
    - 严禁在这里做重计算/磁盘 I/O
    - 仅做轻解析与状态更新
    """
    global _UDP_RX_TERMINATE
    while not _UDP_RX_TERMINATE:
        try:
            item = net.recv()
        except Exception:
            item = None
        if not item:
            # 小睡让出 GIL，避免忙等
            time.sleep(0.0005)
            continue

        pkt_type = int(item.get("pkt_type", -1))
        data: bytes = item.get("data", b"")

        # 分发（保持轻处理）
        if net.is_ack_frame(pkt_type):
            _on_ack_frame(net, data)
            continue

        if net.is_ack_packet(pkt_type):
            _on_ack_packet(net, data)
            continue

        if net.is_res(pkt_type):
            _on_res(net, data)
            continue

        if pkt_type == getattr(net, "PKT_RTCP_REPORT", -2):
            _on_rtcp(net)
            continue

        if pkt_type == getattr(net, "PKT_PONG", -3):
            _on_pong()
            continue

# -------------------- 核心：发送一帧（按要求修改） --------------------
def send_file_via_netgear(frame_data: bytes, frame_id: int, fps: float, fec_policy: str) -> None:
    """
    修改点：
      - loss_rate：从 RTCP 获取（抽干队列，取最近一条；若无则 0）
      - rtt_ms   ：通过 net.get_rtt_stats()（PING/PONG 估计）
      - 'tooth' 分支：从 net.get_slow_prediction() 取 fec_suggest
    """
    net = ensure_global_udp()

    # —— RTCP 最近一条 —— 
    last_rtcp = _pump_all_rtcp_and_get_latest(net)
    if last_rtcp:
        loss_rate = float(last_rtcp.get("lr", 0.0))
        la = float(last_rtcp.get("la", 0.0))
    else:
        loss_rate = 0.0
        la = 0.0

    # —— RTT（由 PING/PONG 得到；你的 RTCP 未携带 RTT 字段）——
    st = net.get_rtt_stats() or {}
    rtt_ms = st.get("median_ms") or st.get("ema_ms") or 0.0
    rtt_ms = float(rtt_ms)

    # —— 选择 FEC 策略 —— 
    if fec_policy == "no-fec":
        fec_rate = 0.0

    elif fec_policy == "webrtc":
        fec_rate = compute_webrtc_fec_rate_strict(loss_rate=loss_rate, rtt_ms=rtt_ms,
                                                  fps=float(fps), bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS)

    elif fec_policy == "hairpin":
        ddl_ms = 100.0
        webrtc_fec_rate = compute_webrtc_fec_rate_strict(loss_rate=loss_rate, rtt_ms=rtt_ms,
                                                         fps=float(fps), bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS)
        fec_rate = max(0.0, min(3.0, 1.0 * (rtt_ms / max(1.0, (ddl_ms - rtt_ms))) * webrtc_fec_rate))

    elif fec_policy == "tooth":
        # === 1) 从底层 slow-module 获得 (lr_f, la_f) ===
        slow_pred = net.get_slow_prediction()  # {'lr_f','la_f'}
        lr_f = float(slow_pred.get("lr_f", 0.0))
        la_f = float(slow_pred.get("la_f", 0.0))

        # === 2) 计算本帧的 fl（分片数） ===
        fl_pkts = max(1, math.ceil(len(frame_data) / MAX_PAYLOAD))

        # === 3) 调用 Tooth fast-module ===
        try:
            # 建议把权重路径放到配置/环境，这里用默认路径
            FAST_MODEL_PATH = "./fast_module.joblib"

            # 尝试复用已加载的模型，避免反复读盘
            _FAST = getattr(send_file_via_netgear, "_FAST_MODEL", None)
            if _FAST is None:
                _FAST = ToothFastModuleRF.load(FAST_MODEL_PATH)
                setattr(send_file_via_netgear, "_FAST_MODEL", _FAST)

            # 1) 主路径：使用 fast-module 预测 fec_rate
            fec_rate = float(_FAST.predict(lr_f=lr_f, la_f=la_f, fl_pkts=fl_pkts))

            # 2) 健壮性校验：若预测异常（NaN/Inf/负值），触发回退
            if not (fec_rate == fec_rate) or fec_rate in (float("inf"), float("-inf")) or fec_rate < 0.0:
                raise ValueError(f"fast-module predicted invalid fec={fec_rate}")

            # 3) 裁剪到工程允许范围（例如 0..3 倍冗余比）
            fec_rate = max(0.0, min(3.0, fec_rate))
            print(f"[TOOTH] lr_f={lr_f:.4f} la_f={la_f:.3f} fl_pkts={fl_pkts} -> fec={fec_rate:.3f}")

        except Exception as e:
            # —— 回退策略：改用现有的 WebRTC 估计函数 —— 
            # 说明：
            #  - 仍然用 slow-module 的 lr_f 作为“下一周期丢包率”的最佳估计；
            #  - rtt_ms/fps/bitrate_mbps 采用现有路径中的参数；
            # print(f"[TOOTH] fast-module not ready ({e}), fallback to WebRTC mapping")
            fec_rate = compute_webrtc_fec_rate_strict(
                loss_rate=lr_f,                  # 用预测的下一周期丢包率 lr_f
                rtt_ms=rtt_ms,                   # 来自 PING/PONG 的 RTT 估计
                fps=float(fps),                  # 当前发送帧率
                bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS
            )
            # 保险：裁剪
            fec_rate = max(0.0, min(3.0, float(fec_rate)))


    else:
        print("Error! No fec policy match!")
        fec_rate = 0.0

    # —— 分片与发送 —— 
    if sendFrame is not None:
        try:
            pkts: List[bytes] = sendFrame(frame_data, loss_rate=loss_rate, rtt_ms=rtt_ms,
                                          fec_rate=fec_rate, max_pay_load=MAX_PAYLOAD)  # type: ignore
        except Exception as e:
            print(f"[Send] sendFrame 失败: {e}")
            pkts = _fallback_sendFrame(frame_data, MAX_PAYLOAD)
    else:
        pkts = _fallback_sendFrame(frame_data, MAX_PAYLOAD)

    t0 = time.time()
    _pf_update(int(frame_id), t_send_start=float(t0))

    bytes_sum = 0
    # === 新增：批量发送时定期让出 GIL，保证接收线程及时运行 ===
    _yield_every = 64  # 每发送 64 个分片小让步
    for i, raw in enumerate(pkts):
        net.send(raw)  # 默认 DATA
        bytes_sum += (len(raw) + APP_HEADER_BYTES)
        if (i % _yield_every) == 0:
            # 让出调度，避免长时间占用 GIL
            time.sleep(0)

    t1 = time.time()
    _pf_update(int(frame_id), t_send_end=float(t1), bytes_total=int(bytes_sum))

    _try_flush(int(frame_id))
    if frame_id % 50 ==0:
        print(f"[Sender] frame_id={frame_id} bytes={len(frame_data)} "
          f"lr(rtpc)={loss_rate:.4f} rtt(ping)={rtt_ms:.1f} fec={fec_rate:.3f} sent_pkts={len(pkts)}")

# -------------------- 其它：回放/主函数（与原版一致） --------------------
def _fallback_sendFrame(frame_data: bytes, max_pay_load: int) -> List[bytes]:
    pkts: List[bytes] = []
    n = len(frame_data); off = 0
    while off < n:
        pkts.append(frame_data[off: off+max_pay_load])
        off += max_pay_load
    return pkts

def _load_sizes_from_csv(csv_path: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics csv 不存在: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        for i, row in enumerate(rdr):
            if not row:
                continue
            try:
                size = int(row[1])
            except Exception:
                continue
            try:
                fid = int(row[0])
            except Exception:
                fid = len(pairs)
            if size < 0:
                continue
            pairs.append((fid, size))
    if not pairs:
        raise RuntimeError(f"未能在 {csv_path} 解析出任何有效的 size")
    return pairs

def _random_payload(nbytes: int) -> bytes:
    return os.urandom(int(nbytes))

def replay_from_metrics(csv_path: str, interval_ms: float = 30.0, fec_policy: str = "webrtc") -> None:
    net = ensure_global_udp()
    sizes = _load_sizes_from_csv(csv_path)
    fps = 1000.0 / float(interval_ms)
    next_ts = time.time()
    for (fid, size) in sizes:
        payload = _random_payload(size)
        send_file_via_netgear(payload, frame_id=int(fid), fps=fps, fec_policy=fec_policy)
        next_ts += interval_ms / 1000.0
        sleep_s = next_ts - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_ts = time.time()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="metrics_serial.csv",
                        help="含 size(第二列) 的 CSV 路径")
    parser.add_argument("--interval_ms", type=float, default=30.0,
                        help="两帧间隔（毫秒）")
    parser.add_argument("--fec_policy", type=str, default="webrtc",
                        help="no-fec, webrtc, hairpin, tooth")
    args = parser.parse_args()
    try:
        replay_from_metrics(csv_path=args.metrics_csv, interval_ms=float(args.interval_ms), fec_policy=args.fec_policy)
    except KeyboardInterrupt:
        pass
    finally:
        while True:
            with _PER_LOCK:
                empty = (len(_PER_FRAME) == 0)
            if empty:
                break
            time.sleep(0.05)
        global _UDP_RX_TERMINATE
        _UDP_RX_TERMINATE = True
        time.sleep(0.05)

if __name__ == "__main__":
    main()
