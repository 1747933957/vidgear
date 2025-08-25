#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
先运行shb_ns3_server_receiver.py和shb_ns3_client.py，然后运行shb_tcp_server_sender.py来发送内容
shb_ns3_client.py
功能总览：
1) 作为 TCP 接收端（ZeroMQ/REQ-REP 模式的“REP端”）：监听本机 0.0.0.0:5556，
   接收发送端发来的“文件内容”（承载于 np.uint8 一维数组）以及 message={"frame_id": ...}。
2) 解析出 frame_id，将字节内容立刻通过 UDP（你实现的 NetGearLike）发往 114.212.86.152:5558。
3) 后台线程从 114.212.86.152:5558 接收两类包：
   - PKT_ACK_PACKET：仅解析 send_ts(double, 秒)，更新 RTT（毫秒）
   - PKT_RES       ：打印回传结果（UTF-8）
"""

import argparse
import math
import struct
import time
import threading
import numpy as np

# ========== 1) 导入：TCP（ZeroMQ）与 UDP（你的实现）分别使用不同入口 ==========
from vidgear.gears.netgear import NetGear as TcpNetGear              # TCP/REQ-REP
from vidgear.gears.unified_netgear import NetGearLike as NetGear     # UDP（你的实现）

# ========== 2) 常量：地址端口与运行参数 ==========
# TCP 监听（接收“文件”）
TCP_BIND_ADDR = "0.0.0.0"   # 新变量：本机监听地址
TCP_BIND_PORT = 5556        # 新变量：本机监听端口（发送端会连接到这里）

# UDP 目的（发包 & 接收 ACK/RES）
UDP_DST_ADDR = "127.0.0.1" # "114.212.86.152"  # 新变量：UDP 目标地址（也是 ACK/RES 的来源）
UDP_DST_PORT = 5558              # 新变量：UDP 目标端口（也是 ACK/RES 的来源）

# 本地 UDP 绑定（作为发送源 & ACK/RES 的接收端口）
UDP_BIND_ADDR = "0.0.0.0"  # 新变量：本机 UDP 绑定地址
UDP_BIND_PORT = 5557       # 新变量：本机 UDP 绑定端口

# FEC 相关默认参数（保持你原逻辑接口）
NS3_DEFAULT_BITRATE_MBPS = 10.0
MAX_PAYLOAD = 1400
DEFAULT_FPS = 30.0

# ========== 3) UDP 协议的 pkt_type 常量 ==========
from vidgear.gears.netgear_udp import (
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_PACKET
)  # 仅取常量，不改其实现

# ========== 4) FEC 策略与 ns3 分包：保持原有调用 ==========
from vidgear.gears.netgear_webrtc import WebRtcPolicy
from ns3.sender import sendFrame

# ========== 5) 全局 UDP NetGear（你的实现） ==========
_GLOBAL_UDP = None  # 新变量：全局 UDP NetGear（发送与接收 ACK/RES）
def ensure_global_udp() -> NetGear:
    """
    创建/复用全局 UDP 会话（你的 NetGearLike）：
    - receive_mode=True：同一个 socket 用来收 ACK/RES
    - 显式设置 peer 为 (UDP_DST_ADDR, UDP_DST_PORT)
    """
    global _GLOBAL_UDP
    if _GLOBAL_UDP is not None:
        return _GLOBAL_UDP
    _GLOBAL_UDP = NetGear(
        address=UDP_BIND_ADDR,
        port=UDP_BIND_PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1500,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=655360,
        deliver_per_packet=True,  # 逐包派发
    )
    # 指定发送目标（同一个对象既 send 也 recv）
    _GLOBAL_UDP._peer_addr = (UDP_DST_ADDR, UDP_DST_PORT)  # type: ignore[attr-defined]
    _ensure_ack_res_thread(_GLOBAL_UDP)
    return _GLOBAL_UDP

def shutdown_global_udp():
    global _GLOBAL_UDP
    if _GLOBAL_UDP is not None:
        try:
            _GLOBAL_UDP.close()
        finally:
            _GLOBAL_UDP = None

# ========== 6) ACK/RES 后台接收与 RTT 维护 ==========
_ACK_FMT = "!d"                          # 仅有 send_ts: double(秒)
_ACK_SIZE = struct.calcsize(_ACK_FMT)
_RTT_MS_LAST = float("nan")              # 最近一次ACK计算出的RTT(ms)
_RTT_LOCK = threading.Lock()

def get_last_rtt_ms():
    with _RTT_LOCK:
        return None if math.isnan(_RTT_MS_LAST) else float(_RTT_MS_LAST)

def _ack_res_rx_loop(net: NetGear):
    """后台线程：收 PKT_ACK_PACKET 与 PKT_RES。"""
    global _RTT_MS_LAST
    while True:
        try:
            item = net.recv()
        except Exception:
            break
        if item is None:
            time.sleep(0.001)
            continue
        pkt_type = int(item.get("pkt_type", -1))
        data: bytes = item.get("data", b"")
        if pkt_type == PKT_ACK_PACKET:
            if len(data) >= _ACK_SIZE:
                try:
                    (send_ts,) = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
                    rtt = max(0.0, (time.time() - float(send_ts)) * 1000.0)
                    with _RTT_LOCK:
                        _RTT_MS_LAST = rtt
                except Exception:
                    pass
            continue
        if pkt_type == PKT_RES:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = f"<{len(data)} bytes>"
            print(f"[Client] PKT_RES: {text}")

def _ensure_ack_res_thread(net: NetGear):
    if not hasattr(_ensure_ack_res_thread, "_started"):
        t = threading.Thread(target=_ack_res_rx_loop, args=(net,), name="ACK-RES-RX", daemon=True)
        t.start()
        _ensure_ack_res_thread._started = True  # type: ignore[attr-defined]

# ========== 7) WebRTC FEC 查表（保持原有严格口径） ==========
_WEBRTC_POLICY = None
def _get_webrtc_policy() -> WebRtcPolicy:
    global _WEBRTC_POLICY
    if _WEBRTC_POLICY is not None:
        return _WEBRTC_POLICY
    _WEBRTC_POLICY = WebRtcPolicy(use_star=True, star_order=1, star_coeff=1.0, fec_table_path=None)
    return _WEBRTC_POLICY

def compute_webrtc_fec_rate_strict(*, loss_rate: float, rtt_ms: float, fps: float, bitrate_mbps: float) -> float:
    if not (0.0 <= loss_rate <= 1.0): raise ValueError(f"loss_rate 非法: {loss_rate}")
    if rtt_ms <= 0: raise ValueError(f"rtt_ms 非法: {rtt_ms}")
    if fps <= 0: raise ValueError(f"fps 非法: {fps}")
    if bitrate_mbps <= 0: raise ValueError(f"bitrate_mbps 非法: {bitrate_mbps}")
    policy = _get_webrtc_policy()
    ddl_left_ms = int(round(1000.0 / fps))
    _g, beta = policy.compute(
        cur_loss=float(loss_rate),
        bitrate_mbps=float(bitrate_mbps),
        ddl_left_ms=int(ddl_left_ms),
        rtt_ms=int(round(rtt_ms)),
        is_rtx=False,
        max_group_size=48,
    )
    if not (0.0 <= beta <= 1.0):
        raise RuntimeError(f"FEC 超界: {beta}")
    return float(beta)

# ========== 8) 保持原有的发送函数（统计->丢包率/RTT->FEC->ns3->send） ==========
def send_file_via_netgear(frame_data: bytes, frame_id: int, fps: float):
    """
    严格版发送函数（保持你原有流程）
    """
    net = ensure_global_udp()
    # 2) 读取网络统计 
    stats = net.get_stats()
    if not isinstance(stats, dict):
        raise RuntimeError("[Send] get_stats() 未返回 dict")
    # 兼容：若缺字段，补默认值
    stats.setdefault("total_packets_lost", 0)
    stats.setdefault("rtt_ms", 1)
    stats.setdefault("sent_packets", 0)
    if "total_packets_received" not in stats or "total_packets_lost" not in stats or "rtt_ms" not in stats:
        raise RuntimeError("[Send] get_stats() 缺少必要字段（total_packets_received/total_packets_lost/rtt_ms）")

    # 丢包率按你的口径：1 - recv/sent
    sp = int(stats.get("sent_packets", 0))
    resv = int(stats.get("total_packets_received", 0))
    if sp <= 0:
        print("[Send] sent_packets==0 或缺失，按默认丢包率0.2处理")
        loss_rate = 0.2
    else:
        loss_rate = 1.0 - (resv / float(sp))
        loss_rate = max(0.0, min(1.0, loss_rate))
    print(f"****************************resv:{resv},sp:{sp},loss_rate:{loss_rate}****************************")
    # RTT 来自 ACK
    rtt_from_ack = get_last_rtt_ms()
    if rtt_from_ack is None:
        print("[Send] 尚未从 ACK 收到 RTT；使用最小正数占位 1ms")
        rtt_ms = 1.0
    else:
        rtt_ms = float(rtt_from_ack)

    # 3) 计算 WebRTC 冗余率
    fec_rate = compute_webrtc_fec_rate_strict(
        loss_rate=loss_rate, rtt_ms=rtt_ms, fps=float(fps), bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS
    )
    # 4) 调 ns3 分包并发送
    pkts = sendFrame(frame_data, loss_rate=loss_rate, rtt_ms=rtt_ms, fec_rate=fec_rate, max_pay_load=MAX_PAYLOAD)
    for raw in pkts:
        # net._peer_addr = (UDP_DST_ADDR, UDP_DST_PORT)
        net.send(raw)
    print(f"[Sender] frame_id={frame_id} bytes={len(frame_data)} loss={loss_rate:.4f} rtt={rtt_ms:.2f}ms fec={fec_rate:.4f}")

# ========== 9) TCP（ZeroMQ/REQ-REP）接收：解析 frame_id 并转发 ==========
_TCP_RX = None  # 新变量：全局 TCP 接收端
def ensure_tcp_receiver() -> TcpNetGear:
    """
    在本机 0.0.0.0:5556 开启 REPLY 端（receive_mode=True）：
    - 开启 bidirectional_mode=True：这样 recv() 返回 (message, frame)
    - 发送端调用 NetGear(..., pattern=1, receive_mode=False).send(..., message=...)
    """
    global _TCP_RX
    if _TCP_RX is not None:
        return _TCP_RX
    _TCP_RX = TcpNetGear(
        address=TCP_BIND_ADDR,
        port=TCP_BIND_PORT,
        protocol="tcp",
        pattern=1,                   # REQ/REP
        receive_mode=True,           # 作为接收端(bind)
        logging=True,
        bidirectional_mode=True,     # 关键：使 recv() 返回 (message, frame)
        jpeg_compression=False,      # 关闭 JPEG，还原出 np.uint8 一维数组
        max_retries =0
    )
    return _TCP_RX

def tcp_receive_and_forward_loop(fps: float):
    """
    主循环：
    - 从 TCP NetGear.recv() 取数据（(message, frame) 或退化为 frame）
    - 从 message 解析 frame_id
    - 将 frame(np.uint8) -> bytes，调用 send_file_via_netgear(...) 通过 UDP 发出
    """
    net = ensure_tcp_receiver()
    ensure_global_udp()  # 确保 UDP 可用（也会启动 ACK/RES 线程）
    pre_time = time.time()
    while True:
        item = net.recv()
        if item is None:
            time.sleep(0.001)
            continue

        # 兼容：bidirectional_mode=True时，item为(message, frame)；否则可能是frame
        if isinstance(item, tuple) and len(item) == 2:
            message, frame = item
        else:
            message, frame = None, item
        # 解析 frame_id
        if isinstance(message, dict) and "frame_id" in message:
            frame_id = int(message["frame_id"])
        else:
            raise RuntimeError("[TCP] 接收的数据缺少 message.frame_id")

        # frame 是 np.uint8 的一维数组（发送端采用 np.frombuffer）
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise RuntimeError(f"[TCP] 非法的帧数据类型：{type(frame)} / {getattr(frame, 'dtype', None)}")

        frame_bytes = frame.tobytes()
        print(f"[TCP] 收到 frame_id={frame_id} {len(frame_bytes):,} bytes，准备通过 UDP 转发")
        cur_time = time.time()
        if cur_time - pre_time < 1.0 / fps:
            time.sleep(max(0.0, 1.0 / fps - (cur_time - pre_time)))
        send_file_via_netgear(frame_bytes, frame_id=frame_id, fps=fps)
        pre_time = time.time()

# ========== 10) main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="用于 FEC 查表的帧率")
    args = parser.parse_args()

    try:
        tcp_receive_and_forward_loop(fps=float(args.fps))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
