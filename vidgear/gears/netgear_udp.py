#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import time
import logging as log
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import struct
import random

logger = log.getLogger("NetGearUDP-RTP")
logger.setLevel(log.DEBUG)
if not logger.handlers:
    _h = log.StreamHandler()
    _f = log.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    _h.setFormatter(_f)
    logger.addHandler(_h)

# ===================== 常量与参数 =====================
DEFAULT_MTU = 1400               # UDP 整包上限（含 IP/UDP/RTP/负载）
RECV_QUEUE_MAXLEN = 32768        # 接收队列上限
REASS_TTL_SECONDS = 5.0          # 重组超时时间
CLOCK_RATE = 90000               # RTP 90kHz
DEFAULT_FPS = 30.0
RTCP_INTERVAL = 1.0

# RTP payload type：动态范围 96..127，选 96
PT_DYNAMIC = 96

# 自定义负载小头（payload header）
# 大端: pkt_type(u8) | flags(u8) | frame_id(u32) | total_frags(u16) | frag_idx(u16) | payload_len(u16)
PAYLOAD_HDR_FMT = "!BBIHHH"
PAYLOAD_HDR_SIZE = struct.calcsize(PAYLOAD_HDR_FMT)

# pkt_type 定义
PKT_DATA = 0      # client发送的普通数据（文件字节）
PKT_RES  = 1      # YOLO 检测结果
PKT_SV_DATA = 2   # server发送的普通数据（文件字节）
PKT_TERM = 3      # 终止（可选）

# =============== RTP/RTCP 工具 ===============
def _now_unix() -> float:
    return time.time()

def _unix_to_ntp(ts: float) -> Tuple[int, int]:
    ntp_epoch = 2208988800
    sec = int(ts) + ntp_epoch
    frac = int((ts - int(ts)) * (1 << 32)) & 0xFFFFFFFF
    return sec & 0xFFFFFFFF, frac

def pack_rtp_header(seq: int, timestamp: int, ssrc: int, marker: int, pt: int = PT_DYNAMIC,
                    cc: int = 0, x: int = 0, p: int = 0, v: int = 2) -> bytes:
    b0 = ((v & 0x3) << 6) | ((p & 0x1) << 5) | ((x & 0x1) << 4) | (cc & 0xF)
    b1 = ((marker & 0x1) << 7) | (pt & 0x7F)
    return struct.pack("!BBHII", b0, b1, seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)

def parse_rtp_header(buf: bytes) -> Tuple[int, int, int, int, int, int, int, int, int]:
    if len(buf) < 12:
        raise ValueError("RTP header too short")
    b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", buf[:12])
    v = (b0 >> 6) & 0x3
    p = (b0 >> 5) & 0x1
    x = (b0 >> 4) & 0x1
    cc = b0 & 0xF
    m = (b1 >> 7) & 0x1
    pt = b1 & 0x7F
    if cc != 0 or x != 0:
        raise ValueError("Unsupported RTP header (CC!=0 or X!=0)")
    return v, p, x, cc, m, pt, seq, ts, ssrc

# RTCP 类型
RTCP_SR = 200
RTCP_RR = 201
RTCP_SDES = 202
RTCP_BYE = 203
RTCP_APP = 204

def _rtcp_len_words(payload_len: int) -> int:
    return ((payload_len // 4) - 1)

def pack_rtcp_sr(ssrc: int, ntp_sec: int, ntp_frac: int, rtp_ts: int,
                 pkt_cnt: int, octet_cnt: int) -> bytes:
    body = struct.pack("!IIIIII",
                       ssrc & 0xFFFFFFFF,
                       ntp_sec & 0xFFFFFFFF, ntp_frac & 0xFFFFFFFF,
                       rtp_ts & 0xFFFFFFFF,
                       pkt_cnt & 0xFFFFFFFF, octet_cnt & 0xFFFFFFFF)
    length = _rtcp_len_words(len(body) + 4)
    return struct.pack("!BBH", 0x80, RTCP_SR, length) + body

def pack_rtcp_rr(ssrc: int, src_ssrc: int, fraction_lost: int, cum_lost: int,
                 ext_highest_seq: int, jitter: int, lsr: int, dlsr: int) -> bytes:
    body = struct.pack("!I", ssrc & 0xFFFFFFFF)
    cum_lost_24 = cum_lost & 0xFFFFFF
    fl = fraction_lost & 0xFF
    rb1 = struct.pack("!B", fl) + struct.pack("!I", cum_lost_24)[1:]
    rb2 = struct.pack("!I", ext_highest_seq & 0xFFFFFFFF)
    rb3 = struct.pack("!I", jitter & 0xFFFFFFFF)
    rb4 = struct.pack("!I", lsr & 0xFFFFFFFF)
    rb5 = struct.pack("!I", dlsr & 0xFFFFFFFF)
    block = struct.pack("!I", src_ssrc & 0xFFFFFFFF) + rb1 + rb2 + rb3 + rb4 + rb5
    length = _rtcp_len_words(len(body) + len(block) + 4)
    return struct.pack("!BBH", 0x81, RTCP_RR, length) + body + block

def pack_rtcp_bye(ssrc: int) -> bytes:
    body = struct.pack("!I", ssrc & 0xFFFFFFFF)
    length = _rtcp_len_words(len(body) + 4)
    return struct.pack("!BBH", 0x80, RTCP_BYE, length) + body

def is_rtcp_packet(buf: bytes) -> bool:
    if len(buf) < 2:
        return False
    pt = buf[1]
    return 200 <= pt <= 204

# ======================= 类实现 =======================
class NetGearUDP:
    """
    轻量 RTP/RTCP UDP：
      - receive_mode=False: 以“主动发送”为主（也能收 RTCP）
      - receive_mode=True : 以“主动接收”为主（也能回 RR，并允许 send 回对端）
    仅支持 bytes/bytearray/memoryview。
    关键扩展：
      * 支持 pkt_type（PKT_DATA/PKT_RES/PKT_TERM）
      * 支持显式 frame_id
      * recv() 返回 dict: {"pkt_type", "frame_id", "data"}
    """

    def __init__(
        self,
        address: str = "0.0.0.0",
        port: Union[int, str] = 5556,
        protocol: str = "udp",
        receive_mode: bool = False,
        logging: bool = True,
        **options
    ) -> None:

        self._logging = bool(logging)
        self._addr = address
        self._port = int(port)
        if protocol.lower() != "udp":
            raise ValueError("NetGearUDP 仅支持 protocol='udp'。")

        # 选项
        self._mtu = int(options.get("mtu", DEFAULT_MTU))
        self._fps = float(options.get("fps", DEFAULT_FPS))
        self._recv_buf_size = int(options.get("recv_buffer_size", 16 * 1024 * 1024))
        self._send_buf_size = int(options.get("send_buffer_size", 16 * 1024 * 1024))
        self._queue_maxlen = int(options.get("queue_maxlen", RECV_QUEUE_MAXLEN))

        # RTP 基本参数
        self._ssrc = random.getrandbits(32)
        self._seq = random.getrandbits(16)
        self._ts = random.getrandbits(32)
        self._ts_step = int(CLOCK_RATE / max(1.0, self._fps))

        # 发送统计
        self._sent_packets = 0
        self._sent_octets = 0

        # 接收统计
        self._recv_mode = bool(receive_mode)
        self._queue: deque = deque(maxlen=self._queue_maxlen)

        # 重组缓存: key=(ssrc, frame_id) -> {'t0','total','chunks','count','pkt_type'}
        self._frames: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # 序号/抖动
        self._base_seq: Optional[int] = None
        self._max_seq: int = -1
        self._cycles: int = 0
        self._received_pkts: int = 0
        self._expected_prior: int = 0
        self._received_prior: int = 0
        self._jitter: float = 0.0
        self._transit_prev: Optional[float] = None

        # 对端
        self._peer_addr: Optional[Tuple[str, int]] = None
        self._peer_ssrc: Optional[int] = None

        # LSR/DLSR/RTT
        self._last_sr_mid32: int = 0
        self._last_sr_arrival_ntp32: int = 0
        self._rtt_ms: float = -1.0

        # 额外统计
        self._total_frames_received = 0
        self._total_packets_received = 0
        self._total_packets_lost = 0
        self._last_stats_time = time.time()

        # 套接字
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf_size)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_buf_size)
            self._sock.setblocking(False)
        except Exception as e:
            self._logging and logger.warning(f"Socket option setting failed: {e}")

        self._terminate = False

        if self._recv_mode:
            self._sock.bind((self._addr, self._port))
            self._logging and logger.info(f"[RTP/RTCP] Bind at {self._addr}:{self._port}")
            self._rx_thread = threading.Thread(target=self._io_loop, name="RTP-RX", daemon=True)
            self._rx_thread.start()
            self._rtcp_thread = threading.Thread(target=self._rtcp_rr_loop, name="RTCP-RR", daemon=True)
            self._rtcp_thread.start()
            self._stats_thread = threading.Thread(target=self._stats_loop, name="Stats", daemon=True)
            self._stats_thread.start()
        else:
            self._peer_addr = (self._addr, self._port)
            self._logging and logger.info(f"[RTP/RTCP] Send to {self._addr}:{self._port}")
            self._rx_thread = threading.Thread(target=self._io_loop, name="RTP-SR-RR-RX", daemon=True)
            self._rx_thread.start()
            self._rtcp_thread = threading.Thread(target=self._rtcp_sr_loop, name="RTCP-SR", daemon=True)
            self._rtcp_thread.start()

        self._max_rtp_payload = max(200, self._mtu - 28 - 12)
        self._frame_id = 0  # 本端默认自增 frame_id（若 send 未显式指定）

    # ======================= IO/RTCP 线程 =======================
    def _io_loop(self) -> None:
        while not self._terminate:
            try:
                import select
                ready = select.select([self._sock], [], [], 0.1)
                if not ready[0]:
                    continue
                pkt, peer = self._sock.recvfrom(65535)
                self._total_packets_received += 1
            except socket.error as e:
                if getattr(e, "errno", None) in (socket.EAGAIN, socket.EWOULDBLOCK):
                    time.sleep(0.0001)
                    continue
                elif not self._terminate:
                    logger.error(f"recv error: {e}")
                continue
            except Exception as e:
                if not self._terminate:
                    logger.error(f"recv error: {e}")
                break

            if self._peer_addr is None:
                self._peer_addr = peer

            try:
                if is_rtcp_packet(pkt):
                    self._handle_rtcp(pkt)
                else:
                    self._handle_rtp(pkt)
            except Exception as e:
                self._logging and logger.warning(f"parse error: {e}")

    def _stats_loop(self) -> None:
        while not self._terminate:
            time.sleep(5.0)
            if self._recv_mode:
                current_time = time.time()
                self._last_stats_time = current_time
                self._gc_stale_frames()
                logger.info(f"[Stats] Frames received: {self._total_frames_received}, "
                            f"Packets received: {self._total_packets_received}, "
                            f"Queue size: {len(self._queue)}, "
                            f"Frames in progress: {len(self._frames)}")

    def _rtcp_sr_loop(self) -> None:
        while not self._terminate:
            time.sleep(RTCP_INTERVAL)
            if self._peer_addr is None:
                continue
            ntp_sec, ntp_frac = _unix_to_ntp(_now_unix())
            sr = pack_rtcp_sr(self._ssrc, ntp_sec, ntp_frac, self._ts & 0xFFFFFFFF,
                              self._sent_packets, self._sent_octets)
            try:
                self._sock.sendto(sr, self._peer_addr)
            except Exception:
                pass

    def _rtcp_rr_loop(self) -> None:
        while not self._terminate:
            time.sleep(RTCP_INTERVAL)
            if self._peer_addr is None or self._peer_ssrc is None:
                continue
            rr = self._build_rr()
            if rr is None:
                continue
            try:
                self._sock.sendto(rr, self._peer_addr)
            except Exception:
                pass

    # ======================= RTP 处理 =======================
    def _handle_rtp(self, pkt: bytes) -> None:
        try:
            v,p,x,cc,m,pt,seq,ts,ssrc = parse_rtp_header(pkt)
        except ValueError:
            return

        if v != 2 or pt != PT_DYNAMIC:
            return

        arrival = _now_unix()
        payload = pkt[12:]

        self._peer_ssrc = ssrc

        if self._recv_mode:
            self._update_seq_stats(seq)
            self._update_jitter(arrival, ts)

        if len(payload) < PAYLOAD_HDR_SIZE:
            return

        try:
            pkt_type, flags, frame_id, total_frags, frag_idx, payload_len = struct.unpack(
                PAYLOAD_HDR_FMT, payload[:PAYLOAD_HDR_SIZE]
            )
        except struct.error:
            return

        if payload_len > len(payload) - PAYLOAD_HDR_SIZE:
            return

        frag = payload[PAYLOAD_HDR_SIZE:PAYLOAD_HDR_SIZE+payload_len]
        if len(frag) != payload_len:
            return

        if pkt_type == PKT_TERM:
            # 可按需处理终止信号
            return

        # 重组
        key = (ssrc, frame_id)
        fr = self._frames.get(key)
        if fr is None:
            fr = dict(t0=arrival, total=int(total_frags), chunks={}, count=0, pkt_type=int(pkt_type))
            self._frames[key] = fr
        fr["total"] = max(fr["total"], int(total_frags))
        if frag_idx not in fr["chunks"]:
            fr["chunks"][int(frag_idx)] = frag
            fr["count"] += 1

        # 收齐则组装
        if fr["total"] > 0 and fr["count"] >= fr["total"]:
            total = fr["total"]
            if all(i in fr["chunks"] for i in range(total)):
                data = b"".join(fr["chunks"][i] for i in range(total))
                try:
                    if len(self._queue) < self._queue_maxlen:
                        # 关键：把 pkt_type 和 frame_id 一起交给上层
                        self._queue.append({
                            "pkt_type": fr["pkt_type"],
                            "frame_id": int(frame_id),
                            "data": data,
                        })
                        self._total_frames_received += 1
                except Exception:
                    pass
                self._frames.pop(key, None)

        # 垃圾回收
        now = _now_unix()
        if len(self._frames) > 10 or any(now - x["t0"] > REASS_TTL_SECONDS for x in self._frames.values()):
            self._gc_stale_frames()

    def _gc_stale_frames(self) -> None:
        now = _now_unix()
        stale = [k for k, fr in self._frames.items() if now - fr["t0"] > REASS_TTL_SECONDS]
        if stale:
            self._logging and logger.info(f"Cleaning {len(stale)} stale frames (>{REASS_TTL_SECONDS}s)")
        for k in stale:
            fr = self._frames.pop(k, None)
            if fr:
                self._total_packets_lost += 1

    def _update_seq_stats(self, seq: int) -> None:
        if self._base_seq is None:
            self._base_seq = seq
            self._max_seq = seq
            self._cycles = 0
            self._received_pkts = 1
            self._expected_prior = 0
            self._received_prior = 0
            return
        if seq < self._max_seq and self._max_seq - seq > 30000:
            self._cycles += 1
        if seq > self._max_seq:
            self._max_seq = seq
        self._received_pkts += 1

    def _update_jitter(self, arrival: float, rtp_ts: int) -> None:
        send_time = rtp_ts / float(CLOCK_RATE)
        transit = arrival - send_time
        if self._transit_prev is None:
            self._transit_prev = transit
            return
        d = abs(transit - self._transit_prev)
        self._transit_prev = transit
        self._jitter += (d - self._jitter) / 16.0

    # ======================= RTCP 处理/构造 =======================
    def _handle_rtcp(self, pkt: bytes) -> None:
        if len(pkt) < 8:
            return
        _v_p_count, pt, _length = struct.unpack("!BBH", pkt[:4])
        if pt == RTCP_SR:
            if len(pkt) < 28:
                return
            ssrc, ntp_sec, ntp_frac, rtp_ts, pkt_cnt, octet_cnt = struct.unpack("!IIIIII", pkt[4:28])
            if self._recv_mode:
                self._last_sr_mid32 = ((ntp_sec & 0xFFFF) << 16) | ((ntp_frac >> 16) & 0xFFFF)
                now_sec, now_frac = _unix_to_ntp(_now_unix())
                self._last_sr_arrival_ntp32 = ((now_sec & 0xFFFF) << 16) | ((now_frac >> 16) & 0xFFFF)
        elif pt == RTCP_RR:
            if len(pkt) < 8 + 24 + 4:
                return
            _rcv_ssrc, = struct.unpack("!I", pkt[4:8])
            rb = pkt[8:8+24]
            if len(rb) < 24:
                return
            _fraction = rb[0]
            cum_lost = int.from_bytes(rb[1:4], "big", signed=False)
            ext_seq, jitter, lsr, dlsr = struct.unpack("!IIII", rb[4:24])
            if lsr != 0:
                now_sec, now_frac = _unix_to_ntp(_now_unix())
                A = ((now_sec & 0xFFFF) << 16) | ((now_frac >> 16) & 0xFFFF)
                rtt_units = (A - lsr - dlsr) & 0xFFFFFFFF
                self._rtt_ms = (rtt_units / 65536.0) * 1000.0
                self._logging and logger.info(f"[RTCP] RTT≈{self._rtt_ms:.2f}ms jitter={jitter} lost(cum)={cum_lost}")

    def _build_rr(self) -> Optional[bytes]:
        if self._base_seq is None or self._peer_ssrc is None:
            return None
        ext_max = (self._cycles << 16) + self._max_seq
        expected = (ext_max - self._base_seq) + 1
        lost = expected - self._received_pkts
        if lost < 0:
            lost = 0
        expected_interval = expected - self._expected_prior
        self._expected_prior = expected
        received_interval = self._received_pkts - self._received_prior
        self._received_prior = self._received_pkts
        lost_interval = expected_interval - received_interval
        if expected_interval <= 0 or lost_interval < 0:
            fraction = 0
        else:
            fraction = int((lost_interval << 8) / expected_interval) & 0xFF
        jitter_int = int(self._jitter * CLOCK_RATE) & 0xFFFFFFFF
        if self._last_sr_mid32 != 0 and self._last_sr_arrival_ntp32 != 0:
            lsr = self._last_sr_mid32
            now_sec, now_frac = _unix_to_ntp(_now_unix())
            now32 = ((now_sec & 0xFFFF) << 16) | ((now_frac >> 16) & 0xFFFF)
            dlsr = (now32 - self._last_sr_arrival_ntp32) & 0xFFFFFFFF
        else:
            lsr = 0
            dlsr = 0
        return pack_rtcp_rr(self._ssrc, self._peer_ssrc, fraction, lost, ext_max, jitter_int, lsr, dlsr)

    # ======================= 发送/接收 API =======================
    def send(self, frame: Union[bytes, bytearray, memoryview],
             pkt_type: int = PKT_DATA,
             frame_id: Optional[int] = None) -> None:
        """
        向对端发送一个“帧”（可切片多个 RTP 包）。
        参数:
          - frame: 待发送字节
          - pkt_type: 负载类型（PKT_DATA/PKT_RES/PKT_SV_DATA/PKT_TERM）
          - frame_id: 指定该帧的 frame_id；None 则使用内部自增
        """
        if frame is None:
            return
        if self._peer_addr is None and not self._recv_mode:
            # 主动发送端：构造时已指定 address:port；接收端：收过一次后会记录 _peer_addr
            pass

        data = bytes(frame)
        fid = int(frame_id) if frame_id is not None else int(self._frame_id)
        if frame_id is None:
            self._frame_id = (self._frame_id + 1) & 0xFFFFFFFF

        max_data = self._max_rtp_payload - PAYLOAD_HDR_SIZE
        if max_data <= 0:
            raise ValueError("MTU 过小，无法发送负载，请增大 mtu。")

        total = len(data)
        total_frags = (total + max_data - 1) // max_data if total > 0 else 1

        off = 0
        for frag_idx in range(total_frags):
            chunk = data[off: off + max_data]
            off += len(chunk)
            marker = 1 if (frag_idx == total_frags - 1) else 0

            ph = struct.pack(PAYLOAD_HDR_FMT, int(pkt_type) & 0xFF, 0, fid,
                             total_frags, frag_idx, len(chunk))
            rtp_hdr = pack_rtp_header(self._seq, self._ts, self._ssrc, marker, PT_DYNAMIC)
            pkt = rtp_hdr + ph + chunk
            try:
                if self._peer_addr:
                    self._sock.sendto(pkt, self._peer_addr)
                else:
                    # 退路（理论上不应命中）
                    self._sock.send(pkt)
            except Exception as e:
                self._logging and logger.warning(f"send error: {e}")
                break

            self._sent_packets += 1
            self._sent_octets += len(ph) + len(chunk)
            self._seq = (self._seq + 1) & 0xFFFF
            if total_frags > 1 and frag_idx < total_frags - 1:
                time.sleep(0.0001)

        self._ts = (self._ts + self._ts_step) & 0xFFFFFFFF

    def recv(self) -> Optional[Dict[str, Any]]:
        """返回: None 或 dict{'pkt_type':int, 'frame_id':int, 'data':bytes}"""
        if self._queue:
            return self._queue.popleft()
        return None

    def close(self) -> None:
        self._terminate = True
        try:
            if self._peer_addr is not None:
                bye = pack_rtcp_bye(self._ssrc)
                try:
                    self._sock.sendto(bye, self._peer_addr)
                except Exception:
                    pass
            threads = []
            if hasattr(self, "_rx_thread") and self._rx_thread.is_alive():
                threads.append(self._rx_thread)
            if hasattr(self, "_rtcp_thread") and self._rtcp_thread.is_alive():
                threads.append(self._rtcp_thread)
            if hasattr(self, "_stats_thread") and self._stats_thread.is_alive():
                threads.append(self._stats_thread)
            for t in threads:
                t.join(timeout=1.0)
        finally:
            try:
                self._sock.close()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_frames_received": self._total_frames_received,
            "total_packets_received": self._total_packets_received,
            "total_packets_lost": self._total_packets_lost,
            "queue_size": len(self._queue),
            "frames_in_progress": len(self._frames),
            "rtt_ms": self._rtt_ms,
            "jitter": self._jitter,
            "sent_packets": self._sent_packets,
            "sent_octets": self._sent_octets
        }
