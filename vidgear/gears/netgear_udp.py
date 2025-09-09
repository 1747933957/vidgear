#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NetGearUDP —— UDP 传输 + RTCP 反馈 + Tooth slow-module（按论文）在线推理
----------------------------------------------------------------------
对外仅暴露 NetGearUDP；模块内无全局变量/函数，所有状态内聚在类内。
"""

import socket
import threading
import time
import logging as log
import struct
import statistics
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union, List
import os
import json
import base64
from datetime import datetime

# ====== 慢模块加载（保持原逻辑，并支持动态导入） ======
try:
    # 若你的工程是包方式导入，可在同目录放 tooth_slow_module.py
    from .tooth_slow_module import ToothSlowModule, load_weights  # type: ignore
    _SLOW_AVAILABLE = True
except Exception:
    ToothSlowModule = None  # type: ignore
    load_weights = None     # type: ignore
    _SLOW_AVAILABLE = False

import importlib.util as _imp  # MODIFIED: 动态导入兜底
import types as _types         # MODIFIED: 动态导入兜底
import queue                   # MODIFIED: 异步日志队列


def _load_slow_module_dynamic(path: str):
    """MODIFIED: 从给定文件路径动态加载 tooth_slow_module.py（或 slow_module.py）。"""
    spec = _imp.spec_from_file_location("tooth_slow_module_dyn", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"spec_from_file_location failed for {path}")
    mod = _imp.module_from_spec(spec)  # type: _types.ModuleType
    spec.loader.exec_module(mod)       # type: ignore[attr-defined]
    return mod


class NetGearUDP:
    """
    角色：
      - receive_mode=False: 发送端（发 DATA/PING；收 RTCP/PONG/ACK）
      - receive_mode=True : 接收端（收 DATA/PING；发 RTCP/PONG）

    关键 API：
      - send(data: bytes, pkt_type: int=None) -> None
      - recv() -> Optional[Dict[str, Any]]
      - get_rtcp_report() -> Optional[Dict[str, Any]]
      - get_rtt_stats() -> Dict[str, Any]
      - get_stats() / reset_stats() / close()
      - udp_ack_packet(packet_id: int, ts_ms: Optional[int]=None) -> None
      - get_slow_prediction() -> Dict[str, float]
      - get_network_status() -> Dict[str, Any]
    """

    DEFAULT_MTU = 1500
    RECV_QUEUE_MAXLEN = 32768

    PKT_DATA = 0
    PKT_RES = 1
    PKT_SV_DATA = 2
    PKT_TERM = 3
    PKT_ACK_FRAME = 4
    PKT_ACK_PACKET = 5
    PKT_PING = 6
    PKT_PONG = 7
    PKT_RTCP_REPORT = 8

    _LOGGER = log.getLogger("NetGearUDP")
    _LOGGER.setLevel(log.DEBUG)
    if not _LOGGER.handlers:
        _h = log.StreamHandler()
        _f = log.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
        _h.setFormatter(_f)
        _LOGGER.addHandler(_h)

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

        self._mtu = int(options.get("mtu", self.DEFAULT_MTU))
        self._recv_buf_size = int(options.get("recv_buffer_size", 16 * 1024 * 1024))
        self._send_buf_size = int(options.get("send_buffer_size", 16 * 1024 * 1024))
        self._queue_maxlen = int(options.get("queue_maxlen", self.RECV_QUEUE_MAXLEN))
        self._enable_seq = bool(options.get("enable_seq_header", True))
        self._rtcp_interval_ms = int(options.get("rtcp_interval_ms", 100))

        # ---- slow-module 日志与权重路径 ----
        self._slow_log_dir: str = str(options.get("slow_log_dir", "./tmp/slow_logs_sender" if not receive_mode else "./tmp/slow_logs_receiver"))
        self._slow_weights_path: str = str(options.get("slow_weights_path", "./tmp/slow_module_weights.json"))
        self._slow_hist_n: int = int(options.get("slow_hist_n", 10))  # 论文 n=10

        # ---- 固定位图长度配置 ----
        self._slow_slots_per_window: int = int(options.get("slow_slots_per_window", 32))
        self._slow_bytes_per_window: int = (self._slow_slots_per_window + 7) // 8

        # ---- 慢模块文件路径（动态导入兜底） ----
        self._slow_module_path: Optional[str] = options.get("slow_module_path")

        # 创建日志目录与本次 session 的 JSONL 文件
        try:
            os.makedirs(self._slow_log_dir, exist_ok=True)
        except Exception:
            pass
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 端侧 session 文件（发送端/接收端各一份）
        self._slow_jsonl_path = os.path.join(self._slow_log_dir, f"session_{ts_name}.jsonl")

        # ---- 发送/接收统计 ----
        self._sent_packets = 0
        self._total_packets_received = 0

        # RTT（发送端 PING/PONG 估计，单位 ms）
        self._rtt_samples = deque(maxlen=512)
        self._rtt_ema_ms: Optional[float] = None
        self._ping_seq = 0
        self._ping_pending: Dict[int, float] = {}

        # RTCP（仅发送端使用拉取队列）
        self._rtcp_reports: deque = deque(maxlen=1024)
        self._last_rtcp: Optional[Dict[str, Any]] = None

        # 接收端窗口统计（用于 lr/la 与接收端 P_ls）
        self._win_start_ts = time.time()
        self._win_rx_count = 0
        self._win_expected_count = 0
        self._last_seq_rx: Optional[int] = None
        self._rx_missing_seq_times: List[float] = []

        # ====== 发送端：慢模块输入历史（使用对端 P_ls） ======
        self._hist_ls_bits: deque = deque(maxlen=self._slow_hist_n)
        self._hist_lr: deque = deque(maxlen=self._slow_hist_n)
        self._hist_la: deque = deque(maxlen=self._slow_hist_n)

        # ====== 发送端：包级 ACK 记录与滚动落盘（MODIFIED） ======
        self._acked_seq: set = set()            # 仍维护，供诊断
        self._tx_records: deque = deque()       # 每项: {"seq":int,"ts_send":float,"ts_ack":Optional[float]}
        self._tx_index: Dict[int, Dict[str, Any]] = {}
        self._tx_pkt_log_path = os.path.join(self._slow_log_dir, f"tx_packets_sender_{ts_name}.jsonl")
        self._last_tx_dump_ts = time.time()

        # ====== 慢模块 ======
        self._slow_pred: Dict[str, float] = {"lr_f": 0.0, "la_f": 0.0}

        global _SLOW_AVAILABLE, ToothSlowModule, load_weights
        if (not _SLOW_AVAILABLE) and self._slow_module_path:
            try:
                _mod = _load_slow_module_dynamic(self._slow_module_path)
                ToothSlowModule = getattr(_mod, "ToothSlowModule")
                load_weights = getattr(_mod, "load_weights", None)
                _SLOW_AVAILABLE = True
                if self._logging:
                    self._LOGGER.info(f"[SlowModule] Loaded from file: {self._slow_module_path}")
            except Exception as e:
                if self._logging:
                    self._LOGGER.warning(f"[SlowModule] dynamic import failed: {e}")
                _SLOW_AVAILABLE = False

        if _SLOW_AVAILABLE:
            try:
                self._slow_model = ToothSlowModule()  # type: ignore
                if os.path.isfile(self._slow_weights_path) and (load_weights is not None):
                    load_weights(self._slow_model, self._slow_weights_path)  # type: ignore
                    if self._logging:
                        self._LOGGER.info(f"[SlowModule] Loaded weights: {self._slow_weights_path}")
            except Exception as e:
                self._slow_model = None  # type: ignore
                if self._logging:
                    self._LOGGER.warning(f"[SlowModule] init failed: {e}")
        else:
            self._slow_model = None  # type: ignore
            if self._logging:
                self._LOGGER.warning("[SlowModule] slow_module.py 未可用，在线推理被禁用。")

        # ---- UDP 套接字 ----
        self._recv_mode = bool(receive_mode)
        self._queue: deque = deque(maxlen=self._queue_maxlen)
        self._peer_addr: Optional[Tuple[str, int]] = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf_size)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_buf_size)
            self._sock.setblocking(False)
        except Exception as e:
            if self._logging:
                self._LOGGER.warning(f"Socket option setting failed: {e}")

        self._terminate = False

        if self._recv_mode:
            self._sock.bind((self._addr, self._port))
            if self._logging:
                self._LOGGER.info(f"[UDP] Bind at {self._addr}:{self._port}")
        else:
            self._peer_addr = (self._addr, self._port)
            if self._logging:
                self._LOGGER.info(f"[UDP] Send to {self._addr}:{self._port}")
        local_bind_port = options.get("local_bind_port", None)
        if (not self._recv_mode) and local_bind_port:
            self._sock.bind(("0.0.0.0", int(local_bind_port)))

        # ---- 异步日志线程（MODIFIED） ----
        self._logq: "queue.SimpleQueue[Tuple[str,str]]" = queue.SimpleQueue()
        self._log_thr = threading.Thread(target=self._log_sink_loop, name="UDP-LOG", daemon=True)
        self._log_thr.start()

        # ---- I/O 与 RTCP 线程 ----
        self._rx_thread = threading.Thread(target=self._io_loop, name="UDP-RX", daemon=True)
        self._rx_thread.start()

        self._rtcp_thread = threading.Thread(target=self._rtcp_loop, name="UDP-RTCP", daemon=True)
        self._rtcp_thread.start()

    # ===================== 工具 =====================
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # ===================== 类型判断（供上层使用） =====================
    def is_ack_frame(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_ACK_FRAME

    def is_ack_packet(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_ACK_PACKET

    def is_res(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_RES

    # ===================== 包级 ACK 入口（上层收到 PKT_ACK_PACKET 后调用） =====================
    def udp_ack_packet(self, packet_id: int, ts_ms: Optional[int] = None) -> None:
        """
        标记一个已确认的包序号（发送端诊断用）；同时更新滚动 tx 记录（MODIFIED）。
        """
        try:
            self._acked_seq.add(int(packet_id))
        except Exception:
            pass
        # MODIFIED: 更新滚动记录的 ts_ack
        try:
            rec = self._tx_index.get(int(packet_id))
            if rec is not None and rec.get("ts_ack") is None:
                rec["ts_ack"] = float(time.time()) if ts_ms is None else (float(ts_ms)/1000.0)
        except Exception:
            pass

    # ===================== 慢模块预测/网络状态查询 =====================
    def get_slow_prediction(self) -> Dict[str, float]:
        return dict(self._slow_pred)

    def get_network_status(self) -> Dict[str, Any]:
        return {
            "rtcp": (None if self._last_rtcp is None else dict(self._last_rtcp)),
            "rtt": self.get_rtt_stats(),
            "slow_pred": self.get_slow_prediction(),
            "hist": {
                "lr": list(self._hist_lr),
                "la": list(self._hist_la),
                "ls_windows": len(self._hist_ls_bits),
            },
        }

    # ===================== 发送 =====================
    def send(self, frame: Union[bytes, bytearray, memoryview], pkt_type: int = None) -> None:
        """
        发送一个逻辑包；若启用 seq 头，则对 DATA/RES/SV_DATA/ACK_PACKET 写入 4B 序号。
        """
        if frame is None:
            return
        if pkt_type is None:
            pkt_type = self.PKT_DATA
        if self._peer_addr is None and not self._recv_mode:
            raise RuntimeError("No peer address configured for sender.")

        custom_hdr_len = 1 + (4 if (self._enable_seq and pkt_type in
                                    (self.PKT_DATA, self.PKT_RES, self.PKT_SV_DATA, self.PKT_ACK_PACKET)) else 0)
        max_payload = self._mtu - 28 - custom_hdr_len
        if max_payload <= 0:
            raise ValueError(f"MTU 设置过小：mtu={self._mtu}。")
        data = bytes(frame)
        if len(data) > max_payload:
            raise ValueError(f"数据过大：len(data)={len(data)} > {max_payload}。")

        out = bytearray()
        out.append(int(pkt_type) & 0xFF)

        seq_for_log: Optional[int] = None
        if self._enable_seq and pkt_type in (self.PKT_DATA, self.PKT_RES, self.PKT_SV_DATA, self.PKT_ACK_PACKET):
            if not hasattr(self, "_seq_tx"):
                self._seq_tx = 0  # type: ignore[attr-defined]
            self._seq_tx = int((self._seq_tx + 1) & 0xFFFFFFFF)  # type: ignore[attr-defined]
            out.extend(struct.pack(">I", self._seq_tx))  # type: ignore[attr-defined]
            seq_for_log = int(self._seq_tx)  # type: ignore[attr-defined]

        out.extend(data)

        try:
            if self._peer_addr:
                self._sock.sendto(out, self._peer_addr)
            else:
                raise RuntimeError("Receiver has no known peer address to send to yet.")
            self._sent_packets += 1

            # MODIFIED: 发送端滚动记录DATA包（仅发送端 && DATA）
            if (not self._recv_mode) and (pkt_type == self.PKT_DATA) and (seq_for_log is not None):
                now = time.time()
                rec = {"seq": int(seq_for_log), "ts_send": float(now), "ts_ack": None}
                self._tx_records.append(rec)
                self._tx_index[int(seq_for_log)] = rec
                # 控制极端增长（非常规兜底）：超过 200k 也做一次瘦身
                if len(self._tx_records) > 200000:
                    self._dump_old_tx_records(force=True)

        except Exception as e:
            if self._logging:
                self._LOGGER.warning(f"send error: {e}")
            raise

    # ===================== 接收队列/RTT/统计 =====================
    def recv(self) -> Optional[Dict[str, Any]]:
        if self._queue:
            return self._queue.popleft()
        return None

    def get_rtcp_report(self) -> Optional[Dict[str, Any]]:
        if self._rtcp_reports:
            return self._rtcp_reports.popleft()
        return None

    def get_rtt_stats(self) -> Dict[str, Any]:
        median_ms = statistics.median(self._rtt_samples) if self._rtt_samples else None
        return {
            "ema_ms": None if self._rtt_ema_ms is None else float(self._rtt_ema_ms),
            "median_ms": None if median_ms is None else float(median_ms),
            "samples": int(len(self._rtt_samples)),
        }

    def get_stats(self) -> Dict[str, int]:
        return {"total_packets_received": int(self._total_packets_received),
                "sent_packets": int(self._sent_packets)}

    def reset_stats(self) -> None:
        self._win_start_ts = time.time()
        self._win_rx_count = 0
        self._win_expected_count = 0
        self._last_seq_rx = None
        self._rx_missing_seq_times.clear()
        self._total_packets_received = 0
        self._sent_packets = 0

    def close(self) -> None:
        self._terminate = True
        try:
            if hasattr(self, "_rtcp_thread") and self._rtcp_thread.is_alive():
                self._rtcp_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if hasattr(self, "_rx_thread") and self._rx_thread.is_alive():
                self._rx_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if hasattr(self, "_log_thr") and self._log_thr.is_alive():
                # 给日志线程一点时间刷盘
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            try:
                self._sock.close()
            except Exception:
                pass

    # ===================== I/O 线程（收包/解析） =====================
    def _io_loop(self) -> None:
        import select
        while not self._terminate:
            try:
                ready = select.select([self._sock], [], [], 0.05)
                if not ready[0]:
                    # 空闲时机：发送端执行滚动落盘（MODIFIED）
                    if not self._recv_mode:
                        self._dump_old_tx_records()
                    continue

                pkt, peer = self._sock.recvfrom(65535)
                if not pkt:
                    continue
                self._peer_addr = peer

                pkt_type = pkt[0]
                offset = 1
                seq: Optional[int] = None

                # 可选序号头
                if self._enable_seq and pkt_type in (
                    self.PKT_DATA, self.PKT_RES, self.PKT_SV_DATA, self.PKT_ACK_PACKET
                ):
                    if len(pkt) < 1 + 4:
                        if self._logging:
                            self._LOGGER.warning("packet too short for seq header")
                        continue
                    seq = struct.unpack(">I", pkt[offset:offset+4])[0]
                    offset += 4

                body = pkt[offset:]

                # === 控制面包的内部处理（不上抛） ===
                if pkt_type == self.PKT_ACK_PACKET:
                    self._total_packets_received += 1
                    # MODIFIED: 自动解析 data_seq 并登记 ACK（避免依赖上层）
                    try:
                        if len(body) >= 4:
                            data_seq = struct.unpack(">I", body[:4])[0]  # 按 ACK 格式调整
                            self.udp_ack_packet(int(data_seq))
                    except Exception:
                        pass
                    continue  # MODIFIED: 不上抛

                if pkt_type == self.PKT_PING:
                    if self._recv_mode and self._peer_addr:
                        out = bytes([self.PKT_PONG]) + body
                        try:
                            self._sock.sendto(out, self._peer_addr)
                        except Exception as e:
                            if self._logging:
                                self._LOGGER.warning(f"send PONG error: {e}")
                    continue  # MODIFIED: 不上抛

                if pkt_type == self.PKT_PONG:
                    now = time.time()
                    if len(body) >= 12:
                        ts, pseq = struct.unpack(">dI", body[:12])
                        rtt_ms = (now - ts) * 1000.0
                        self._rtt_samples.append(rtt_ms)
                        if self._rtt_ema_ms is None:
                            self._rtt_ema_ms = rtt_ms
                        else:
                            self._rtt_ema_ms = 0.8 * self._rtt_ema_ms + 0.2 * rtt_ms
                        self._ping_pending.pop(pseq, None)
                    continue  # MODIFIED: 不上抛

                if pkt_type == self.PKT_RTCP_REPORT and not self._recv_mode:
                    if len(body) >= 28:
                        try:
                            win_start_ts, lr, la, rx_pkts, win_ms, last_seq = struct.unpack(">dffffI", body[:28])
                            rep = {
                                "win_start_ts": float(win_start_ts),
                                "lr": float(lr),
                                "la": float(la),
                                "rx_pkts": int(rx_pkts),
                                "win_ms": float(win_ms),
                                "last_seq_rx": int(last_seq),
                            }
                            self._last_rtcp = rep
                            if len(self._rtcp_reports) < self._rtcp_reports.maxlen:
                                self._rtcp_reports.append(rep)

                            bits = None
                            if len(body) > 28:
                                blen = body[28]
                                bmap = body[29:29+blen]
                                if len(bmap) == blen:
                                    S = blen * 8
                                    bits = []
                                    for j in range(S):
                                        byte = bmap[j // 8]
                                        bits.append((byte >> (j % 8)) & 1)  # LSB-first

                            if bits is not None:
                                # 用接收端位图推进历史 & 写发送端 JSONL & 触发慢模块
                                self._hist_ls_bits.append(bits)
                                self._hist_lr.append(float(lr))
                                self._hist_la.append(float(la))
                                self._write_jsonl_sample(float(win_start_ts), float(lr), float(la),
                                                         float(win_ms), bits_override=bits)  # 发送端日志
                                self._run_slow_forward()
                            else:
                                # 未携带位图（兼容老对端）：此处可选择跳过或进入旧的 pending 流程
                                pass
                        except Exception as e:
                            if self._logging:
                                self._LOGGER.warning(f"parse RTCP report error: {e}")
                    continue  # MODIFIED: 不上抛

                # === 以下是数据面包（会上抛） ===
                # 接收端：统计窗口内按序号缺口（用于计算 lr/la 与构建接收端 P_ls）
                if self._recv_mode and self._enable_seq and (pkt_type in (self.PKT_DATA, self.PKT_RES, self.PKT_SV_DATA)):
                    now = time.time()
                    self._win_rx_count += 1
                    if seq is not None:
                        if self._last_seq_rx is None:
                            self._last_seq_rx = seq
                            self._win_expected_count += 1
                        else:
                            gap = seq - self._last_seq_rx
                            if gap <= 0:
                                self._win_expected_count += 1
                            else:
                                self._win_expected_count += gap
                                if gap > 1:
                                    # 记录缺口时间（gap-1 次）
                                    self._rx_missing_seq_times.extend([now] * (gap - 1))
                            self._last_seq_rx = seq

                # 入队给上层（仅非控制包）
                if len(self._queue) < self._queue_maxlen:
                    self._queue.append({
                        "pkt_type": int(pkt_type),
                        "data": body,
                        "seq": None if seq is None else int(seq),
                    })

            except socket.error as e:
                err = getattr(e, "errno", None)
                if err in (socket.EAGAIN, socket.EWOULDBLOCK):
                    time.sleep(0.0001)
                    continue
                if not self._terminate and self._logging:
                    self._LOGGER.error(f"recv error: {e}")
            except Exception as e:
                if not self._terminate and self._logging:
                    self._LOGGER.error(f"recv error: {e}")

    # ===================== RTCP/PING 线程 =====================
    def _rtcp_loop(self) -> None:
        """
        - 发送端：定期发 PING，基于 PONG 估计 RTT（毫秒）
        - 接收端：每 rtcp_interval_ms 汇总一次窗口，发送 RTCP_REPORT：
            body: [win_start_ts(double, seconds), lr(float), la(float), rx_pkts(float), win_ms(float), last_seq_rx(u32)]
            + [bitmap_len(1B)] + [bitmap_bytes]
        """
        ping_interval = max(2 * self._rtcp_interval_ms, 100) / 1000.0
        last_ping_ts = time.time()

        while not self._terminate:
            now = time.time()

            # 发送端：周期性 PING
            if not self._recv_mode and self._peer_addr and (now - last_ping_ts) >= ping_interval:
                last_ping_ts = now
                try:
                    self._ping_seq = (self._ping_seq + 1) & 0xFFFFFFFF
                    payload = struct.pack(">dI", now, self._ping_seq)
                    out = bytes([self.PKT_PING]) + payload
                    self._ping_pending[self._ping_seq] = now
                    self._sock.sendto(out, self._peer_addr)
                except Exception as e:
                    if self._logging:
                        self._LOGGER.debug(f"send PING error: {e}")

            # 接收端：汇总 RTCP（MODIFIED: 附带接收端位图，并在发送时写接收端 JSONL）
            if self._recv_mode and (now - self._win_start_ts) * 1000.0 >= self._rtcp_interval_ms:
                rx = self._win_rx_count
                expected = max(self._win_expected_count, rx)
                lost = max(expected - rx, 0)
                lr = (lost / expected) if expected > 0 else 0.0

                la = 0.0
                n = len(self._rx_missing_seq_times)
                if n > 1:
                    centroid = sum(self._rx_missing_seq_times) / n
                    denom = sum(abs(t - centroid) for t in self._rx_missing_seq_times) + 0.5
                    la = float(n) / denom if denom > 0 else float(n)

                win_ms = (now - self._win_start_ts) * 1000.0

                # MODIFIED: 计算接收端位图
                S = int(self._slow_slots_per_window)
                bits = self._build_rx_pls_bits(self._win_start_ts, win_ms, S)
                bmap_bytes = bytearray((S + 7) // 8)
                for i, b in enumerate(bits):
                    if b:
                        bmap_bytes[i // 8] |= (1 << (i % 8))

                if self._peer_addr:
                    body = struct.pack(">dffffI",
                                       float(self._win_start_ts),
                                       float(lr), float(la),
                                       float(rx), float(win_ms),
                                       int(self._last_seq_rx or 0))
                    # 附带位图（长度 + 字节）
                    body += struct.pack("B", len(bmap_bytes)) + bytes(bmap_bytes)
                    out = bytes([self.PKT_RTCP_REPORT]) + body
                    try:
                        self._sock.sendto(out, self._peer_addr)
                    except Exception as e:
                        if self._logging:
                            self._LOGGER.debug(f"send RTCP report error: {e}")

                    # 接收端：发送 RTCP 时写一行 JSONL（接收端日志）
                    self._write_jsonl_sample(
                        win_start_sec=float(self._win_start_ts),
                        lr=float(lr),
                        la=float(la),
                        win_ms=float(win_ms),
                        bits_override=bits,
                        out_path_override=self._slow_jsonl_path,  # 接收端自己的 session 文件
                    )

                # 窗口复位
                self._win_start_ts = now
                self._win_rx_count = 0
                self._win_expected_count = 0
                self._rx_missing_seq_times.clear()

            # 发送端：周期性滚动落盘（MODIFIED）
            if not self._recv_mode:
                self._dump_old_tx_records()

            time.sleep(0.01)

    # ===================== 接收端位图构造（MODIFIED） =====================
    def _build_rx_pls_bits(self, win_start_sec: float, win_ms: float, S: int) -> List[int]:
        """
        接收端 P_ls：用窗口内“缺失序号”的时间戳集合 -> slot 索引置 1。
        1 表示该时间段内出现了缺口（成团丢/极迟到）；0 表示未丢或无发送。
        """
        S = max(1, int(S))
        bits = [0] * S
        if win_ms <= 0:
            return bits
        slot_ms = float(win_ms) / float(S)
        win_start = float(win_start_sec)
        win_end = win_start + float(win_ms) / 1000.0
        for t in self._rx_missing_seq_times:
            if t < win_start or t >= win_end:
                continue
            rel_ms = (t - win_start) * 1000.0
            idx = int(rel_ms // slot_ms)
            if idx < 0:
                idx = 0
            elif idx >= S:
                idx = S - 1
            bits[idx] = 1
        return bits

    # ===================== JSONL 写入（异步入队） =====================
    def _write_jsonl_sample(self, win_start_sec: float, lr: float, la: float, win_ms: float,
                            bits_override: Optional[List[int]] = None,
                            out_path_override: Optional[str] = None) -> None:
        """
        写单窗口 JSONL（异步）：
        - pls_bits_b64 固定长度：bytes_len = ceil(S/8)
        - 采用 LSB-first 打包：slot0 -> 第0位的 LSB，以此类推
        - pls_slot_ms = window_ms / S（固定）
        """
        try:
            S = int(self._slow_slots_per_window)
            B = int(self._slow_bytes_per_window)

            if bits_override is not None:
                bits = list(bits_override)
            else:
                bits = self._hist_ls_bits[-1] if self._hist_ls_bits else [0] * S

            by = bytearray(B)
            for i, bit in enumerate(bits):
                if bit:
                    by[i // 8] |= (1 << (i % 8))

            b64 = base64.b64encode(bytes(by)).decode("ascii")

            rec = {
                "ts_ms": int(float(win_start_sec) * 1000.0),
                "lr": float(lr),
                "la": float(la),
                "pls_bits_b64": b64,
                "window_ms": float(win_ms),
                "pls_slot_ms": float(win_ms) / max(1, S),
            }
            path = out_path_override or self._slow_jsonl_path
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            # MODIFIED: 异步入队，后台批量写盘
            self._logq.put((path, line))
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_write_jsonl_sample error: {e}")

    # ===================== 异步日志落盘线程（MODIFIED） =====================
    def _log_sink_loop(self) -> None:
        """
        将 (path, line) 从队列中取出，按文件聚合批量写入。
        """
        buffers: Dict[str, List[str]] = {}
        last_flush = time.time()
        FLUSH_LINES = 200
        FLUSH_SEC = 0.2
        while not self._terminate:
            try:
                path, line = self._logq.get(timeout=0.05)
                buf = buffers.setdefault(path, [])
                buf.append(line)
            except Exception:
                pass
            now = time.time()
            if (now - last_flush) >= FLUSH_SEC:
                for path, buf in list(buffers.items()):
                    if not buf:
                        continue
                    try:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "a", encoding="utf-8") as f:
                            f.write("".join(buf))
                    except Exception:
                        pass
                    buffers[path].clear()
                last_flush = now
            else:
                # 若某个文件的缓存积累到阈值，也立即刷新
                for path, buf in list(buffers.items()):
                    if len(buf) >= FLUSH_LINES:
                        try:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "a", encoding="utf-8") as f:
                                f.write("".join(buf))
                        except Exception:
                            pass
                        buffers[path].clear()

    # ===================== 慢模块在线前向 =====================
    def _run_slow_forward(self) -> None:
        """
        将最近 n=self._slow_hist_n 个窗口的定长位图按时间顺序拼接（不足 n 个则左侧补 0），
        联合 P_lr/P_la 历史，完成一次慢模块前向。
        """
        if self._slow_model is None:
            return
        try:
            import torch  # 本地 CPU 前向足够
            n = int(self._slow_hist_n)
            S = int(self._slow_slots_per_window)

            wins = list(self._hist_ls_bits)[-n:]
            if len(wins) < n:
                pad = [[0]*S for _ in range(n - len(wins))]
                wins = pad + wins

            concat_bits: List[int] = []
            for w in wins:
                concat_bits.extend(w)

            P_lr = list(self._hist_lr)[-n:]
            P_la = list(self._hist_la)[-n:]
            if len(P_lr) < n:
                P_lr = [0.0]*(n - len(P_lr)) + P_lr
            if len(P_la) < n:
                P_la = [0.0]*(n - len(P_la)) + P_la

            with torch.no_grad():
                y = self._slow_model(P_lr=P_lr, P_la=P_la, l_s_bits=concat_bits)  # type: ignore
            lr_f = float(y[0].item())
            la_f = float(y[1].item())
            self._slow_pred = {"lr_f": lr_f, "la_f": la_f}
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_run_slow_forward error: {e}")

    # ===================== 发送端包级 ACK 记录滚动落盘（MODIFIED） =====================
    def _dump_old_tx_records(self, force: bool = False) -> None:
        """
        每 10 秒处理“10 秒前”的记录；策略：
        - 10s：仅写出“已 ACK”的记录；
        - 20s：强制写出（即便仍未 ACK）；
        所有落盘通过异步日志队列，避免热路径 IO。
        """
        try:
            now = time.time()
            if (not force) and ((now - self._last_tx_dump_ts) < 10.0):
                return
            self._last_tx_dump_ts = now
            cutoff_dump = now - 10.0   # <= 10s 前 -> 若已 ACK 则写
            cutoff_force = now - 20.0  # <= 20s 前 -> 强制写（未 ACK 也写）

            out_lines = []
            while self._tx_records:
                rec = self._tx_records[0]
                ts_send = float(rec.get("ts_send", now))
                if ts_send <= cutoff_force:
                    # 强制写（即便未 ACK）
                    self._tx_records.popleft()
                    self._tx_index.pop(int(rec.get("seq", -1)), None)
                    line = {
                        "seq": int(rec.get("seq", -1)),
                        "ts_send_ms": int(round(ts_send * 1000.0)),
                        "ts_ack_ms": (None if rec.get("ts_ack") is None else int(round(float(rec["ts_ack"]) * 1000.0)))
                    }
                    out_lines.append(json.dumps(line, ensure_ascii=False) + "\n")
                elif ts_send <= cutoff_dump:
                    # 只写已 ACK 的，未 ACK 继续等待
                    if rec.get("ts_ack") is not None:
                        self._tx_records.popleft()
                        self._tx_index.pop(int(rec.get("seq", -1)), None)
                        line = {
                            "seq": int(rec.get("seq", -1)),
                            "ts_send_ms": int(round(ts_send * 1000.0)),
                            "ts_ack_ms": int(round(float(rec["ts_ack"]) * 1000.0)),
                        }
                        out_lines.append(json.dumps(line, ensure_ascii=False) + "\n")
                    else:
                        break
                else:
                    break  # 后面的都是更近的，保留

            # 异步写盘
            for s in out_lines:
                self._logq.put((self._tx_pkt_log_path, s))

        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_dump_old_tx_records error: {e}")
