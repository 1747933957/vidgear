#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NetGearUDP —— UDP 传输 + RTCP 反馈 + Tooth slow-module（按论文）在线推理
----------------------------------------------------------------------
对外仅暴露 NetGearUDP；模块内无全局变量/函数，所有状态内聚在类内。

本版要点：
  1) 集成 ToothSlowModule（来自 slow_module.py，严格按论文结构）。
  2) 在发送端构造逐包位图 l_s（0/1，1=丢失，0=收到），并按 RTCP 窗口写 JSONL 日志。
  3) 【MODIFIED】收到 RTCP 后不立即结算窗口，而是引入 ACK 宽限期（基于 RTT），
     到期后再生成定长位图 P_ls，写入 JSONL，并触发慢模块前向。
  4) __init__ 时如 slow_module_weights.json 存在，则加载权重。训练与加载均与 slow_module.py 配合。
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

# ====== 引入严格按论文实现的 slow-module ======
try:
    # 你刚刚提供的 slow_module.py 中的接口
    from vidgear.gears.tooth_slow_module import ToothSlowModule, load_weights  # type: ignore
    _SLOW_AVAILABLE = True
except Exception:
    ToothSlowModule = None  # type: ignore
    load_weights = None     # type: ignore
    _SLOW_AVAILABLE = False


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
      - udp_ack_packet(packet_id: int, ts_ms: Optional[int]=None) -> None   # 包级 ACK 入口（上层收到 PKT_ACK_PACKET 后调用）
      - get_slow_prediction() -> Dict[str, float]    # 返回 {'lr_f','la_f'}
      - get_network_status() -> Dict[str, Any]       # 打包当前网络状态（含慢模块最新预测）

    训练/日志相关（与 slow_module.py 配合）：
      - 初始化参数：
          * slow_log_dir: str = "./slow_logs"                  # JSONL 日志目录
          * slow_weights_path: str = "./slow_module_weights.json"  # 权重 JSON 路径
          * slow_hist_n: int = 10                              # 历史窗口数 n（论文用 10）
    """

    # ===================== 包类型常量 =====================
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

    # ===================== 日志器 =====================
    _LOGGER = log.getLogger("NetGearUDP")
    _LOGGER.setLevel(log.DEBUG)
    if not _LOGGER.handlers:
        _h = log.StreamHandler()
        _f = log.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
        _h.setFormatter(_f)
        _LOGGER.addHandler(_h)

    # ===================== 构造 =====================
    def __init__(
        self,
        address: str = "0.0.0.0",
        port: Union[int, str] = 5556,
        protocol: str = "udp",
        receive_mode: bool = False,
        logging: bool = True,
        **options
    ) -> None:
        # ---- 基础配置 ----
        self._logging = bool(logging)
        self._addr = address
        self._port = int(port)
        if protocol.lower() != "udp":
            raise ValueError("NetGearUDP 仅支持 protocol='udp'。")

        # ---- UDP 参数 ----
        self._mtu = int(options.get("mtu", self.DEFAULT_MTU))
        self._recv_buf_size = int(options.get("recv_buffer_size", 16 * 1024 * 1024))
        self._send_buf_size = int(options.get("send_buffer_size", 16 * 1024 * 1024))
        self._queue_maxlen = int(options.get("queue_maxlen", self.RECV_QUEUE_MAXLEN))
        self._enable_seq = bool(options.get("enable_seq_header", True))
        self._rtcp_interval_ms = int(options.get("rtcp_interval_ms", 100))

        # ---- slow-module 日志与权重路径 ----
        self._slow_log_dir: str = str(options.get("slow_log_dir", "./slow_logs"))
        self._slow_weights_path: str = "/home/wxk/workspace/nsdi/Viduce/net/vidgear/tmp/slow_logs_sender/out.pt"
        self._slow_hist_n: int = int(options.get("slow_hist_n", 10))  # 论文 n=10

        # === 定长位图配置（确保数据集/在线一致）===
        self._slow_slots_per_window: int = int(options.get("slow_slots_per_window", 32))  # 建议 20/32/64
        self._slow_bytes_per_window: int = (self._slow_slots_per_window + 7) // 8

        # === 【MODIFIED】ACK 宽限策略 ===
        self._ack_grace_min_ms: int = int(options.get("ack_grace_min_ms", 50))
        self._ack_grace_rtt_factor: float = float(options.get("ack_grace_rtt_factor", 2.0))

        # 创建日志目录与本次 session 的 JSONL 文件
        try:
            os.makedirs(self._slow_log_dir, exist_ok=True)
        except Exception:
            pass
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._slow_jsonl_path = os.path.join(self._slow_log_dir, f"session_{ts_name}.jsonl")

        # ---- 发送/接收统计 ----
        self._sent_packets = 0
        self._total_packets_received = 0

        # RTT（发送端 PING/PONG 估计，单位 ms）
        self._rtt_samples = deque(maxlen=512)
        self._rtt_ema_ms: Optional[float] = None
        self._ping_seq = 0
        self._ping_pending: Dict[int, float] = {}

        # RTCP（仅发送端使用）
        self._rtcp_reports: deque = deque(maxlen=1024)   # 队列给上层拉取
        self._last_rtcp: Optional[Dict[str, Any]] = None # 最近一条 RTCP

        # 接收端窗口内的序号缺口计数（用于接收端计算 lr/la）
        self._win_start_ts = time.time()
        self._win_rx_count = 0
        self._win_expected_count = 0
        self._last_seq_rx: Optional[int] = None
        self._rx_missing_seq_times: List[float] = []

        # ====== 发送端逐包位图 l_s 所需的运行态 ======
        # 1) 最近发送的包（(seq, ts_send)），用于按 RTCP 窗口筛选
        self._tx_recent: deque = deque(maxlen=200000)  # type: deque[tuple[int,float]]
        # 2) 已确认的包序号集合（收到 PKT_ACK_PACKET 后标记）
        self._acked_seq: set = set()
        # 3) 最近 <=10 个窗口的逐包位图，按窗口顺序存储（每项为 List[int] 的 0/1 序列）
        self._hist_ls_bits: deque = deque(maxlen=self._slow_hist_n)
        # 4) 历史 P_lr/P_la（长度<=10）
        self._hist_lr: deque = deque(maxlen=self._slow_hist_n)
        self._hist_la: deque = deque(maxlen=self._slow_hist_n)
        # 5) 【MODIFIED】待结算窗口队列（收到 RTCP 后延迟到期再结算）
        self._pending_windows: deque = deque()  # 每项: dict(win_start, win_ms, lr, la, enqueue_ts)

        # ====== 慢模块（论文实现） ======
        self._slow_pred: Dict[str, float] = {"lr_f": 0.0, "la_f": 0.0}  # 最新预测缓存
        if _SLOW_AVAILABLE:
            try:
                self._slow_model = ToothSlowModule()  # type: ignore
                # 若存在权重 JSON，则加载
                if os.path.isfile(self._slow_weights_path):
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
            # 发送端如强制固定源端口，则主动bind；但端口不能等于server的5558
            self._sock.bind(("0.0.0.0", int(local_bind_port)))

        # ---- I/O 与 RTCP 线程 ----
        self._rx_thread = threading.Thread(target=self._io_loop, name="UDP-RX", daemon=True)
        self._rx_thread.start()

        self._rtcp_thread = threading.Thread(target=self._rtcp_loop, name="UDP-RTCP", daemon=True)
        self._rtcp_thread.start()

    # ===================== 工具 =====================
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # 【MODIFIED】动态计算 ACK 宽限（毫秒）
    def _current_ack_grace_ms(self) -> float:
        rtt = None if self._rtt_ema_ms is None else float(self._rtt_ema_ms)
        base = float(self._ack_grace_min_ms)
        if rtt is None:
            return base
        return max(base, float(self._ack_grace_rtt_factor) * rtt)

    # ===================== 类型判断（供上层使用） =====================
    def is_ack_frame(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_ACK_FRAME

    def is_ack_packet(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_ACK_PACKET

    def is_res(self, pkt_type: int) -> bool:
        return int(pkt_type) == self.PKT_RES

    # ===================== 包级 ACK 入口（上层在收到 PKT_ACK_PACKET 后调用） =====================
    def udp_ack_packet(self, packet_id: int, ts_ms: Optional[int] = None) -> None:
        """
        标记一个已确认的包序号，用于构造逐包位图 l_s（0=收到，1=丢失）。
        参数：
          - packet_id: DATA 包的序号（与 send 时写入的 seq 对齐）
          - ts_ms: 可选，ACK 到达时间（目前不需要）
        """
        try:
            self._acked_seq.add(int(packet_id))
        except Exception:
            pass

    # ===================== 慢模块预测/网络状态查询 =====================
    def get_slow_prediction(self) -> Dict[str, float]:
        """返回最近一次 online 推理的 slow 预测结果：{'lr_f','la_f'}。"""
        return dict(self._slow_pred)

    def get_network_status(self) -> Dict[str, Any]:
        """打包当前网络状态（供调试/上层查看）。"""
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
        新增：发送 DATA 时，记录 (seq, ts_send) 到 _tx_recent 供 l_s 构造使用。
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

        # 分配/写入序号
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

            # 记录 DATA 包的 (seq, ts_send) 以备 l_s 构造
            if (not self._recv_mode) and (pkt_type == self.PKT_DATA) and (seq_for_log is not None):
                self._tx_recent.append((seq_for_log, time.time()))
                # 适度清理陈旧发送记录（保留近 60 秒）
                cutoff = time.time() - 60.0
                while self._tx_recent and self._tx_recent[0][1] < cutoff:
                    self._tx_recent.popleft()

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
                    # 【MODIFIED】即便无包可收，也尝试在发送端结算到期窗口
                    if not self._recv_mode:
                        self._finalize_pending_windows()
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

                # 包级 ACK 数量（统计用途）
                if pkt_type == self.PKT_ACK_PACKET:
                    self._total_packets_received += 1

                # PING→PONG（RTT）
                if pkt_type == self.PKT_PING:
                    if self._recv_mode and self._peer_addr:
                        out = bytes([self.PKT_PONG]) + body
                        try:
                            self._sock.sendto(out, self._peer_addr)
                        except Exception as e:
                            if self._logging:
                                self._LOGGER.warning(f"send PONG error: {e}")
                    continue

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
                    continue

                # RTCP 报告：发送端入队【延迟结算】
                if pkt_type == self.PKT_RTCP_REPORT:
                    if not self._recv_mode:
                        try:
                            if len(body) >= 28:
                                win_start_ts, lr, la, rx_pkts, win_ms, last_seq = struct.unpack(">dffffI", body[:28])
                                rep = {
                                    "win_start_ts": float(win_start_ts),  # seconds
                                    "lr": float(lr),
                                    "la": float(la),
                                    "rx_pkts": int(rx_pkts),
                                    "win_ms": float(win_ms),
                                    "last_seq_rx": int(last_seq),
                                }
                                self._last_rtcp = rep
                                if len(self._rtcp_reports) < self._rtcp_reports.maxlen:
                                    self._rtcp_reports.append(rep)

                                # 【MODIFIED】仅入队，等宽限期后再结算
                                self._enqueue_pending_window(rep)

                                # 机会性结算（避免只在定时器里结算）
                                self._finalize_pending_windows()

                        except Exception as e:
                            if self._logging:
                                self._LOGGER.warning(f"parse RTCP report error: {e}")
                    continue

                # 接收端：统计窗口内按序号缺口（用于计算 lr/la）
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
                                    self._rx_missing_seq_times.extend([now] * (gap - 1))
                            self._last_seq_rx = seq

                # 入队给上层
                if len(self._queue) < self._queue_maxlen:
                    self._queue.append({
                        "pkt_type": int(pkt_type),
                        "data": body,
                        "seq": None if seq is None else int(seq),
                    })

                # 【MODIFIED】每次 I/O 后也尝试结算到期窗口
                if not self._recv_mode:
                    self._finalize_pending_windows()

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
          la 的计算仍照你之前的近似实现。
        - 【MODIFIED】发送端：周期性尝试结算到期窗口（补偿 I/O 循环可能的空档）
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

            # 接收端：汇总 RTCP
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

                if self._peer_addr:
                    body = struct.pack(">dffffI",
                                       float(self._win_start_ts),
                                       float(lr), float(la),
                                       float(rx), float(win_ms),
                                       int(self._last_seq_rx or 0))
                    out = bytes([self.PKT_RTCP_REPORT]) + body
                    try:
                        self._sock.sendto(out, self._peer_addr)
                    except Exception as e:
                        if self._logging:
                            self._LOGGER.debug(f"send RTCP report error: {e}")

                # 窗口复位
                self._win_start_ts = now
                self._win_rx_count = 0
                self._win_expected_count = 0
                self._rx_missing_seq_times.clear()

            # 【MODIFIED】发送端：周期性尝试结算到期窗口
            if not self._recv_mode:
                self._finalize_pending_windows()

            time.sleep(0.01)

    # ===================== 内部：延迟结算窗口 / 构造 l_s / 写日志 / 调慢模块 =====================

    # 【MODIFIED】入队一个待结算窗口（收到 RTCP 后调用）
    def _enqueue_pending_window(self, rep: Dict[str, Any]) -> None:
        try:
            item = dict(rep)
            item["enqueue_ts"] = time.time()
            self._pending_windows.append(item)
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_enqueue_pending_window error: {e}")

    # 【MODIFIED】结算到期窗口：满足 window_end + grace_ms 的窗口按规则生成定长位图并写日志/触发前向
    def _finalize_pending_windows(self) -> None:
        if not self._pending_windows:
            return
        try:
            now = time.time()
            grace_ms = self._current_ack_grace_ms()
            changed = False
            while self._pending_windows:
                it = self._pending_windows[0]
                win_start = float(it["win_start_ts"])
                win_ms = float(it["win_ms"])
                win_end = win_start + win_ms / 1000.0
                # 到期条件：当前时间 >= 窗口结束 + grace
                if now < (win_end + grace_ms / 1000.0):
                    break

                # 出队并结算
                self._pending_windows.popleft()
                lr = float(it["lr"]); la = float(it["la"])
                # 1) 生成定长位图（使用“超时仍未 ACK 才置 1”的规则）
                bits = self._build_ls_bits_with_grace(win_start, win_ms, grace_ms)
                # 2) 进历史
                self._hist_ls_bits.append(bits)
                self._hist_lr.append(lr)
                self._hist_la.append(la)
                # 3) 写 JSONL
                self._write_jsonl_sample(win_start, lr, la, win_ms, bits_override=bits)
                # 4) 在线前向
                self._run_slow_forward()
                changed = True

            # 如果有结算动作，顺便清理 _tx_recent 更旧的记录
            if changed:
                cutoff = now - 60.0
                while self._tx_recent and self._tx_recent[0][1] < cutoff:
                    self._tx_recent.popleft()
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_finalize_pending_windows error: {e}")

    # 【MODIFIED】带 ACK 宽限期的位图生成（定长、1=丢，0=未丢/未发送）
    def _build_ls_bits_with_grace(self, win_start_sec: float, win_ms: float, grace_ms: float) -> List[int]:
        try:
            S = int(self._slow_slots_per_window)
            win_start = float(win_start_sec)
            win_end = win_start + float(win_ms) / 1000.0
            slot_ms = float(win_ms) / max(1, S)
            finalize_ts = win_end + float(grace_ms) / 1000.0  # 结算时刻

            seen_loss = [False] * S  # 仅需记录“该 slot 是否发生过未 ACK 的发送”
            # 遍历窗口内发送的 DATA 包
            for (seq, ts_send) in list(self._tx_recent):
                if ts_send < win_start or ts_send >= win_end:
                    continue
                # 若到“结算时刻”仍未 ACK，则认为该包丢失，置位到对应 slot
                if int(seq) not in self._acked_seq:
                    # 仍未 ACK 且包龄 >= grace → 计为丢
                    if (finalize_ts - ts_send) >= 0:
                        rel_ms = (ts_send - win_start) * 1000.0
                        idx = int(rel_ms // slot_ms) if slot_ms > 0 else 0
                        if idx < 0:
                            idx = 0
                        elif idx >= S:
                            idx = S - 1
                        seen_loss[idx] = True

            # 生成定长位图：1=该 slot 发生过“超时未 ACK”的发送；0=未丢或未发送
            return [1 if seen_loss[i] else 0 for i in range(S)]

        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_build_ls_bits_with_grace error: {e}")
            return [0] * int(self._slow_slots_per_window)

    # （保留原无宽限版本以备调试，但不在生产路径使用）
    def _build_and_push_ls_for_window(self, win_start_sec: float, win_ms: float) -> None:
        try:
            S = int(self._slow_slots_per_window)
            win_start = float(win_start_sec)
            win_end = win_start + float(win_ms) / 1000.0
            slot_ms = float(win_ms) / max(1, S)

            seen_loss = [False] * S
            for (seq, ts) in list(self._tx_recent):
                if ts < win_start or ts >= win_end:
                    continue
                rel_ms = (ts - win_start) * 1000.0
                idx = int(rel_ms // slot_ms) if slot_ms > 0 else 0
                if idx < 0:
                    idx = 0
                elif idx >= S:
                    idx = S - 1
                if int(seq) not in self._acked_seq:
                    seen_loss[idx] = True

            l_s_bits = [1 if seen_loss[i] else 0 for i in range(S)]
            self._hist_ls_bits.append(l_s_bits)

        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_build_and_push_ls_for_window error: {e}")

    # 【MODIFIED】写 JSONL：允许传入 bits_override（用于延迟结算后的结果）
    def _write_jsonl_sample(self, win_start_sec: float, lr: float, la: float, win_ms: float,
                            bits_override: Optional[List[int]] = None) -> None:
        """
        写单窗口 JSONL：
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
                # 若历史为空则补 0 窗口
                bits = self._hist_ls_bits[-1] if self._hist_ls_bits else [0] * S

            # ---- 固定字节数打包（LSB-first）----
            by = bytearray(B)  # 预分配固定长度，默认全 0
            for i, bit in enumerate(bits):
                if bit:
                    byte_idx = i // 8
                    bit_off = i % 8          # LSB-first
                    by[byte_idx] |= (1 << bit_off)

            b64 = base64.b64encode(bytes(by)).decode("ascii")

            rec = {
                "ts_ms": int(float(win_start_sec) * 1000.0),
                "lr": float(lr),
                "la": float(la),
                "pls_bits_b64": b64,
                "window_ms": float(win_ms),
                "pls_slot_ms": float(win_ms) / max(1, S),    # 固定
            }
            with open(self._slow_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_write_jsonl_sample error: {e}")

    def _run_slow_forward(self) -> None:
        """
        将最近 n=self._slow_hist_n 个窗口的定长位图按时间顺序拼接（不足 n 个则在左侧用“全 0 窗口”填充），
        联合 P_lr/P_la 历史，完成一次慢模块前向。
        """
        if self._slow_model is None:
            return
        try:
            import torch
            n = int(self._slow_hist_n)
            S = int(self._slow_slots_per_window)

            # 取最近窗口的位图
            wins = list(self._hist_ls_bits)[-n:]
            if len(wins) < n:
                pad = [[0]*S for _ in range(n - len(wins))]  # 左侧 0 窗口
                wins = pad + wins

            # 拼接为固定长度向量（n*S）
            concat_bits: List[int] = []
            for w in wins:
                concat_bits.extend(w)  # 每 w 都已是长度 S

            P_lr = list(self._hist_lr)[-n:]
            P_la = list(self._hist_la)[-n:]
            if len(P_lr) < n:
                P_lr = [0.0]*(n - len(P_lr)) + P_lr
            if len(P_la) < n:
                P_la = [0.0]*(n - len(P_la)) + P_la

            with torch.no_grad():
                y = self._slow_model(P_lr=P_lr, P_la=P_la, l_s_bits=concat_bits)  # 固定长度输入
            lr_f = float(y[0].item())
            la_f = float(y[1].item())
            self._slow_pred = {"lr_f": lr_f, "la_f": la_f}
        except Exception as e:
            if self._logging:
                self._LOGGER.debug(f"_run_slow_forward error: {e}")
