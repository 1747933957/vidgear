#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极简 UDP 版 NetGear：
- 仅以“包”为单位进行发送与接收（不做帧级别的分片/重组）
- 自定义包头只有 1 字节: pkt_type(u8)
- 发送: send(data, pkt_type) 会在 data 前面加上 1 字节 pkt_type
- 接收: 解析首字节为 pkt_type，余下全部作为 data
- 统计: 仅维护
    * _sent_packets: 发送的 UDP 包总数
    * _total_packets_received: （仅当收到 pkt_type == PKT_ACK_PACKET 时）接收计数 +1
"""

import socket
import threading
import time
import logging as log
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

# ===================== 常量与参数 =====================
DEFAULT_MTU = 1500               # 常见以太网 MTU。UDP 可用载荷上限约为 MTU - 28(IP20+UDP8)
RECV_QUEUE_MAXLEN = 32768        # 接收队列上限（按“包”为单位）

# pkt_type 定义（仅保留最小必要集合；你可根据业务继续扩展，但收包计数只对 PKT_ACK_PACKET 生效）
PKT_DATA = 0
PKT_RES = 1
PKT_SV_DATA = 2
PKT_TERM = 3
PKT_ACK_PACKET = 4  # ACK 包类型：收到后用于 _total_packets_received 计数

# ===================== 日志器 =====================
logger = log.getLogger("NetGearUDP")
logger.setLevel(log.DEBUG)
if not logger.handlers:
    _h = log.StreamHandler()
    _f = log.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    _h.setFormatter(_f)
    logger.addHandler(_h)


class NetGearUDP:
    """
    仅 UDP 的轻量封装：
      - receive_mode=False: 作为发送端（也会起接收线程，用于被动接收 ACK/数据）
      - receive_mode=True : 作为接收端（bind 并接收；如需回包，使用最近对端地址）
    API：
      - send(data: bytes, pkt_type: int=PKT_DATA) -> None
      - recv() -> Optional[Dict[str, Any]]  # {'pkt_type': int, 'data': bytes}
      - get_stats() -> Dict[str, int]       # {'total_packets_received', 'sent_packets'}
      - close() -> None
    约束：
      - 默认 data 必须能装进一个 UDP 包（会检查并在超限时抛出 ValueError）
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
        # ---------------- 新变量：基础配置 ----------------
        self._logging = bool(logging)                       # 是否打印日志
        self._addr = address                                # 本端绑定/对端发送的地址
        self._port = int(port)                              # 端口
        if protocol.lower() != "udp":
            raise ValueError("NetGearUDP 仅支持 protocol='udp'。")

        # ---------------- 新变量：运行参数（保留必要选项） ----------------
        self._mtu = int(options.get("mtu", DEFAULT_MTU))    # MTU；用于计算单包最大负载
        self._recv_buf_size = int(options.get("recv_buffer_size", 16 * 1024 * 1024))
        self._send_buf_size = int(options.get("send_buffer_size", 16 * 1024 * 1024))
        self._queue_maxlen = int(options.get("queue_maxlen", RECV_QUEUE_MAXLEN))

        # ---------------- 新变量：统计字段（按你的口径，仅 2 个） ----------------
        self._sent_packets = 0              # 发送的 UDP 包总数（每成功 sendto 一次 +1）
        self._total_packets_received = 0    # 仅当收到 pkt_type == PKT_ACK_PACKET 时 +1

        # ---------------- 新变量：接收与对端状态 ----------------
        self._recv_mode = bool(receive_mode)   # True 表示以接收为主（bind）；False 表示以发送为主（默认直接发往 address:port）
        self._queue: deque = deque(maxlen=self._queue_maxlen)  # 接收队列，元素是 dict{'pkt_type','data'}
        self._peer_addr: Optional[Tuple[str, int]] = None       # 最近一个对端地址（用于回包）

        # ---------------- 新变量：网络套接字与线程控制 ----------------
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 注：SO_RCVBUF/SO_SNDBUF 具体生效值由内核决定
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf_size)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_buf_size)
            self._sock.setblocking(False)
        except Exception as e:
            if self._logging:
                logger.warning(f"Socket option setting failed: {e}")

        self._terminate = False  # 控制接收线程的退出

        # 角色初始化：接收端 bind；发送端保存对端地址
        if self._recv_mode:
            self._sock.bind((self._addr, self._port))
            if self._logging:
                logger.info(f"[UDP] Bind at {self._addr}:{self._port}")
        else:
            self._peer_addr = (self._addr, self._port)
            if self._logging:
                logger.info(f"[UDP] Send to {self._addr}:{self._port}")

        # ---------------- 新增代码块：启动接收线程 ----------------
        # 功能：统一从 UDP socket 非阻塞读包，将 {'pkt_type','data'} 推入队列；
        #      若 pkt_type == PKT_ACK_PACKET，则递增 _total_packets_received 计数。
        self._rx_thread = threading.Thread(target=self._io_loop, name="UDP-RX", daemon=True)
        self._rx_thread.start()

    # ---------------- 新增函数：接收线程主循环 ----------------
    # 功能：select 监听 socket 可读；收包并解析 1 字节 pkt_type；其余作为 data；
    #      若是 ACK 包则计数 +1；最后将字典入队供上层调用 recv()。
    def _io_loop(self) -> None:
        while not self._terminate:
            try:
                import select
                ready = select.select([self._sock], [], [], 0.1)
                if not ready[0]:
                    continue

                pkt, peer = self._sock.recvfrom(65535)
                if not pkt:
                    continue

                # 记录最近对端，便于接收端回包
                self._peer_addr = peer

                # 解析 1 字节自定义头：pkt_type(u8)
                pkt_type = pkt[0]
                data = pkt[1:]

                # 仅当收到 ACK 包时进行总计数
                if pkt_type == PKT_ACK_PACKET:
                    self._total_packets_received += 1

                # 以“包”为单位上抛
                if len(self._queue) < self._queue_maxlen:
                    self._queue.append({
                        "pkt_type": int(pkt_type),
                        "data": data,
                    })

            except socket.error as e:
                # 非阻塞读常见 EAGAIN/EWOULDBLOCK
                err = getattr(e, "errno", None)
                if err in (socket.EAGAIN, socket.EWOULDBLOCK):
                    time.sleep(0.0001)
                    continue
                if not self._terminate and self._logging:
                    logger.error(f"recv error: {e}")
            except Exception as e:
                if not self._terminate and self._logging:
                    logger.error(f"recv error: {e}")

    # ---------------- 新增/重写函数：发送单包 ----------------
    # 功能：仅支持单包发送（不分片）；检查 data 尺寸是否超过单包上限（= mtu - 28 - 1）
    # 参数：
    #   - frame: 待发送的原始字节数据（bytes/bytearray/memoryview）
    #   - pkt_type: 自定义包类型（u8）
    # 行为：
    #   - 发送内容 = [pkt_type(1B)] + data
    #   - 成功发送一次，_sent_packets += 1
    def send(self, frame: Union[bytes, bytearray, memoryview], pkt_type: int = PKT_DATA) -> None:
        if frame is None:
            return

        if self._peer_addr is None and not self._recv_mode:
            # 作为发送端，初始化时应已指定对端；若为空说明未配置正确
            raise RuntimeError("No peer address configured for sender.")

        # 计算单个 UDP 包的最大数据负载:
        # 可用载荷 = MTU - (IP头20 + UDP头8) - 自定义头1 = MTU - 29
        max_payload = self._mtu - 29
        if max_payload <= 0:
            raise ValueError(f"MTU 设置过小：mtu={self._mtu}，无法发送任何负载。")

        data = bytes(frame)
        if len(data) > max_payload:
            # 按你的约束：默认 data 应该装得下；若不满足，直接报错退出（不做分片/截断）
            raise ValueError(
                f"数据过大：len(data)={len(data)} 超过单包上限 {max_payload}（= mtu({self._mtu}) - 28(IP+UDP) - 1(pkt_type)）。"
            )

        # 组包：pkt_type(u8) + data
        out = bytes([int(pkt_type) & 0xFF]) + data

        # 发送至对端（发送端用 sendto；接收端若想回包，需依赖 _peer_addr）
        try:
            if self._peer_addr:
                self._sock.sendto(out, self._peer_addr)
            else:
                # 作为纯接收端但还未收到过对端地址时，不知道往哪发
                raise RuntimeError("Receiver has no known peer address to send to yet.")
            self._sent_packets += 1
        except Exception as e:
            if self._logging:
                logger.warning(f"send error: {e}")
            raise

    # ---------------- 新增/重写函数：取出一包 ----------------
    # 功能：非阻塞返回队列头部的一个包；空队列返回 None
    # 返回：None 或 dict{'pkt_type': int, 'data': bytes}
    def recv(self) -> Optional[Dict[str, Any]]:
        if self._queue:
            return self._queue.popleft()
        return None

    # ---------------- 新增/重写函数：关闭资源 ----------------
    # 功能：退出接收线程并关闭 socket
    def close(self) -> None:
        self._terminate = True
        try:
            if hasattr(self, "_rx_thread") and self._rx_thread.is_alive():
                self._rx_thread.join(timeout=1.0)
        finally:
            try:
                self._sock.close()
            except Exception:
                pass

    # ---------------- 新增/重写函数：统计 ----------------
    # 功能：仅返回两项：接收 ACK 包的总数 与 发送包总数
    def get_stats(self) -> Dict[str, int]:
        return {
            "total_packets_received": int(self._total_packets_received),
            "sent_packets": int(self._sent_packets),
        }
