#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from collections import deque
from typing import Optional

# 统一入口：保持你的原始导入，不改路径/名称
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 保持你原来“写死”的常量，不做任何修改 ======
# ENCODED_DIR = "/data/data/com.termux/files/usr/var/lib/proot-dis...fs/ubuntu/root/vidgear/dataset/"  # 存放编码帧的目录（每个文件名形如 frame_XXXX）
ENCODED_DIR = "/home/wxk/workspace/nsdi/Viduce/net/vidgear/client_temp"
RUN_SECONDS = 300                      # 运行时间（5分钟）
PORT = 5556

RECEIVER_ADDR = "172.27.149.174"       # client 本机地址（用于绑定）
SERVER_ADDR = "114.212.86.152"         # server 地址
SERVER_PORT = 5557
# # 本机测试配置
# RECEIVER_ADDR = "114.212.86.152"       # 接收端公网/内网可达地址
# SERVER_ADDR = "127.0.0.1"              # server地址
# SERVER_PORT = 5557                     # server端口

# ====== 新增：pkt_type 常量（用于识别 DATA/RES），不影响你的既有常量 ======
try:
    # 与底层 UDP 实现对齐的常量；若不可用则兜底为默认值
    from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM = 0, 1, 2, 3

# ====== 新增：发送轮询间隔（30ms） ======
SEND_INTERVAL_SEC = 0.03
"""
SEND_INTERVAL_SEC: 发送轮询的时间间隔（单位：秒）。每隔 30ms 检查一次 FIFO 队列，若有文件则发送队首文件。
"""

def _ensure_dir(d: str) -> None:
    """辅助函数：确保目录存在"""
    os.makedirs(d, exist_ok=True)

def _save_frame_bytes(dirpath: str, frame_id: int, data: bytes) -> str:
    """
    新增功能块：将收到的字节保存为 frame_{frame_id}.bin
    返回：保存后的完整路径
    """
    fname = f"frame_{frame_id}.bin"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

def _parse_frame_id_from_name(fname: str) -> Optional[int]:
    """辅助函数：从文件名中解析 frame_id（形如 frame_123.bin）"""
    base = os.path.basename(fname)
    if base.startswith("frame_") and base.endswith(".bin"):
        mid = base[len("frame_"):-len(".bin")]
        try:
            return int(mid)
        except Exception:
            return None
    return None

def main():
    """
    主流程（修改后的核心逻辑）：
      1) 持续监听接收服务器下发的帧：
         - 若 pkt_type==PKT_SV_DATA：落盘到 ENCODED_DIR，并将文件名入 FIFO 队列
         - 若 pkt_type==PKT_RES ：打印 YOLO 结果（不落盘）
      2) 每隔 30ms：若 FIFO 非空，发送队首（最早收到）的那个文件，并从磁盘删除
    """
    _ensure_dir(ENCODED_DIR)

    # 以接收模式绑定本地端口：服务器先发数据给本机 -> 学到对端地址 -> 可回发
    net = NetGear(
        address="0.0.0.0",           # 本地任意地址绑定（不改你的 RECEIVER_ADDR）
        port=PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1400,
        recv_buffer_size=32 * 1024 * 1024,
        send_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

    # 新增：向server发送的网络连接
    server_net = NetGear(
        address=SERVER_ADDR,
        port=SERVER_PORT,
        protocol="udp",
        receive_mode=False,           # 发送模式
        logging=True,
        mtu=1400,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

    # 新增：FIFO 队列，仅保存“文件名”（先进先出）
    fifo = deque(maxlen=1 << 20)

    start = time.time()
    last_send_ts = 0.0
    received_files = 0  # 新增：统计收到并落盘的文件数
    forwarded_files = 0 # 新增：统计转发出去的文件数

    try:
        while True:
            # === A) 接收环节：把接收缓冲尽量清空 ===
            item = net.recv()  # None 或 {'pkt_type','frame_id','data'}
            while item is not None:
                try:
                    pkt_type = int(item.get("pkt_type"))
                    frame_id = int(item.get("frame_id"))
                    data: bytes = item.get("data", b"")
                except Exception:
                    # 兼容极端情形（旧实现返回裸 bytes），此处直接忽略
                    item = net.recv()
                    continue

                if pkt_type == PKT_SV_DATA:
                    # 收到服务器下发的文件：落盘 + 入队
                    path = _save_frame_bytes(ENCODED_DIR, frame_id, data)
                    fifo.append(os.path.basename(path))
                    received_files += 1
                    print(f"[Client] Received file {received_files}: frame_id={frame_id}, size={len(data)}, saved to {os.path.basename(path)}")
                    if received_files % 100 == 0:
                        print(f"[Client] Total Received={received_files}, FIFO={len(fifo)}")

                elif pkt_type == PKT_RES:
                    # 收到服务器回传的 YOLO 结果：打印
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = f"<{len(data)} bytes>"
                    print(f"[Client] YOLO result for frame_id={frame_id}: {text}")

                # 继续清空接收队列
                item = net.recv()

            # === B) 发送环节：每 30ms 检查一次 FIFO ===
            now = time.time()
            if now - last_send_ts >= SEND_INTERVAL_SEC:
                last_send_ts = now
                if fifo:
                    fname = fifo.popleft()
                    fpath = os.path.join(ENCODED_DIR, fname)
                    try:
                        with open(fpath, "rb") as f:
                            payload = f.read()
                        # 复用文件名中的 frame_id；若解析失败则让底层自增
                        fid = _parse_frame_id_from_name(fname)
                        server_net.send(payload, pkt_type=PKT_DATA, frame_id=fid)
                        forwarded_files += 1
                        print(f"[Client] Forwarded to server: {fname} (frame_id={fid}, size={len(payload)})")
                        # 发送成功，删除文件
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
                        if forwarded_files % 100 == 0:
                            print(f"[Client] Total Forwarded={forwarded_files}, FIFO={len(fifo)}")
                    except FileNotFoundError:
                        # 文件可能被外部清理，忽略
                        pass
                    except Exception as e:
                        print(f"[Client] Send error for {fname}: {e}")

            # 运行时长控制
            if RUN_SECONDS > 0 and (time.time() - start) >= RUN_SECONDS:
                break

            # 轻微休眠，避免空转
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        print(f"[Client] Done. received={received_files}, forwarded={forwarded_files}, elapsed={elapsed:.2f}s")
        if hasattr(net, "get_stats"):
            print(f"[Client] Receiver stats: {net.get_stats()}")
        if hasattr(server_net, "get_stats"):
            print(f"[Client] Sender stats: {server_net.get_stats()}")
        net.close()
        server_net.close()

if __name__ == "__main__":
    main()
