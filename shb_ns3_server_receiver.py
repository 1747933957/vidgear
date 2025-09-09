#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import struct
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable
import threading                 # === 新增/保留：线程基元 ===
import queue                     # === 新增/保留：有界队列 ===
import torch

# ===== 标准依赖（图像/数值） =====
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Manager

# ===== 你原有的导入（保持） =====
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ===== 绑定地址与端口 =====
BIND_ADDR = "127.0.0.1"  # 127.0.0.1, 114.212.86.152
PORT = 5558

# ===== 回传给 client 的地址 =====
CLIENT_ADDR = "127.0.0.1"  # "172.27.144.86" "127.0.0.1""172.27.155.100"
CLIENT_PORT = 5559

# ===== 保存目录（保持你的默认路径；首次运行清空重建） =====
DATA_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/sender_out")
SAVE_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/reciever_out")
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== 服务器端计时日志 =====
SERVER_TIME_LOG = SAVE_DIR / f"server_times_{CLIENT_ADDR}.csv"
with SERVER_TIME_LOG.open("w", newline="") as f:
    import csv as _csv
    w = _csv.writer(f)
    w.writerow(["frame_id","grace_decode_ms","yolo_ms"])

def _server_log_row(frame_id: int, dec_ms: int, yolo_ms: int):
    with SERVER_TIME_LOG.open("a", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow([int(frame_id), int(dec_ms), int(yolo_ms)])


# ===== NS-3 收包函数（保持）=====
try:
    from ns3.receiver import receive_packet
except Exception as e:
    raise RuntimeError(f"[Init] 无法导入 ns3 函数 receive_packet: {e}")

# ===== ACK/RES 头部格式（保持）=====
_ACK_FMT = "!id"
_ACK_SIZE = struct.calcsize(_ACK_FMT)
RES_MAGIC = b'RSID'
RES_HDR_FMT = '!4sI'  # magic(4s) + frame_id(u32)


# ===== import runner（保持）=====
_CUR = Path(__file__).resolve().parent
for p in {str(_CUR), str(SAVE_DIR.resolve()), str(SAVE_DIR.resolve().parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shb_viduce_runner import iter_read_grace_records, GraceBundle  # noqa: E402

# ===== CUDA 可用标志（保持）=====
try:
    _CAN_SYNC_CUDA = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
except Exception:
    _CAN_SYNC_CUDA = False

def _sync_cuda():
    """仅当主线程已初始化 CUDA 时尝试同步，避免后台线程触发懒初始化抖动。"""
    if _CAN_SYNC_CUDA:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def _save_bin(save_dir: str, frame_id: int, data: bytes) -> str:
    """
    保存完整帧为 .bin（当前流程默认由 ns3.receiver 自行落盘；如需自行保存可调用）
    参数:
      - save_dir: 目录
      - frame_id: 帧号
      - data:     二进制
    返回: 保存路径
    """
    fname = f"frame_{frame_id:04d}.bin"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

# =========================
# 两线程流水线（保留） + 接入 YOLOContext
# =========================

# --- 全局对象（带注释） ---
_q_decode = queue.Queue(maxsize=16)  # 有界队列：待解码的 frame_id；背压上游
_q_detect = queue.Queue(maxsize=16)  # 有界队列：待 YOLO 的 (frame_id, png_path, dec_ms)
_stop_event = threading.Event()      # 结束信号


def main():
    """
    主流程：
      - 接收 NS-3 送来的 UDP 包
      - receive_packet(data) → “刚刚完整的帧号列表”
      - 对每个新完成的帧 push -> _q_decode（阻塞时形成背压）
      - 两个工作线程常驻：_decoder_worker, _yolo_worker
      - 【新增】在此处一次性加载 YOLOContext，并传入 YOLO 工作线程
    """
    manager = Manager()
    done_frames_shared = manager.list()

    net = NetGear(
        address=BIND_ADDR,
        port=PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1500,
        recv_buffer_size=32 * 1024 * 1024,
        send_buffer_size=32 * 1024 * 1024,
        queue_maxlen=655360,
        slow_log_dir="./tmp/logs_receiver",          # 这端一般不会写 slow 日志，但可保留
        slow_weights_path="./tmp/slow_module_weights.json",
        rtcp_interval_ms=100,
    )
    net._peer_addr = (CLIENT_ADDR, CLIENT_PORT)

    # === 在实例创建后，从实例上“绑定” pkt 常量到本地变量 ===
    # 这样后面代码仍可以用 PKT_DATA 等名字，不改流程。
    global PKT_DATA, PKT_SV_DATA, PKT_RES, PKT_TERM, PKT_ACK_FRAME, PKT_ACK_PACKET, PKT_PING, PKT_PONG
    PKT_DATA = net.PKT_DATA
    PKT_SV_DATA = net.PKT_SV_DATA
    PKT_RES = net.PKT_RES
    PKT_TERM = net.PKT_TERM
    PKT_ACK_FRAME = net.PKT_ACK_FRAME
    PKT_ACK_PACKET = net.PKT_ACK_PACKET
    PKT_PING = net.PKT_PING
    PKT_PONG = net.PKT_PONG


    try:
        while True:
            pkt = net.recv()
            if pkt is not None:
                ptype = pkt.get("pkt_type")
                data = pkt.get("data", b"")
                if ptype in {PKT_DATA, PKT_SV_DATA}:
                    frame_ids, packet_id = receive_packet(data)
                    # 立即 ACK 数据包（包级 ACK）
                    if packet_id:
                        ack_payload = struct.pack(_ACK_FMT, int(packet_id), time.time())
                        net.send(ack_payload, pkt_type=PKT_ACK_PACKET)


                continue

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        # 尝试等队列清空（可选）
        try:
            _q_decode.join()
            _q_detect.join()
        except Exception:
            pass
        if hasattr(net, "get_stats"):
            print(f"[Server] Receiver stats: {net.get_stats()}")
        net.close()

if __name__ == "__main__":
    main()
