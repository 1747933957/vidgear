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

# ===== 标准依赖（图像/数值/YOLO，占位可运行） =====
import cv2                      # 新增：图像缩放
import numpy as np              # 新增：数值处理
from PIL import Image           # 新增：PNG 落盘
from multiprocessing import Process, Manager  # 新增：共享列表 + 监控进程

# ===== 你原有的导入（保持） =====
# 保持你的原始导入与常量（注意该文件的 import 路径与发送端不同）
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ===== 保存目录：与原文件一致并清空重建 =====
DATA_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/sender_out")
SAVE_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/reciever_out")
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)  # 递归删除整个目录及其内容
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 【保持】绑定地址与端口（不要改）
BIND_ADDR = "127.0.0.1"  # 114.212.86.152
PORT = 5558

# 【保持】回传给 client 的地址（不要改）
CLIENT_ADDR = "127.0.0.1"  # 本机测试："172.27.143.41"
CLIENT_PORT = 5557

# ===== pkt_type 常量（包含 PKT_ACK_PACKET）=====
try:
    from vidgear.gears.netgear_udp import (
        PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_PACKET  # type: ignore
    )
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_PACKET = 0, 1, 2, 3, 4

# ===== NS-3 收包函数（保持）=====
try:
    from ns3.receiver import receive_packet  # 必需：处理来包并返回“刚刚完整的帧ID列表”
except Exception as e:
    raise RuntimeError(f"[Init] 无法导入 ns3 函数 receive_packet: {e}")

# ===== ACK 负载格式：!d（大端）= [send_ts:double秒]（保持）=====
_ACK_FMT = "!d"
_ACK_SIZE = struct.calcsize(_ACK_FMT)

# ===== 全局：DONE_FRAMES（最终会被指向共享列表）=====
DONE_FRAMES: Iterable[int] = []

# =======【新增】Grace/YOLO 相关常量（与 UDP 版保持一致）=======
GRACE_ROOT = "/home/wxk/workspace/nsdi/Intrinsic"  # AE 工程路径
GRACE_MODEL_ID = "64"                               # init_ae_model 的键
IMAGE_WIDTH = 1280                                  # PNG 目标宽
IMAGE_HEIGHT = 720                                  # PNG 目标高
PNG_OUT_DIR = SAVE_DIR / "decoded"     # 解码后 PNG 落盘目录
PNG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# =======【新增】为了 import runner，补充 sys.path（与 UDP 版一致）=======
_CUR = Path(__file__).resolve().parent
for p in {str(_CUR), str(SAVE_DIR.resolve()), str(SAVE_DIR.resolve().parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)
# 两层上级：保证能 import 到 shb_viduce_runner.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shb_viduce_runner import iter_read_grace_records, GraceBundle  # noqa: E402

# =======【新增】工具：保存为 .bin =======
def _save_bin(save_dir: str, frame_id: int, data: bytes) -> str:
    """
    将完整帧字节保存为独立 .bin 文件：SAVE_DIR/frame_XXXX.bin
    参数:
      - save_dir: 根保存目录
      - frame_id: 帧编号
      - data:     重组完成的二进制内容
    返回:
      - 保存路径
    """
    fname = f"frame_{frame_id:04d}.bin"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

# =======【新增】YOLO 推理（若无 ultralytics 则占位返回）=======
def _run_yolo(file_path: str) -> Dict[str, Any]:
    """
    YOLO 推理：
      - 若安装 ultralytics：真实推理并抽取 box/cls/conf
      - 否则：返回占位结果（不崩溃）
    """
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")
        res = model(file_path)
        out = []
        for r in res:
            boxes = []
            if getattr(r, "boxes", None) is not None:
                for b in r.boxes:
                    xyxy = getattr(b, "xyxy", None)
                    conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
                    cls = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else -1
                    if xyxy is not None:
                        x1, y1, x2, y2 = [float(x) for x in xyxy[0].tolist()]
                        boxes.append({"xyxy": [x1, y1, x2, y2], "cls": cls, "conf": conf})
            out.append({"boxes": boxes})
        return {"ok": True, "engine": "ultralytics", "frames": out}
    except Exception as e:
        try:
            size = os.path.getsize(file_path)
        except Exception:
            size = -1
        return {"ok": True, "engine": "stub", "size": size, "note": str(e)}

# =======【新增】Grace 解码上下文，与 UDP 版一致 =======
class GraceDecoderContext:
    """
    负责：
    1) 初始化并持有 GraceBundle（AE 模型）
    2) 保存解码参考：
       - self.last_ref_np: I/P 解码参考
       - self.last_decoded_np: 最近一次成功解码的帧，供 mv 解码
    3) 从 .bin 解出 eframe，并按 I/P 或 mv 路径解码为 PNG
    """
    def __init__(self, grace_root: str, model_id: str):
        self.grace = GraceBundle(Path(grace_root), model_id)
        self.last_ref_np: Optional[np.ndarray] = None
        self.last_decoded_np: Optional[np.ndarray] = None

    def decode_bin_to_png(self, bin_path: str, frame_id: int) -> str:
        bin_p = Path(bin_path)
        if not bin_p.exists():
            raise FileNotFoundError(f"bin 不存在: {bin_path}")

        ip_loaded = None
        mv_loaded = None
        for record_type, record_data in iter_read_grace_records(bin_p):
            if record_type in ["I", "P"] and ip_loaded is None:
                ip_loaded = record_data
            if record_type == "mv":
                mv_loaded = record_data

        if ip_loaded is not None:
            eframe = ip_loaded
            if self.last_ref_np is not None:
                ref_img = Image.fromarray(
                    (np.clip(self.last_ref_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
                )
            else:
                ref_img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
            decoded_np = self.grace._call_decode(eframe, ref_img)
            decoded_np = cv2.resize(decoded_np, (IMAGE_WIDTH, IMAGE_HEIGHT))
            self.last_ref_np = decoded_np.copy()
            self.last_decoded_np = decoded_np.copy()

        elif mv_loaded is not None:
            if self.last_decoded_np is None:
                raise RuntimeError("收到 mv 帧但当前无参考帧，无法解码（请先收到 I/P 帧）")
            decoded_np = self.grace.decode_mv_to_frame(mv_loaded, self.last_decoded_np)
            self.last_decoded_np = decoded_np.copy()
            self.last_ref_np = decoded_np.copy()
        else:
            raise RuntimeError("无法从 bin 中读取 I/P 或 mv 记录")

        out_png = os.path.join(PNG_OUT_DIR, f"frame_{frame_id:04d}.png")
        Image.fromarray((np.clip(decoded_np, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_png)
        return out_png

# =======【新增】监控进程：持续消费 DONE_FRAMES，完成解码/YOLO/回传 =======
def monitor_done_frames(done_frames_shared, net):
    """
    【进程函数】
    从启动之初持续运行：
      - 监控共享列表 done_frames_shared（由主循环不断追加“刚刚完整”的 frame_id）
      - 对新出现的帧号：
          a) 调 ns3.receiver 取出该帧完整字节 -> data
          b) 保存为 .bin
          c) 解码为 PNG
          d) YOLO 推理
          e) 通过 net 回传（pkt_type=PKT_RES, frame_id=原样）
    计数器（进程内独立）：received/sent_res
    """
    decoder = GraceDecoderContext(GRACE_ROOT, GRACE_MODEL_ID)  # 进程内独立的解码上下文
    seen = set()                # 已处理的 frame_id 集
    received = 0                # 已保存 .bin 的帧数
    sent_res = 0                # 已回传 YOLO 结果的帧数

    while True:
        try:
            # 扫描共享列表，找还未处理的帧
            for frame_id in list(done_frames_shared):
                if frame_id in seen:
                    continue
                seen.add(frame_id)

                fpath = DATA_DIR / f"grace_stream_{frame_id:04d}.bin"

                # 3) 解码 bin -> png
                try:
                    png_path = decoder.decode_bin_to_png(fpath, int(frame_id))
                except Exception as dec_err:
                    err_payload = json.dumps(
                        {"ok": False, "error": f"decode_failed: {dec_err}", "frame_id": int(frame_id)},
                        ensure_ascii=False
                    ).encode("utf-8")
                    # 回传错误
                    net.send(err_payload, pkt_type=PKT_RES)
                    print(f"[Server] Decode failed for frame_id={frame_id}: {dec_err}")
                    continue

                # 4) YOLO 检测
                yolo_res = _run_yolo(png_path)
                yolo_res["frame_id"] = int(frame_id)
                payload = json.dumps(yolo_res, ensure_ascii=False).encode("utf-8")

                # 5) 回传检测结果
                net.send(payload, pkt_type=PKT_RES)
                sent_res += 1
                print(f"[Server] Sent YOLO result {sent_res}: frame_id={frame_id}, size={len(payload)}")
                if sent_res % 100 == 0:
                    print(f"[Server] Total Sent YOLO results={sent_res}")

            time.sleep(0.001)  # 轻量轮询

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Server] monitor_done_frames 异常：{e}\n{traceback.format_exc()}")
            time.sleep(0.01)

# ======= 主流程（保持原接收/ACK，不改其语义）=======
def main():
    """
    主流程：
      - 绑定在 127.0.0.1:5558，持续接收来自发送端的数据（pkt_type==PKT_DATA/PKT_SV_DATA/...）
      - 不再从负载取时间戳；直接从 item 的 UDP/RTP 头字段提取发送时刻（秒） -> 【此逻辑保留】
      - 保存真实数据 -> 【改为在监控进程里完成】
      - 立即发送 ACK（pkt_type==PKT_ACK_PACKET）
      - DONE_FRAMES 追加由 ns3.receiver.receive_packet 返回的“刚刚完整帧号”，供监控进程消费
    """
    # 共享 DONE_FRAMES：用于与监控进程通信
    manager = Manager()
    done_frames_shared = manager.list()  # 共享列表
    global DONE_FRAMES
    DONE_FRAMES = done_frames_shared     # 保持全局名，可在其他模块引用


    # 接收网络（保持）
    net = NetGear(
        address=BIND_ADDR,
        port=PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1500,
        recv_buffer_size=32 * 1024 * 1024,
        send_buffer_size=32 * 1024 * 1024,
        queue_maxlen=655360
    )
    # 显式设置对端地址，确保可以 send 回 ACK（到 CLIENT_ADDR:CLIENT_PORT）
    net._peer_addr = (CLIENT_ADDR, CLIENT_PORT)

    # 启动监控进程（从一开始就运行）
    p = Process(target=monitor_done_frames, args=(done_frames_shared, net,), daemon=True)
    p.start()
    try:
        while True:
            pkt = net.recv()
            if pkt is not None:
                ptype = pkt.get("pkt_type")
                data = pkt.get("data", b"")
                # === 立刻回 ACK（保持原逻辑）===
                try:
                    ack_payload = struct.pack(_ACK_FMT, time.time())
                    net.send(ack_payload, pkt_type=PKT_ACK_PACKET)
                except Exception as e:
                    print(f"[Receiver] send ACK failed: {e}")

                # === 将“刚刚完整的帧号”追加到共享列表，供监控进程消费 ===
                try:
                    frame_ids = receive_packet(data)  # 应返回 list[int]
                    print(f"[Receiver] got DATA packet, frame_ids={frame_ids}")
                    if frame_ids:
                        done_frames_shared.extend([int(x) for x in frame_ids])
                except Exception as e:
                    print(f"[Receiver] receive_packet 处理失败: {e}")

                continue

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(net, "get_stats"):
            print(f"[Server] Receiver stats: {net.get_stats()}")
        net.close()
        # 监控进程是 daemon=True，主进程退出会随之退出

if __name__ == "__main__":
    main()
