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

# ===== 保存目录（保持你的默认路径；首次运行清空重建） =====
DATA_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/sender_out")
SAVE_DIR = Path("/data/wxk/workspace/mirage/dataset/video000/Viduce/reciever_out")
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===== 服务器端计时日志 =====
SERVER_TIME_LOG = SAVE_DIR / "server_times.csv"
with SERVER_TIME_LOG.open("w", newline="") as f:
    import csv as _csv
    w = _csv.writer(f)
    w.writerow(["frame_id","grace_decode_ms","yolo_ms"])

def _server_log_row(frame_id: int, dec_ms: int, yolo_ms: int):
    with SERVER_TIME_LOG.open("a", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow([int(frame_id), int(dec_ms), int(yolo_ms)])

# ===== 绑定地址与端口 =====
BIND_ADDR = "114.212.86.152"  # 127.0.0.1, 114.212.86.152
PORT = 5558

# ===== 回传给 client 的地址 =====
CLIENT_ADDR = "172.27.155.100"  # "172.27.144.86" "127.0.0.1""172.27.155.100"
CLIENT_PORT = 5559

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

# ===== 全局：DONE_FRAMES（最终会被指向共享列表）（保持）=====
DONE_FRAMES: Iterable[int] = []

# ===== Grace/YOLO 相关常量（保持）=====
GRACE_ROOT = "/home/wxk/workspace/nsdi/Intrinsic"
GRACE_MODEL_ID = "64"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
PNG_OUT_DIR = SAVE_DIR / "decoded"
PNG_OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ==========================================================
# === 新增：YOLOContext —— 一次加载，反复复用（关键改动）===
# ==========================================================
class YOLOContext:
    """
    YOLO 模型上下文：
      - 程序启动时只实例化一次（避免每帧重复加载权重）
      - 提供 run(file_path) 方法进行推理
      - 若未安装 ultralytics，则使用占位 stub，但同样只初始化一次
    """
    def __init__(self, device: Optional[str] = None, model_name: str = "yolov8n.pt"):
        # === 新增变量：是否使用真实 YOLO 引擎 ===
        self.use_ultralytics: bool = False  # True 表示使用 ultralytics 引擎
        # === 新增变量：模型对象 or 占位标志 ===
        self.model = None                   # 若 use_ultralytics=True，则为 YOLO 模型实例
        # === 新增变量：设备字符串（'cuda' 或 'cpu'）===
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # === 新增变量：模型权重名称 ===
        self.model_name = model_name

        try:
            # 尝试加载 ultralytics YOLO（只做一次）
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(self.model_name)
            # 提前构造一次空张量以触发 CUDA 上下文（可减少首次推理抖动）
            _ = torch.zeros(1, device=self.device)
            self.use_ultralytics = True
            print(f"[YOLO] 模型已加载: {self.model_name} @ {self.device}")
        except Exception as e:
            # 退回占位模式（不抛异常，保证服务不中断）
            self.use_ultralytics = False
            self.model = {"stub": True, "note": str(e)}
            print(f"[YOLO] 未启用真实 YOLO（占位模式）：{e}")

    def run(self, file_path: str) -> Dict[str, Any]:
        """
        执行推理。若使用真实 YOLO，则返回 boxes 等结构；
        否则返回占位数据（包含文件大小）。
        """
        if self.use_ultralytics and self.model is not None:
            try:
                res = self.model(file_path)
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
                return {"ok": False, "engine": "ultralytics", "error": str(e)}
        else:
            try:
                size = os.path.getsize(file_path)
            except Exception:
                size = -1
            return {"ok": True, "engine": "stub", "size": size, "note": self.model.get("note", "") if isinstance(self.model, dict) else ""}

# ==========================================================
# === Grace 解码上下文（保持原有语义）                      ===
# ==========================================================
class GraceDecoderContext:
    """
    职责：
      1) 初始化并持有 GraceBundle（AE 模型）
      2) 管理 I/P/MV 解码参考
      3) .bin -> PNG
    """
    def __init__(self, grace_root: str, model_id: str):
        self.grace = GraceBundle(Path(grace_root), model_id)

    @staticmethod
    def _wait_png_ready(path: str, max_wait_ms: int = 800, interval_ms: int = 20) -> bool:
        deadline = time.time() + max_wait_ms / 1000.0
        while time.time() < deadline:
            try:
                with Image.open(path) as im:
                    im.verify()
                return True
            except Exception:
                time.sleep(interval_ms / 1000.0)
        return False

    def decode_bin_to_png(self, bin_path: str, frame_id: int) -> str:
        bin_p = Path(bin_path)
        if not bin_p.exists():
            raise FileNotFoundError(f"bin 不存在: {bin_path}")

        i_loaded = None
        p_loaded = None
        mv_loaded = None
        for record_type, record_data in iter_read_grace_records(bin_p):
            if record_type == "I":
                i_loaded = record_data
            if record_type == "P":
                p_loaded = record_data
            if record_type == "mv":
                mv_loaded = record_data

        if i_loaded is not None:
            eframe = i_loaded
            H, W = int(getattr(eframe, 'shapex', IMAGE_HEIGHT)), int(getattr(eframe, 'shapey', IMAGE_WIDTH))
            ref_np = np.random.rand(H, W, 3).astype(np.float32)
            ref_img = Image.fromarray((ref_np * 255.0).astype(np.uint8), mode="RGB")
            decoded_np = self.grace._call_decode(eframe, ref_img)
            decoded_np = cv2.resize(decoded_np, (IMAGE_WIDTH, IMAGE_HEIGHT))
        elif p_loaded is not None:
            eframe = p_loaded
            ref_id = int(getattr(eframe, 'ref_id', 0))
            ref_png = os.path.join(PNG_OUT_DIR, f"frame_{ref_id:04d}.png")
            if not self._wait_png_ready(ref_png, max_wait_ms=1000, interval_ms=20):
                raise RuntimeError(f"P 帧解码需要参考帧 PNG 不存在: {ref_png}")
            ref_np = np.asarray(Image.open(ref_png).convert("RGB")).astype(np.float32) / 255.0
            ref_img = Image.fromarray((np.clip(ref_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
            decoded_np = self.grace._call_decode(eframe, ref_img)
            decoded_np = cv2.resize(decoded_np, (IMAGE_WIDTH, IMAGE_HEIGHT))
        elif mv_loaded is not None:
            ref_id = frame_id - 1
            if ref_id < 0:
                raise RuntimeError("首帧为 MV 无参考帧")
            ref_png = os.path.join(PNG_OUT_DIR, f"frame_{ref_id:04d}.png")
            if not self._wait_png_ready(ref_png, max_wait_ms=1000, interval_ms=20):
                raise RuntimeError(f"MV 帧解码需要参考帧 PNG 不存在: {ref_png}")
            ref_np = np.asarray(Image.open(ref_png).convert("RGB")).astype(np.float32) / 255.0
            ref_np = np.clip(ref_np, 0.0, 1.0)
            decoded_np = self.grace.decode_mv_to_frame(mv_loaded, ref_np)
        else:
            raise RuntimeError("无法从 bin 中读取 I/P 或 mv 记录")

        out_png = os.path.join(PNG_OUT_DIR, f"frame_{frame_id:04d}.png")
        Image.fromarray((np.clip(decoded_np, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_png)
        return out_png

# =========================
# 两线程流水线（保留） + 接入 YOLOContext
# =========================

# --- 全局对象（带注释） ---
_q_decode = queue.Queue(maxsize=16)  # 有界队列：待解码的 frame_id；背压上游
_q_detect = queue.Queue(maxsize=16)  # 有界队列：待 YOLO 的 (frame_id, png_path, dec_ms)
_stop_event = threading.Event()      # 结束信号

def _decoder_worker(decoder: GraceDecoderContext):
    """
    解码线程主体：
      - 从 _q_decode 阻塞取出 frame_id
      - 调用 Grace 解码 .bin -> .png
      - 记录解码时延，放入 _q_detect
    """
    while not _stop_event.is_set():
        try:
            fid = _q_decode.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            _sync_cuda()
            t0 = time.perf_counter()
            bin_path = DATA_DIR / f"grace_stream_{int(fid):04d}.bin"
            png_path = decoder.decode_bin_to_png(str(bin_path), int(fid))
            _sync_cuda()
            t1 = time.perf_counter()
            dec_ms = int(round((t1 - t0) * 1000))
            _q_detect.put((int(fid), str(png_path), dec_ms))  # 有界，必要时阻塞，形成背压
        except Exception as e:
            print(f"[Decoder] frame {fid} 解码失败: {e}")
        finally:
            _q_detect.task_done() if False else None  # 占位，保持结构清晰
            _q_decode.task_done()

def _yolo_worker(net: NetGear, yolo_ctx: YOLOContext):
    """
    YOLO 线程主体：
      - 从 _q_detect 阻塞取出 (fid, png_path, dec_ms)
      - 使用【已加载】的 yolo_ctx 进行推理（不再每次加载模型）
      - 组装 JSON 并通过 UDP 回传
      - 记录 yolo_ms 到 server_times.csv
    """
    while not _stop_event.is_set():
        try:
            fid, png_path, dec_ms = _q_detect.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            _sync_cuda()
            ty0 = time.perf_counter()
            yolo_res = yolo_ctx.run(png_path)      # === 关键改动：复用已加载模型 ===
            _sync_cuda()
            ty1 = time.perf_counter()
            yolo_ms = int(round((ty1 - ty0) * 1000))
            _server_log_row(int(fid), int(dec_ms), int(yolo_ms))

            # 回传（与原格式一致）
            hdr = struct.pack(RES_HDR_FMT, RES_MAGIC, int(fid))
            yolo_res["frame_id"] = int(fid)
            body = json.dumps(yolo_res, ensure_ascii=False).encode("utf-8")
            payload = hdr + body
            net.send(payload, pkt_type=net.PKT_RES)  # 用实例上的常量
            print(f"[Server] Sent YOLO result: frame_id={fid}, size={len(payload)}, dec={dec_ms}ms, yolo={yolo_ms}ms")
        except Exception as e:
            print(f"[YOLO] frame {fid} 推理/回传失败: {e}")
        finally:
            _q_detect.task_done()

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
    global DONE_FRAMES
    DONE_FRAMES = done_frames_shared

    decoder = GraceDecoderContext(GRACE_ROOT, GRACE_MODEL_ID)

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
        slow_log_dir="./tmp/slow_logs_receiver",          # 这端一般不会写 slow 日志，但可保留
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

    # === 新增：只在启动时加载一次 YOLO ===
    # 新变量 yolo_ctx: YOLOContext —— 已加载的 YOLO 引擎或占位引擎
    yolo_ctx = YOLOContext(device=("cuda" if torch.cuda.is_available() else "cpu"),
                           model_name="yolov8n.pt")

    # === 启动两名固定工作线程 ===
    t_dec = threading.Thread(target=_decoder_worker, args=(decoder,), daemon=True)
    t_yolo = threading.Thread(target=_yolo_worker, args=(net, yolo_ctx), daemon=True)  # 传入 yolo_ctx
    t_dec.start()
    t_yolo.start()

    seen = set()
    print(f"[INFO] Pipeline started. decodeQ={_q_decode.maxsize}, detectQ={_q_detect.maxsize}")

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

                    # 将“刚刚完整”的帧号送入解码队列
                    if frame_ids:
                        for fid in frame_ids:
                            if fid not in seen:
                                seen.add(fid)
                                ack_payload = struct.pack(_ACK_FMT, int(fid), time.time())
                                net.send(ack_payload, pkt_type=PKT_ACK_FRAME)
                                _q_decode.put(int(fid))      # 有界队列，必要时阻塞，给系统建立背压
                                DONE_FRAMES.append(int(fid))  # 若其他模块需要观测

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
