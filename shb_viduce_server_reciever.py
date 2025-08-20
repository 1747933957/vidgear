#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改动总览：
- 接收的数据改为二进制 .bin（单帧一个文件），不再是 .png
- 复用 shb_viduce_runner.py 中的 iter_read_grace_records 与 GraceBundle，对 bin 做 I/P/mv 解码
- 解码生成 PNG 到 SAVE_DIR/decoded/frame_XXXX.png
- 对 PNG 走 YOLO 推理，结果以 pkt_type=PKT_RES 回传（frame_id 保持不变）
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import shutil
# ========= 保持你的原始导入与常量（不动）=========
from net.vidgear.vidgear.gears.unified_netgear import NetGearLike as NetGear

SAVE_DIR = "/data/wxk/workspace/mirage/dataset/video000/Viduce/reciever_out"
# 删除SAVE_DIR文件夹
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)  # 递归删除整个目录及其内容
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_SECONDS = 300  # 至少运行5分钟
PORT = 5557        # 使用不同于client的端口
BIND_ADDR = "0.0.0.0"  # 绑定所有接口，便于本地测试

# ====== 保持原有“client响应”网络配置（不动） ======
CLIENT_ADDR = "172.27.149.174" # 本机测试："127.0.0.1"
CLIENT_PORT = 5556

# 图像尺寸
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# ====== pkt_type 常量（保留） ======
try:
    from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM = 0, 1, 2, 3

# ========= 新增常量：Grace 相关 =========
# GRACE_ROOT: Intrinsic 工程根路径（与 runner 中默认一致）
GRACE_ROOT = "/home/wxk/workspace/nsdi/Intrinsic"
# GRACE_MODEL_ID: init_ae_model() 的键，默认 "64"
GRACE_MODEL_ID = "64"
# PNG_OUT_DIR: 解码后 PNG 的落盘目录
PNG_OUT_DIR = os.path.join(SAVE_DIR, "decoded")
os.makedirs(PNG_OUT_DIR, exist_ok=True)

# ========= 复用你在 shb_viduce_runner.py 里的实现 =========
# 为了“就地可用”，先把当前目录与 SAVE_DIR 的上级加入 sys.path，保证能 import 成功
_CUR = Path(__file__).resolve().parent
for p in [str(_CUR), str(Path(SAVE_DIR).resolve().parent), str(Path(SAVE_DIR).resolve())]:
    if p not in sys.path:
        sys.path.insert(0, p)

# iter_read_grace_records: 读取 bin 中的记录流；GraceBundle: 封装 AE 解码接口
from shb_viduce_runner import iter_read_grace_records, GraceBundle  # noqa: E402

# ========= 工具函数：把 numpy(RGB,[0,1]) 转 PIL =========
def _np_to_pil(rgb_np: np.ndarray) -> Image.Image:
    """
    将 (H,W,3) float32 in [0,1] 或 uint8 的 RGB 数组转换为 PIL.Image
    """
    if not isinstance(rgb_np, np.ndarray):
        raise TypeError("rgb_np 必须是 numpy.ndarray")
    if rgb_np.dtype != np.uint8:
        arr = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr = rgb_np
    return Image.fromarray(arr, mode="RGB")

# ========= 新增类：解码上下文 =========
class GraceDecoderContext:
    """
    负责：
    1) 初始化并持有 GraceBundle（AE 模型）
    2) 保存解码参考：
       - self.last_ref_np: I/P 解码所需参考帧，作为 ref_img
       - self.last_decoded_np: 最近一次成功解码的帧，供 mv 解码时作为 prev_p2p_np
    3) 从 .bin 解出 eframe，并按 I/P 或 mv 路径解码为 PNG
    """

    def __init__(self, grace_root: str, model_id: str):
        # self.grace: 你在 runner 里的 AE 封装，包含 _call_decode/_crop_np_to/decode_mv_to_frame 等
        self.grace = GraceBundle(Path(grace_root), model_id)
        # self.last_ref_np: np.ndarray or None，供 I/P 解码时作为参考
        self.last_ref_np: Optional[np.ndarray] = None
        # self.last_decoded_np: np.ndarray or None，供 mv 解码时作为参考(prev_p2p_np)
        self.last_decoded_np: Optional[np.ndarray] = None

    def _infer_target_hw_from_eframe(self, eframe) -> Optional[Tuple[int, int]]:
        """
        从 eframe 信息推断“原始目标 (H,W)”，用于解码后裁剪。
        优先：
          - I: 用 (shapey, shapex) 作为 (H,W)
          - P: 用 ipart 中的 (shapey, shapex) 作为 (H,W)
        若拿不到，返回 None（则不做裁剪）。
        """
        try:
            if getattr(eframe, "frame_type", "").upper() == "I":
                sx = int(getattr(eframe, "shapex"))
                sy = int(getattr(eframe, "shapey"))
                if sx > 0 and sy > 0:
                    return (sy, sx)
            if getattr(eframe, "frame_type", "").upper() == "P" and getattr(eframe, "ipart", None) is not None:
                ip = eframe.ipart
                sx = int(getattr(ip, "shapex", 0))
                sy = int(getattr(ip, "shapey", 0))
                if sx > 0 and sy > 0:
                    return (sy, sx)
        except Exception:
            pass
        return None

    def _blank_ref_of(self, hw: Tuple[int, int]) -> Image.Image:
        """
        基于提供的 (H,W)，构造一个黑底参考图（I/P 首帧无参考时兜底）
        """
        H, W = int(hw[0]), int(hw[1])
        H = max(H, 1); W = max(W, 1)
        return Image.new("RGB", (W, H), (0, 0, 0))

    def decode_bin_to_png(self, bin_path: str, frame_id: int) -> str:
        """
        读取 .bin → 解析记录 → 分支解码 → 写出 PNG 文件（路径返回）

        参数:
          - bin_path: .bin 文件路径
          - frame_id: 当前帧编号（仅用于输出文件命名）

        返回:
          - 生成的 PNG 完整路径
        """
        bin_p = Path(bin_path)
        if not bin_p.exists():
            raise FileNotFoundError(f"bin 不存在: {bin_path}")

        # ========= 解析 I/P / mv =========
        # I/P：取第一次出现的 I 或 P
        ip_loaded = None
        # mv：取最后一条 mv（按你的示例实现）
        mv_loaded = None

        for record_type, record_data in iter_read_grace_records(bin_p):
            if record_type in ["I", "P"] and ip_loaded is None:
                ip_loaded = record_data
            if record_type == "mv":
                mv_loaded = record_data

        # ========= 分支解码：I/P 优先；否则 mv =========
        if ip_loaded is not None:
            eframe = ip_loaded
            # 参考帧 ref_img：优先用最近解码帧，否则用黑底兜底
            # target_hw = self._infer_target_hw_from_eframe(eframe)
            target_hw = (IMAGE_HEIGHT, IMAGE_WIDTH)
            if self.last_ref_np is not None:
                ref_img = _np_to_pil(self.last_ref_np)
            else:
                # 首帧没有参考，按 eframe 推断的尺寸造一张黑底参考
                ref_img = self._blank_ref_of(target_hw if target_hw is not None else (720, 1280))

            # === 你的解码片段（I/P）等价实现 ===
            # loaded_data = ip_loaded
            # eframe = loaded_data
            decoded_np = self.grace._call_decode(eframe, ref_img)  # noqa: SLF001 (按你的代码复用内部方法)

            # 如果有目标尺寸，则裁剪回原始尺寸
            if target_hw is not None:
                decoded_np = cv2.resize(decoded_np, (IMAGE_WIDTH, IMAGE_HEIGHT))
                # decoded_np = self.grace._crop_np_to(decoded_np, target_hw)

            # 更新参考
            self.last_ref_np = decoded_np.copy()
            self.last_decoded_np = decoded_np.copy()

        elif mv_loaded is not None:
            eframe = mv_loaded
            # mv 解码需要上一帧的像素图作为 prev_p2p_np
            if self.last_decoded_np is None:
                # 没有参考无法做 mv，还原；直接抛错更清晰
                raise RuntimeError("收到 mv 帧但当前无参考帧可用，无法解码（请先收到 I/P 帧）")

            # === 你的解码片段（mv）等价实现 ===
            # eframe = loaded_data
            # decoded_frame = grace.decode_mv_to_frame(eframe, prev_p2p_np)
            decoded_np = self.grace.decode_mv_to_frame(eframe, self.last_decoded_np)

            # 更新参考
            self.last_decoded_np = decoded_np.copy()
            # 对于后续 I/P 作为参考也 OK
            self.last_ref_np = decoded_np.copy()
        else:
            raise RuntimeError("无法从保存的 bin 中读取 I/P 或 mv 记录")

        # ========= 落盘 PNG =========
        # 放缩为（IMAGE_HEIGHT, IMAGE_WIDTH）大小的PNG
        to_save_np = decoded_np.copy()
        out_png = os.path.join(PNG_OUT_DIR, f"frame_{frame_id:04d}.png")
        Image.fromarray((np.clip(to_save_np, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_png)
        return out_png


# ========= 保留/复用：YOLO 推理 =========
def _run_yolo(file_path: str) -> Dict[str, Any]:
    """
    YOLO 推理：
      - 若安装 ultralytics：真实推理并抽取 box/cls/conf
      - 否则：返回占位结果（不崩溃）
    """
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")   # 轻量模型；如需自定义请在此处替换文件名
        res = model(file_path)
        out = []
        for r in res:
            boxes = []
            if getattr(r, "boxes", None) is not None:
                for b in r.boxes:
                    xyxy = getattr(b, "xyxy", None)
                    conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
                    cls  = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else -1
                    if xyxy is not None:
                        x1, y1, x2, y2 = [float(x) for x in xyxy[0].tolist()]
                        boxes.append({"xyxy": [x1, y1, x2, y2], "cls": cls, "conf": conf})
            out.append({"boxes": boxes})
        return {"ok": True, "engine": "ultralytics", "frames": out}
    except Exception as e:
        # 占位推理：回传文件大小和异常说明，保证流程不断
        try:
            size = os.path.getsize(file_path)
        except Exception:
            size = -1
        return {"ok": True, "engine": "stub", "size": size, "note": str(e)}

# ========= 修改：保存“接收到的二进制帧”为 .bin =========
def _save_bin(save_dir: str, frame_id: int, data: bytes) -> str:
    """
    新增功能块（替代原 _save_frame）：
    将收到的数据保存为 SAVE_DIR/frame_{frame_id}.bin
    返回：保存后的完整路径

    参数:
      - save_dir: 保存根目录
      - frame_id: 帧编号
      - data:     收到的二进制内容
    """
    fname = f"frame_{frame_id:04d}.bin"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

def main():
    """
    主流程（修改后的核心逻辑）：
      - 持续接收来自 client 的二进制帧（pkt_type==PKT_DATA）
      - 保存为 SAVE_DIR/frame_{frame_id}.bin
      - 解析 .bin 并按 I/P 或 mv 解码为 PNG（PNG_OUT_DIR/frame_{frame_id}.png）
      - 运行 YOLO 检测
      - 将检测结果（JSON 字节）以 pkt_type==PKT_RES 且复用相同 frame_id 的形式回传 client
    """
    net = NetGear(
        address=BIND_ADDR,
        port=PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1400,
        recv_buffer_size=32 * 1024 * 1024,
        send_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

    # 向 client 发送响应的网络连接（保留）
    client_net = NetGear(
        address=CLIENT_ADDR,
        port=CLIENT_PORT,
        protocol="udp",
        receive_mode=False,
        logging=True,
        mtu=1400,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

    # 新增：Grace 解码上下文（持久化参考）
    decoder = GraceDecoderContext(GRACE_ROOT, GRACE_MODEL_ID)

    start = time.time()
    received = 0
    sent_res = 0

    try:
        while True:
            item = net.recv()  # None 或 {'pkt_type','frame_id','data'}
            if item is not None:
                try:
                    pkt_type = int(item.get("pkt_type"))
                    frame_id = int(item.get("frame_id"))
                    data: bytes = item.get("data", b"")
                except Exception:
                    # 兼容极端情形（旧实现返回裸 bytes），此处无法提取 frame_id，直接忽略
                    continue

                if pkt_type == PKT_DATA:
                    # 1) 保存收到的 .bin
                    fpath = _save_bin(SAVE_DIR, frame_id, data)
                    received += 1
                    print(f"[Server] Received frame {received}: frame_id={frame_id}, size={len(data)}, saved to {os.path.basename(fpath)}")
                    if received % 100 == 0:
                        print(f"[Server] Total Received={received}")

                    # 2) 解码 bin -> png（复用 runner 中的 iter_read_grace_records / GraceBundle）
                    try:
                        png_path = decoder.decode_bin_to_png(fpath, frame_id)
                    except Exception as dec_err:
                        # 解码失败则把错误回传给 client，避免“黑盒”沉默
                        err_payload = json.dumps(
                            {"ok": False, "error": f"decode_failed: {dec_err}", "frame_id": frame_id},
                            ensure_ascii=False
                        ).encode("utf-8")
                        client_net.send(err_payload, pkt_type=PKT_RES, frame_id=frame_id)
                        print(f"[Server] Decode failed for frame_id={frame_id}: {dec_err}")
                        continue

                    # 3) YOLO 检测
                    yolo_res = _run_yolo(png_path)
                    payload = json.dumps(yolo_res, ensure_ascii=False).encode("utf-8")

                    # 4) 回传检测结果（pkt_type=PKT_RES，frame_id 原样复用）
                    client_net.send(payload, pkt_type=PKT_RES, frame_id=frame_id)
                    sent_res += 1
                    print(f"[Server] Sent YOLO result {sent_res}: frame_id={frame_id}, size={len(payload)}")
                    if sent_res % 100 == 0:
                        print(f"[Server] Total Sent YOLO results={sent_res}")

                elif pkt_type == PKT_RES:
                    # 理论上 server 不该收到 RES；若发生则记录
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = f"<{len(data)} bytes>"
                    print(f"[Server] (unexpected) YOLO result for frame_id={frame_id}: {text}")

            # 运行时长控制
            if RUN_SECONDS > 0 and (time.time() - start) >= RUN_SECONDS:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        print(f"[Server] Done. received={received}, sent_res={sent_res}, elapsed={elapsed:.2f}s")
        if hasattr(net, "get_stats"):
            print(f"[Server] Receiver stats: {net.get_stats()}")
        if hasattr(client_net, "get_stats"):
            print(f"[Server] Sender stats: {client_net.get_stats()}")
        net.close()
        client_net.close()

if __name__ == "__main__":
    main()
