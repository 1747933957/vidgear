#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import Dict, Any

# 保持你的原始导入与常量
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 保持原有“写死”的常量，不做任何修改 ======
SAVE_DIR = "/home/wxk/workspace/nsdi/Viduce/net/vidgear/temp"
os.makedirs(SAVE_DIR, exist_ok=True)
RUN_SECONDS = 300  # 至少运行5分钟
PORT = 5557        # 使用不同于client的端口
BIND_ADDR = "0.0.0.0"  # 绑定所有接口，便于本地测试

# ====== 新增：client配置 ======  
CLIENT_ADDR = "127.0.0.1"              # client地址
CLIENT_PORT = 5556                     # client端口

# ====== 新增：pkt_type 常量 ======
try:
    from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM = 0, 1, 2, 3

def _save_frame(save_dir: str, frame_id: int, data: bytes) -> str:
    """
    新增功能块：将收到的数据保存为 SAVE_DIR/frame_{frame_id}.bin
    返回：保存后的完整路径
    """
    fname = f"frame_{frame_id}.png"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

def _run_yolo(file_path: str) -> Dict[str, Any]:
    """
    YOLO 推理：
      - 若安装 ultralytics：真实推理并抽取 box/cls/conf
      - 否则：返回占位结果（不崩溃）
    """
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")   # 轻量模型；如需自定义请在此处替换文件名
        # 注意：此处假设 file_path 指向可被 YOLO 解析的图像文件
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

def main():
    """
    主流程（修改后的核心逻辑）：
      - 持续接收来自 client 的文件（pkt_type==PKT_DATA）
      - 保存为 SAVE_DIR/frame_{frame_id}.bin
      - 运行 YOLO 检测
      - 将检测结果（JSON 字节）以 pkt_type==PKT_RES 且复用相同 frame_id 的形式回传 client
    """
    net = NetGear(
        address=BIND_ADDR,       # 按你的写死地址绑定监听
        port=PORT,
        protocol="udp",
        receive_mode=True,
        logging=True,
        mtu=1400,
        recv_buffer_size=32 * 1024 * 1024,
        send_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

    # 新增：向client发送响应的网络连接
    client_net = NetGear(
        address=CLIENT_ADDR,
        port=CLIENT_PORT,
        protocol="udp",
        receive_mode=False,       # 发送模式
        logging=True,
        mtu=1400,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )

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
                    # 1) 保存收到的文件
                    fpath = _save_frame(SAVE_DIR, frame_id, data)
                    received += 1
                    print(f"[Server] Received frame {received}: frame_id={frame_id}, size={len(data)}, saved to {os.path.basename(fpath)}")
                    if received % 100 == 0:
                        print(f"[Server] Total Received={received}")

                    # 2) YOLO 检测
                    yolo_res = _run_yolo(fpath)
                    payload = json.dumps(yolo_res, ensure_ascii=False).encode("utf-8")

                    # 3) 回传检测结果（pkt_type=PKT_RES，frame_id 原样复用）
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
