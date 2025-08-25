#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import glob
from typing import List

# 统一入口：保持与其他文件一致的导入
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 常量配置 ======
IMAGE_DIR = "/data/wxk/workspace/mirage/dataset/video000/rgb"
CLIENT_ADDR = "127.0.0.1"    # 明确指向本地client
CLIENT_PORT = 5556           # client 绑定的端口
SEND_INTERVAL_MS = 29        # 每29ms发送一张图片
MAX_IMAGES = 20              # 最多发送20张图片

# ====== 导入 pkt_type 常量 ======
try:
    from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM = 0, 1, 2, 3

def get_image_files(directory: str, max_count: int = 20) -> List[str]:
    """
    获取目录中的前max_count张图片文件，按文件名排序
    """
    pattern = os.path.join(directory, "frame_*.png")
    files = glob.glob(pattern)
    files.sort()  # 按文件名排序
    return files[:max_count]

def load_image_data(file_path: str) -> bytes:
    """
    读取图片文件的二进制数据
    """
    with open(file_path, "rb") as f:
        return f.read()

def extract_frame_id(file_path: str) -> int:
    """
    从文件路径中提取frame_id
    例如: frame_0000.png -> 0
    """
    basename = os.path.basename(file_path)
    if basename.startswith("frame_") and basename.endswith(".png"):
        try:
            # 提取数字部分
            frame_num_str = basename[6:-4]  # 去掉 "frame_" 和 ".png"
            return int(frame_num_str)
        except ValueError:
            return 0
    return 0

def main():
    """
    主流程：
    1. 读取指定目录中的前20张图片
    2. 每29ms发送一张图片到client进程（使用PKT_SV_DATA类型）
    3. 发送完毕后保持连接，便于观察后续的通信
    """
    print(f"[Sender] Starting image sender...")
    print(f"[Sender] Image directory: {IMAGE_DIR}")
    print(f"[Sender] Target: {CLIENT_ADDR}:{CLIENT_PORT}")
    print(f"[Sender] Send interval: {SEND_INTERVAL_MS}ms")
    
    # 获取图片文件列表
    image_files = get_image_files(IMAGE_DIR, MAX_IMAGES)
    if not image_files:
        print(f"[Sender] Error: No image files found in {IMAGE_DIR}")
        return
    
    print(f"[Sender] Found {len(image_files)} images to send")
    for i, f in enumerate(image_files):
        size = os.path.getsize(f)
        print(f"[Sender]   {i:2d}: {os.path.basename(f)} ({size:,} bytes)")
    
    # 创建NetGear发送端
    # 注意：这里我们不使用receive_mode，直接向指定地址发送
    net = NetGear(
        address=CLIENT_ADDR,
        port=CLIENT_PORT,
        protocol="udp",
        receive_mode=False,  # 发送模式
        logging=True,
        mtu=1400,
        send_buffer_size=32 * 1024 * 1024,
        recv_buffer_size=32 * 1024 * 1024,
        queue_maxlen=65536,
    )
    
    sent_count = 0
    start_time = time.time()
    
    try:
        print(f"[Sender] Starting to send images...")
        
        for image_file in image_files:
            # 读取图片数据
            try:
                image_data = load_image_data(image_file)
                frame_id = extract_frame_id(image_file)
                
                # 发送图片数据，使用PKT_SV_DATA类型
                net.send(image_data, pkt_type=PKT_SV_DATA, frame_id=frame_id)
                sent_count += 1
                
                print(f"[Sender] Sent {sent_count:2d}/{len(image_files)}: "
                      f"{os.path.basename(image_file)} (frame_id={frame_id}, {len(image_data):,} bytes)")
                
                # 等待指定的时间间隔
                time.sleep(SEND_INTERVAL_MS / 1000.0)
                
            except Exception as e:
                print(f"[Sender] Error sending {image_file}: {e}")
                continue
        
        # 发送完毕，等待一段时间以观察接收端的处理情况
        print(f"[Sender] All images sent. Waiting for 10 seconds to observe responses...")
        time.sleep(10.0)
        
    except KeyboardInterrupt:
        print(f"[Sender] Interrupted by user")
    except Exception as e:
        print(f"[Sender] Error: {e}")
    finally:
        elapsed = time.time() - start_time
        print(f"[Sender] Done. Sent {sent_count}/{len(image_files)} images in {elapsed:.2f}s")
        
        # 显示网络统计信息（如果可用）
        if hasattr(net, "get_stats"):
            stats = net.get_stats()
            print(f"[Sender] Network stats: {stats}")
        
        net.close()

if __name__ == "__main__":
    main()
