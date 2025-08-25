#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from typing import Dict, Any

# 使用统一的NetGear接口
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 配置常量 ======
SERVER_ADDR = "127.0.0.1"  # 服务器地址
SERVER_PORT = 5557         # 服务器端口
FPS = 30                   # 帧率
BITRATE_MBPS = 10.0       # 比特率 (Mbps)
DDL_LEFT_MS = 100         # 延迟限制 (ms)
STAR_ORDER = 1            # STAR编码参数
STAR_COEFF = 1.0          # STAR编码系数
RUN_SECONDS = 300         # 运行时间 (5分钟)

# 数据包大小范围 (bytes)
MIN_PACKET_SIZE = 1024      # 1KB
MAX_PACKET_SIZE = 1024 * 1024  # 1MB

def generate_random_data(size: int) -> bytes:
    """生成指定大小的随机数据"""
    return bytes([random.randint(0, 255) for _ in range(size)])

def main():
    """
    WebRTC客户端主函数：
    - 连接到WebRTC服务器
    - 定期发送随机大小的数据包
    - 统计发送性能
    """
    print(f"[Client] Starting WebRTC client...")
    print(f"[Client] Target: {SERVER_ADDR}:{SERVER_PORT}")
    print(f"[Client] FPS: {FPS}, Bitrate: {BITRATE_MBPS} Mbps")
    
    # 创建WebRTC发送端
    sender = NetGear(
        protocol="webrtc",
        address=SERVER_ADDR,
        port=SERVER_PORT,
        receive_mode=False,  # 发送模式
        fps=FPS,
        bitrate_mbps=BITRATE_MBPS,
        ddl_left_ms=DDL_LEFT_MS,
        star_order=STAR_ORDER,
        star_coeff=STAR_COEFF,
        logging=True
    )

    start_time = time.time()
    frame_count = 0
    total_bytes = 0
    
    try:
        print(f"[Client] Starting to send data for {RUN_SECONDS} seconds...")
        
        while True:
            # 检查运行时间
            elapsed = time.time() - start_time
            if RUN_SECONDS > 0 and elapsed >= RUN_SECONDS:
                break
            
            # 生成随机大小的数据包
            packet_size = random.randint(MIN_PACKET_SIZE, MAX_PACKET_SIZE)
            data = generate_random_data(packet_size)
            
            # 发送数据
            try:
                sender.send(data)
                frame_count += 1
                total_bytes += len(data)
                
                if frame_count % 100 == 0:
                    print(f"[Client] Sent {frame_count} frames, "
                          f"total {total_bytes / (1024*1024):.2f} MB, "
                          f"avg {total_bytes / frame_count / 1024:.1f} KB/frame")
                    
            except Exception as e:
                print(f"[Client] Error sending frame {frame_count}: {e}")
                continue
            
            # 控制发送频率
            time.sleep(1.0 / FPS)
            
    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user")
    except Exception as e:
        print(f"[Client] Error: {e}")
    finally:
        elapsed = time.time() - start_time
        
        # 统计信息
        print(f"\n[Client] === Performance Statistics ===")
        print(f"[Client] Runtime: {elapsed:.2f} seconds")
        print(f"[Client] Frames sent: {frame_count}")
        print(f"[Client] Total data: {total_bytes / (1024*1024):.2f} MB")
        print(f"[Client] Average FPS: {frame_count / elapsed:.2f}")
        print(f"[Client] Average throughput: {total_bytes * 8 / (1024*1024) / elapsed:.2f} Mbps")
        print(f"[Client] Average frame size: {total_bytes / frame_count / 1024:.1f} KB" if frame_count > 0 else "")
        
        # 获取发送端统计信息
        if hasattr(sender, "get_stats"):
            stats = sender.get_stats()
            print(f"[Client] Sender stats: {stats}")
        
        sender.close()
        print(f"[Client] Connection closed.")

if __name__ == "__main__":
    main()
