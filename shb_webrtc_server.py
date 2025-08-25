#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from typing import Dict, Any, Optional

# 使用统一的NetGear接口
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 配置常量 ======
BIND_ADDR = "0.0.0.0"     # 绑定所有接口
BIND_PORT = 5556          # 监听端口
RUN_SECONDS = 300         # 运行时间 (5分钟)
SAVE_DIR = "/home/wxk/workspace/nsdi/Viduce/net/vidgear/webrtc_temp"

def ensure_save_dir():
    """确保保存目录存在"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR

def save_received_data(frame_id: int, data: bytes) -> str:
    """
    保存接收到的数据到文件
    返回保存的文件路径
    """
    filename = f"webrtc_frame_{frame_id}.bin"
    filepath = os.path.join(SAVE_DIR, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath
    except Exception as e:
        print(f"[Server] Error saving frame {frame_id}: {e}")
        return ""

def analyze_data(data: bytes) -> Dict[str, Any]:
    """
    分析接收到的数据
    返回简单的统计信息
    """
    if not data:
        return {"size": 0, "error": "No data"}
    
    try:
        # 简单的数据分析
        size = len(data)
        
        # 计算一些基本统计
        byte_sum = sum(data)
        byte_avg = byte_sum / size if size > 0 else 0
        
        # 计算数据的一些特征
        unique_bytes = len(set(data))
        
        return {
            "size": size,
            "byte_average": round(byte_avg, 2),
            "unique_bytes": unique_bytes,
            "entropy": round(unique_bytes / 256.0, 3),  # 简单的熵估计
            "checksum": byte_sum % 65536
        }
    except Exception as e:
        return {"size": len(data), "error": str(e)}

def main():
    """
    WebRTC服务器主函数：
    - 启动WebRTC接收端
    - 接收来自客户端的数据包
    - 分析并保存数据
    - 统计接收性能
    """
    print(f"[Server] Starting WebRTC server...")
    print(f"[Server] Listening on {BIND_ADDR}:{BIND_PORT}")
    
    # 确保保存目录存在
    save_dir = ensure_save_dir()
    print(f"[Server] Data will be saved to: {save_dir}")
    
    # 创建WebRTC接收端
    receiver = NetGear(
        protocol="webrtc",
        address=BIND_ADDR,
        port=BIND_PORT,
        receive_mode=True,  # 接收模式
        logging=True
    )

    start_time = time.time()
    frame_count = 0
    total_bytes = 0
    saved_files = 0
    
    try:
        print(f"[Server] Waiting for connections and data...")
        print(f"[Server] Will run for {RUN_SECONDS} seconds...")
        
        while True:
            # 检查运行时间
            elapsed = time.time() - start_time
            if RUN_SECONDS > 0 and elapsed >= RUN_SECONDS:
                break
            
            # 接收数据
            try:
                item = receiver.recv()
                if item is None:
                    time.sleep(0.001)  # 短暂休眠，避免CPU占用过高
                    continue
                
                # 提取数据
                if isinstance(item, dict) and "data" in item:
                    data = item["data"]
                    frame_id = item.get("frame_id", frame_count)
                elif isinstance(item, bytes):
                    data = item
                    frame_id = frame_count
                else:
                    print(f"[Server] Unexpected data format: {type(item)}")
                    continue
                
                frame_count += 1
                data_size = len(data)
                total_bytes += data_size
                
                # 分析数据
                analysis = analyze_data(data)
                
                # 保存数据（每10帧保存一次，避免磁盘I/O过多）
                if frame_count % 10 == 0:
                    filepath = save_received_data(frame_count, data)
                    if filepath:
                        saved_files += 1
                
                # 定期输出统计信息
                if frame_count % 100 == 0:
                    avg_size = total_bytes / frame_count / 1024  # KB
                    throughput = total_bytes * 8 / (1024*1024) / elapsed  # Mbps
                    print(f"[Server] Received {frame_count} frames, "
                          f"total {total_bytes / (1024*1024):.2f} MB, "
                          f"avg {avg_size:.1f} KB/frame, "
                          f"throughput {throughput:.2f} Mbps")
                
                # 详细日志（仅前几帧和每1000帧）
                if frame_count <= 5 or frame_count % 1000 == 0:
                    print(f"[Server] Frame {frame_count}: "
                          f"size={data_size} bytes, "
                          f"analysis={analysis}")
                    
            except Exception as e:
                print(f"[Server] Error receiving frame: {e}")
                time.sleep(0.01)
                continue
            
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user")
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        elapsed = time.time() - start_time
        
        # 最终统计信息
        print(f"\n[Server] === Performance Statistics ===")
        print(f"[Server] Runtime: {elapsed:.2f} seconds")
        print(f"[Server] Frames received: {frame_count}")
        print(f"[Server] Total data: {total_bytes / (1024*1024):.2f} MB")
        print(f"[Server] Files saved: {saved_files}")
        
        if frame_count > 0 and elapsed > 0:
            print(f"[Server] Average FPS: {frame_count / elapsed:.2f}")
            print(f"[Server] Average throughput: {total_bytes * 8 / (1024*1024) / elapsed:.2f} Mbps")
            print(f"[Server] Average frame size: {total_bytes / frame_count / 1024:.1f} KB")
            
        # 获取接收端统计信息
        if hasattr(receiver, "get_stats"):
            stats = receiver.get_stats()
            print(f"[Server] Receiver stats: {stats}")
        
        receiver.close()
        print(f"[Server] Server stopped.")

if __name__ == "__main__":
    main()
