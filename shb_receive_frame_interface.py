#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import time
from typing import Dict, Any

# 保持你的原始导入与常量（注意该文件的 import 路径与发送端不同）
from vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 保持原有常量结构，按你的端口要求修改 ======
SAVE_DIR = "/home/wxk/workspace/nsdi/Viduce/net/vidgear/temp"
os.makedirs(SAVE_DIR, exist_ok=True)
RUN_SECONDS = 300  # 至少运行5分钟

# 【修改】接收端绑定 127.0.0.1:5557
BIND_ADDR = "127.0.0.1"
PORT = 5557

# 【修改】ACK 回发至发送端 127.0.0.1:5556
CLIENT_ADDR = "127.0.0.1"
CLIENT_PORT = 5556

# ====== pkt_type 常量（包含 PKT_ACK_PACKET） ======
try:
    from vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_PACKET  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM, PKT_ACK_PACKET = 0, 1, 2, 3, 4

# 【新增】ACK 负载格式：!Id（大端）= [frame_id:uint32][send_ts:double秒]
_ACK_FMT = "!Id"
_ACK_SIZE = struct.calcsize(_ACK_FMT)

def _save_frame(save_dir: str, frame_id: int, payload: bytes) -> str:
    """
    将收到的数据保存到文件（不再剥离任何“前置时间戳”——因为发送端不再前置时间戳）。
    这里沿用 .png 后缀（如需 .bin 可自行调整）。
    """
    fname = f"frame_{frame_id}.png"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, "wb") as f:
        f.write(payload)
    return fpath

def _extract_send_ts_from_header(item: Dict[str, Any]) -> float:
    """
    【新增】从 UDP/RTP 头部相关字段提取“发送时刻时间戳”，统一换算为“秒”。
    支持以下键（按优先级）：
      - 'rtp_ts' : int，RTP 90kHz 时钟，换算 sec = rtp_ts / 90000.0
      - 'timestamp' / 'send_ts' : float 或可转为 float，单位秒
    若都不存在，则返回 time.time()（退化为 0 RTT，至少不致崩溃）。
    """
    if not isinstance(item, dict):
        return time.time()
    if "rtp_ts" in item:
        try:
            return int(item["rtp_ts"]) / 90000.0
        except Exception:
            pass
    for k in ("timestamp", "send_ts"):
        if k in item:
            try:
                return float(item[k])
            except Exception:
                pass
    return time.time()

def main():
    """
    主流程：
      - 绑定在 127.0.0.1:5557，持续接收来自发送端的数据帧（pkt_type==PKT_DATA/PKT_SV_DATA）
      - 不再从负载取时间戳；直接从 item 的 UDP/RTP 头字段提取发送时刻（秒）
      - 保存真实数据
      - 立即发送 ACK（pkt_type==PKT_ACK_PACKET），frame_id 复用，
        负载为：!Id = [frame_id:uint32][发送时刻:double秒]
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
        deliver_per_packet=True,   # 【新增】要求底层按“每个分片”上抛
    )

    # 显式设置对端地址，确保可以 send 回 ACK（到 127.0.0.1:5556）
    net._peer_addr = (CLIENT_ADDR, CLIENT_PORT)

    start = time.time()
    try:
        while True:
            item = net.recv()  # None 或 {'pkt_type','frame_id','data', ... 以及头部时间戳字段}
            if item is not None:
                # ====== 在处理 item 之前，先识别“包级事件” ======
                if isinstance(item, dict) and item.get("is_fragment"):
                    # 【新增】按包回 ACK

                    # 取出字段
                    frame_id = int(item.get("frame_id", 0))
                    frag_idx = int(item.get("frag_idx", 0))
                    data: bytes = item.get("data", b"")

                    # 【新增】构造“包 ID”：高 16 位帧 ID、低 16 位分片序号（按需调整位宽）
                    packet_id = ((frame_id & 0xFFFF) << 16) | (frag_idx & 0xFFFF)

                    # 【新增】由 RTP 时间戳换算发送时刻（秒）
                    send_ts_sec = float(item.get("arrival_ts", time.time()))

                    # 【新增】ACK 负载：!Id（packet_id:uint32, send_ts:double秒）
                    ack_payload = struct.pack("!Id", int(packet_id) & 0xFFFFFFFF, send_ts_sec)

                    # 【新增】立刻回 ACK（pkt_type=PKT_ACK_PACKET，frame_id 可回传原 frame_id）
                    net.send(ack_payload, pkt_type=PKT_ACK_PACKET, frame_id=frame_id)

                    # 【可选】这里可以直接 return/continue，避免该分片落到“完整帧处理”分支里
                    # 但通常让它继续也没问题（完整帧处理用的是另一条合帧记录）

                    # 解析data
                    # 更新哪些帧已经接收到
                    continue

            # 运行时长控制
            if RUN_SECONDS > 0 and (time.time() - start) >= RUN_SECONDS:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(net, "get_stats"):
            print(f"[Server] Receiver stats: {net.get_stats()}")
        net.close()

if __name__ == "__main__":
    main()
