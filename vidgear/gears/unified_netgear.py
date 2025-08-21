#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一入口工厂：
- protocol='udp'     -> 使用本仓库的 NetGearUDP（底层 UDP + RTP/RTCP + 冗余/重传）
- protocol='webrtc'  -> 使用本仓库的 NetGearWebRTC（底层 WebRTC DataChannel，需 pip install aiortc）
- 其他/未指定        -> 回落到 vidgear.gears.netgear.NetGear（ZeroMQ 版本）

WebRTC 使用指引（简述）：
- 发送侧：
    tx = NetGearLike(protocol="webrtc", receive_mode=False, fec_parity=True, duplication_rate=0.2)
    offer = tx.create_webrtc_offer(); print(offer)
    # 将 offer 传给接收侧，得到 answer 后：
    tx.accept_webrtc_answer(answer_sdp)
- 接收侧：
    rx = NetGearLike(protocol="webrtc", receive_mode=True)
    rx.accept_webrtc_offer(offer_sdp)
    answer = rx.create_webrtc_answer(); print(answer)
    # 回传 answer 给发送侧
握手后即可 tx.send(b"...") / rx.recv()，应用层冗余/重传由本仓库实现，DataChannel 仅提供轻量底层重传。
"""

from typing import Any
import importlib

# 相对导入：同目录中的两种实现
from .netgear_udp import NetGearUDP
from .netgear_webrtc import NetGearWebRTC


class NetGearLike:
    """简单工厂：根据 protocol 选择实现。"""

    def __new__(cls, *args: Any, **kwargs: Any):
        protocol = kwargs.get("protocol", None)
        if isinstance(protocol, str):
            proto = protocol.lower()
            if proto == "udp":
                # UDP 走 NetGearUDP（与你现有逻辑保持一致）
                return NetGearUDP(*args, **kwargs)
            if proto == "webrtc":
                # WebRTC 走我们新增的 NetGearWebRTC
                return NetGearWebRTC(*args, **kwargs)

        # 其他协议：回落到原生 NetGear（ZeroMQ）
        vidgear_netgear = importlib.import_module("vidgear.gears.netgear")
        NetGear = getattr(vidgear_netgear, "NetGear")
        return NetGear(*args, **kwargs)
