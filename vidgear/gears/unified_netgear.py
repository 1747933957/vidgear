#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 统一入口：protocol='udp' 走 NetGearUDP，其它走原 NetGear

from typing import Any
import importlib

from .netgear_udp import NetGearUDP  # 相对导入同目录下的文件


class NetGearLike:
    def __new__(cls, *args: Any, **kwargs: Any):
        protocol = kwargs.get("protocol", None)
        if isinstance(protocol, str) and protocol.lower() == "udp":
            return NetGearUDP(*args, **kwargs)
        # 回落到原生 NetGear（ZeroMQ）
        vidgear_netgear = importlib.import_module("vidgear.gears.netgear")
        NetGear = getattr(vidgear_netgear, "NetGear")
        return NetGear(*args, **kwargs)
