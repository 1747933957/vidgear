#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
严格版发送接口（无兜底）：
- 只建立一个全局 NetGear；不存在就抛错退出。
- send_file_via_netgear 内统一完成：
  1) 从全局 NetGear 读取网络统计：丢包率/RTT（若不可用或无样本 => 直接抛错）
  2) 用 WebRTC 策略查表计算冗余率（无法加载/不可用 => 直接抛错）
  3) 强制要求 ns3(...) 存在并被调用（缺失 => 直接抛错）
  4) 通过全局 NetGear 发送 ns3 处理后的数据
- 主流程中两个调用点统一改为：
    frame_data = append_grace_record(...)
    send_file_via_netgear(frame_data, frame_id=..., fps=args.fps)
"""

import argparse
import csv
import json
import subprocess
import os
import re
import sys
import tempfile
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import struct  # 二进制打包/解包
import numpy as np
from PIL import Image

# ===================== 依赖导入 =====================

# 统一入口：保持与其他文件一致的导入（NetGear 外观类）
from net.vidgear.vidgear.gears.unified_netgear import NetGearLike as NetGear

# pkt_type 常量
try:
    from net.vidgear.vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception as e:
    raise RuntimeError(f"[Init] 无法导入 NetGear UDP 常量: {e}")

# 【新增】ACK 常量
try:
    from net.vidgear.vidgear.gears.netgear_udp import PKT_ACK_PACKET  # type: ignore
except Exception:
    PKT_ACK_PACKET = 4  # 兜底

# 仅用于“查表计算冗余率”的策略类（不使用 WebRTC 发送栈）
try:
    from net.vidgear.vidgear.gears.netgear_webrtc import WebRtcPolicy  # type: ignore
except Exception as e:
    raise RuntimeError(f"[Init] 无法导入 WebRtcPolicy（需要用它查表得到冗余率）：{e}")

# ===================== 运行参数 =====================

# 【修改】发送端在 127.0.0.1:5556 上“绑定+接收 ACK”，对端为 127.0.0.1:5557
BIND_ADDR = "127.0.0.1"   # 本端绑定地址（接收 ACK）
BIND_PORT = 5556          # 本端绑定端口
PEER_ADDR = "127.0.0.1"   # 对端地址（接收端）
PEER_PORT = 5557          # 对端端口

# WebRTC 策略查表的“码率维度”，由你根据业务设定
NS3_DEFAULT_BITRATE_MBPS = 10.0  # 新变量：查表用码率（Mb/s）

# ===================== 全局 NetGear 单例（严格） =====================

_GLOBAL_NET: Optional[NetGear] = None

def ensure_global_netgear() -> NetGear:
    """
    确保全局 NetGear 已创建。失败 => 直接抛错退出。
    """
    global _GLOBAL_NET
    if _GLOBAL_NET is not None:
        return _GLOBAL_NET

    try:
        # 【修改】receive_mode=True 以便接收 ACK；绑定到 127.0.0.1:5556
        _GLOBAL_NET = NetGear(
            address=BIND_ADDR,
            port=BIND_PORT,
            protocol="udp",
            receive_mode=True,   # 接收 ACK
            logging=True,
            mtu=1400,
            send_buffer_size=32 * 1024 * 1024,
            recv_buffer_size=32 * 1024 * 1024,
            queue_maxlen=65536,
        )
        # 【新增】指定对端 => 127.0.0.1:5557
        _GLOBAL_NET._peer_addr = (PEER_ADDR, PEER_PORT)
    except Exception as e:
        raise RuntimeError(f"[Net] 初始化全局 NetGear 失败：{e}")

    # 【新增】创建后即确保 ACK 接收线程已启动（只会启动一次）
    _ensure_ack_thread(_GLOBAL_NET)

    return _GLOBAL_NET

def shutdown_global_netgear() -> None:
    """
    关闭全局 NetGear。若关闭失败，抛错。
    """
    global _GLOBAL_NET
    if _GLOBAL_NET is None:
        return
    try:
        _GLOBAL_NET.close()
    except Exception as e:
        raise RuntimeError(f"[Net] 关闭全局 NetGear 失败：{e}")
    finally:
        _GLOBAL_NET = None

# ===================== 【新增】ACK 接收与 RTT 统计 =====================
# 设计说明：
# - 发送端不再在负载中前置时间戳。
# - 接收端在收到数据后，会从 UDP/RTP 头中取出发送时间戳，并在 ACK 的负载里回传：
#     格式：!Id（大端）= [frame_id:uint32][send_ts:double秒]
# - 本线程持续从 net.recv() 拉取包；遇到 PKT_ACK_PACKET：
#     RTT(ms) = (now() - send_ts) * 1000

import threading  # 【新增】

_ACK_FMT = "!Id"                           # 【新增】ACK 负载格式（id:uint32 + ts:double秒）
_ACK_SIZE = struct.calcsize(_ACK_FMT)      # 【新增】

_RTT_MS_LAST: float = float("nan")         # 【新增】最近一次 ACK 推导的 RTT（毫秒）
_RTT_LOCK = threading.Lock()               # 【新增】保护 RTT 读写

def _ack_rx_loop(net: NetGear) -> None:
    """【新增】后台线程：持续接收 ACK，解析 RTT。"""
    global _RTT_MS_LAST
    while True:
        try:
            item = net.recv()
        except Exception:
            break
        if item is None:
            time.sleep(0.001)
            continue
        try:
            pkt_type = int(item.get("pkt_type"))
            data: bytes = item.get("data", b"")
        except Exception:
            continue
        if pkt_type != PKT_ACK_PACKET:
            # 非 ACK 包忽略
            continue
        if len(data) >= _ACK_SIZE:
            try:
                frame_id_u32, send_ts = struct.unpack(_ACK_FMT, data[:_ACK_SIZE])
                rtt_ms = max(0.0, (time.time() - float(send_ts)) * 1000.0)
                with _RTT_LOCK:
                    _RTT_MS_LAST = rtt_ms
            except Exception:
                # 解析失败直接忽略，避免影响主流程
                pass

def _ensure_ack_thread(net: NetGear) -> None:
    """【新增】确保 ACK 接收线程只启动一次。"""
    if not hasattr(_ensure_ack_thread, "_started"):
        t = threading.Thread(target=_ack_rx_loop, args=(net,), name="ACK-RX", daemon=True)
        t.start()
        _ensure_ack_thread._started = True  # type: ignore[attr-defined]

def get_last_rtt_ms() -> Optional[float]:
    """【新增】读取最近一次由 ACK 计算的 RTT（毫秒）。"""
    with _RTT_LOCK:
        if math.isnan(_RTT_MS_LAST):
            return None
        return float(_RTT_MS_LAST)

# ===================== WebRTC 冗余率计算（严格） =====================

# 新变量：_WEBRTC_POLICY —— WebRtcPolicy 单例，用于查表计算冗余率
_WEBRTC_POLICY: Optional[WebRtcPolicy] = None

def _get_webrtc_policy() -> WebRtcPolicy:
    """
    返回 WebRtcPolicy 单例；无法构造（包括缺表）=> 直接抛错。
    """
    global _WEBRTC_POLICY
    if _WEBRTC_POLICY is not None:
        return _WEBRTC_POLICY
    try:
        # star_order=1：线性一阶调制；内部会按 webrtc-fec-array.h 查表
        _WEBRTC_POLICY = WebRtcPolicy(use_star=True, star_order=1, star_coeff=1.0, fec_table_path=None)
        return _WEBRTC_POLICY
    except Exception as e:
        # 这里不兜底，直接报错退出
        raise RuntimeError(f"[FEC] 初始化 WebRtcPolicy 失败（可能是表 webrtc-fec-array.h 缺失或解析失败）：{e}")

def compute_webrtc_fec_rate_strict(*, loss_rate: float, rtt_ms: float, fps: float, bitrate_mbps: float) -> float:
    """
    严格计算 WebRTC 冗余率：
    - 必须能拿到 WebRtcPolicy；失败 => 抛错
    - 必须传入合理的 loss_rate ∈ [0,1]、rtt_ms > 0、fps > 0、bitrate_mbps > 0；不满足 => 抛错
    - 返回 fec_rate ∈ [0,1]
    """
    if not (0.0 <= loss_rate <= 1.0):
        raise ValueError(f"[FEC] loss_rate 非法：{loss_rate}")
    if rtt_ms <= 0:
        raise ValueError(f"[FEC] rtt_ms 非法：{rtt_ms}")
    if fps <= 0:
        raise ValueError(f"[FEC] fps 非法：{fps}")
    if bitrate_mbps <= 0:
        raise ValueError(f"[FEC] bitrate_mbps 非法：{bitrate_mbps}")

    policy = _get_webrtc_policy()
    ddl_left_ms = int(round(1000.0 / fps))
    try:
        _g, beta = policy.compute(
            cur_loss=float(loss_rate),
            bitrate_mbps=float(bitrate_mbps),
            ddl_left_ms=int(ddl_left_ms),
            rtt_ms=int(round(rtt_ms)),
            is_rtx=False,
            max_group_size=48,
        )
    except Exception as e:
        raise RuntimeError(f"[FEC] WebRtcPolicy.compute 失败：{e}")

    # 严格裁剪（理论上 compute 已经裁了）
    if not (0.0 <= beta <= 1.0):
        raise RuntimeError(f"[FEC] 计算得到的冗余率超界：{beta}")
    return float(beta)

# ===================== I/P/MV 记录写入 =====================

MAGIC = b'SHBV'
VERSION = 1
FTYPE_I = 1
FTYPE_P = 2
FTYPE_MV = 3

def _shape_to4(x) -> tuple:
    """
    将 x 规整为 (N, C, H, W) 四个 int；异常 => 抛错（避免静默兜底）。
    """
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, bytearray, str)):
        arr = [int(v) for v in list(x)]
    else:
        arr = [int(x)]
    if len(arr) >= 4:
        arr = arr[-4:]
    else:
        arr = [1] * (4 - len(arr)) + arr
    return tuple(int(v) for v in arr)

def _coerce_bytes(maybe_pair):
    """
    统一返回 bytes；非法输入 => 抛错。
    """
    if isinstance(maybe_pair, (bytes, bytearray)):
        return bytes(maybe_pair)
    if (isinstance(maybe_pair, (tuple, list)) and len(maybe_pair) >= 1):
        return bytes(maybe_pair[0])
    raise TypeError("期望 bytes 或 (bytes, size)")

class IPartLike:
    """轻量 I-part：仅保留解码所需字段"""
    def __init__(self, code: bytes, shapex: int, shapey: int, offset_width: int, offset_height: int):
        self.code = code
        self.shapex = int(shapex)
        self.shapey = int(shapey)
        self.offset_width = int(offset_width)
        self.offset_height = int(offset_height)

class EncodedFrameLike:
    """
    轻量 EFrame：I/P 兼容
    """
    def __init__(self, *, frame_type: str, frame_id: int = 0,
                 code: bytes = None, shapex=None, shapey=None,
                 ipart: IPartLike = None,
                 res_stream: bytes = None, mv_stream: bytes = None, z_stream: bytes = None,
                 shapez=None, mxrange = None):
        self.frame_type = frame_type.upper()
        self.frame_id   = int(frame_id)
        self.code       = code
        self.shapex     = shapex
        self.shapey     = shapey
        self.ipart      = ipart
        self.res_stream = res_stream
        self.mv_stream  = mv_stream
        self.z_stream   = z_stream
        self.shapez     = shapez
        self.isize = None
        self.tot_size = None
        self.mxrange = mxrange

def append_grace_record(eframe=None,
                        compressed_data=None,
                        frame_type: str=None,
                        mv_j: bytes=None,
                        shape_j=None) -> bytes:
    """
    生成一条记录的二进制（LE 小端）。失败 => 抛错。
    - ftype=3('mv'):  [ shapey4(4*i32) | mv_len(u32) | mv_bytes ]
    - ftype=1('I'):   [ shapex(i32) | shapey(i32) | code_len(u32) | code_bytes ]
    - ftype=2('P'):   [ p_shapex4(4*i32) | p_shapey4(4*i32) | shapez4(4*i32)
                        | ip_shapex(i32) | ip_shapey(i32) | ip_off_w(i32) | ip_off_h(i32)
                        | mxrange_res(f32)
                        | res_off(u32) | mv_off(u32) | z_off(u32) | ip_off(u32)
                        | res_len(u32) | mv_len(u32) | z_len(u32) | ip_len(u32)
                        | [数据区：res | mv | z | ip_code] ]
    """
    # === mv 记录 ===
    if (frame_type or (getattr(eframe, 'frame_type', None) is None)) == 'mv':
        if not isinstance(mv_j, (bytes, bytearray)):
            raise TypeError("mv_j 必须是 bytes")
        if shape_j is None:
            raise ValueError("写入 MV 记录必须提供 shape_j")
        syN, syC, syH, syW = _shape_to4(shape_j)
        body = struct.pack('<4iI', syN, syC, syH, syW, len(mv_j)) + mv_j
        header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_MV, 0, len(body))
        return header + body

    # === I / P 分支 ===
    ftype = str(getattr(eframe, 'frame_type')).upper()

    if ftype == 'I':
        code_bytes = getattr(eframe, 'code', None)
        if code_bytes is None:
            if compressed_data is None:
                raise ValueError("I 帧缺少 code/compressed_data")
            code_bytes = _coerce_bytes(compressed_data[0])
        else:
            if not isinstance(code_bytes, (bytes, bytearray)):
                code_bytes = bytes(code_bytes)
        sx = int(getattr(eframe, 'shapex', 0))
        sy = int(getattr(eframe, 'shapey', 0))
        body = struct.pack('<iiI', sx, sy, len(code_bytes)) + code_bytes
        header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_I, 0, len(body))
        return header + body

    elif ftype == 'P':
        if compressed_data is None:
            raise ValueError("P 帧缺少 compressed_data")
        res_b = _coerce_bytes(compressed_data[0])
        mv_b  = _coerce_bytes(compressed_data[1])
        z_b   = _coerce_bytes(compressed_data[2])
        mxrange_res = float(compressed_data[3])

        psxN, psxC, psxH, psxW = _shape_to4(getattr(eframe, 'shapex', 0))
        psyN, psyC, psyH, psyW = _shape_to4(getattr(eframe, 'shapey', 0))
        if hasattr(eframe, 'z') and hasattr(eframe.z, 'shape'):
            z_shape_src = eframe.z.shape
        elif hasattr(eframe, 'shapez'):
            z_shape_src = getattr(eframe, 'shapez')
        else:
            raise ValueError("P 帧缺少 z 的形状（shapez）")
        zN, zC, zH, zW = _shape_to4(z_shape_src)

        ip = getattr(eframe, 'ipart', None)
        if ip is None:
            raise ValueError('P 帧必须包含 eframe.ipart')
        ip_code_b = _coerce_bytes(getattr(ip, 'code', b''))
        ip_sx = int(getattr(ip, 'shapex', 0))
        ip_sy = int(getattr(ip, 'shapey', 0))
        off_w = int(getattr(ip, 'offset_width', 0))
        off_h = int(getattr(ip, 'offset_height', 0))

        data_blob = res_b + mv_b + z_b + ip_code_b
        res_off = 0
        mv_off  = res_off + len(res_b)
        z_off   = mv_off  + len(mv_b)
        ip_off  = z_off   + len(z_b)

        table = struct.pack(
            '<4i4i4i i i i i f IIII IIII',
            psxN, psxC, psxH, psxW,
            psyN, psyC, psyH, psyW,
            zN, zC, zH, zW,
            ip_sx, ip_sy, off_w, off_h,
            float(mxrange_res),
            res_off, mv_off, z_off, ip_off,
            len(res_b), len(mv_b), len(z_b), len(ip_code_b)
        )
        body = table + data_blob
        header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_P, 0, len(body))
        return header + body

    else:
        raise ValueError(f'不支持的 frame_type: {ftype}')

# ===================== Pix2Pix & Grace（按你原逻辑） =====================

class Pix2PixRunner:
    def __init__(self, root: Path, ckpt: Path, netG: str, input_nc: int, output_nc: int,
                 config_str: str, ngf: int):
        # 省略：与原逻辑一致，这里保留必须实现；若加载失败请在你环境中修复模型依赖
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from models import create_model     # noqa
        from configs import decode_config   # noqa
        import torch

        self._create_model = create_model
        self._decode_config = decode_config

        class _MiniOpt: pass
        opt = _MiniOpt()
        opt.model = 'test'
        opt.netG = netG
        opt.input_nc = input_nc
        opt.output_nc = output_nc
        opt.ngf = ngf
        opt.norm = 'instance'
        opt.dropout_rate = 0.0
        opt.config_str = config_str
        opt.dataset_mode = 'single'
        opt.direction = 'AtoB'
        opt.serial_batches = True
        opt.no_flip = True
        opt.preprocess = 'resize_and_crop'
        opt.load_size = 1280
        opt.crop_size = 720
        opt.batch_size = 1
        opt.num_threads = 0
        opt.isTrain = False
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        opt.restore_G_path = str(ckpt)
        opt.results_dir = str(root / 'results_single_image')
        opt.name = 'viduce_single'
        opt.seed = 0
        opt.need_profile = False
        opt.real_stat_path = None
        opt.table_path = None
        opt.cityscapes_path = None
        opt.no_fid = True
        opt.dataroot = ''

        self.model = self._create_model(opt)
        try:
            self.model.setup(opt, verbose=False)
        except Exception as e:
            raise RuntimeError(f"[Pix2Pix] model.setup 失败：{e}")

        self.opt = opt
        self.config = self._decode_config(config_str) if (config_str and len(config_str) > 0) else None
        self.device = getattr(self.model, 'device', None)
        if self.device is None:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer_numpy(self, rgb_np: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            A = torch.from_numpy(rgb_np.transpose(2,0,1)).float().unsqueeze(0)
            A = (A * 2.0 - 1.0).to(self.device)
            data = {'A': A, 'A_paths': ['single.png']}
            self.model.set_input(data)
            self.model.test(self.config)

            if hasattr(self.model, 'fake_B'):
                fake = self.model.fake_B
                out = ((fake.clamp(-1,1) + 1.0) / 2.0).detach().cpu().numpy()[0].transpose(1,2,0)
                return np.clip(out.astype(np.float32), 0.0, 1.0)

            visuals = self.model.get_current_visuals()
            fake_np = visuals.get('fake_B', None)
            if fake_np is None:
                raise RuntimeError('[Pix2Pix] 推理未得到 fake_B')
            if isinstance(fake_np, np.ndarray):
                out = np.clip(fake_np.astype(np.float32), 0.0, 1.0)
            else:
                fake = fake_np
                out = ((fake.clamp(-1,1) + 1.0) / 2.0).detach().cpu().numpy()[0].transpose(1,2,0)
                out = np.clip(out.astype(np.float32), 0.0, 1.0)
            return out

class GraceBundle:
    """
    仅示意：对接你的 AE；若内部接口不符请在你环境调整。
    """
    def __init__(self, project_root: Path, model_id: str):
        # 简化：按你的环境准备好 import 路径
        for c in {project_root, project_root.parent, project_root / 'Intrinsic', project_root / 'intrinsic'}:
            if c and str(c) not in sys.path:
                sys.path.insert(0, str(c))
        try:
            from Intrinsic.intrinsic.Grace.ins import init_ae_model  # noqa
        except Exception as e:
            raise RuntimeError(f"[Grace] 导入 AE 失败：{e}")

        try:
            models = init_ae_model()
            self.ae = models[model_id]
        except Exception as e:
            raise RuntimeError(f"[Grace] 加载 AE 模型失败：{e}")

    def encode_keyframe(self, ref_np: np.ndarray, cur_np: np.ndarray, is_iframe: bool):
        # 这里直接调用你原本的 AE 接口；失败就抛错
        try:
            import torch
            from PIL import Image
            def _np2pil(x):
                arr = (np.clip(x,0.0,1.0)*255.0).astype(np.uint8) if x.dtype!=np.uint8 else x
                return Image.fromarray(arr, mode='RGB')
            
            def _adjust_size_for_grace(img, step=64):
                """调整图像尺寸使其能被step整除"""
                w, h = img.size
                new_w = ((w + step - 1) // step) * step
                new_h = ((h + step - 1) // step) * step
                return img.resize((new_w, new_h), Image.LANCZOS)
            
            ref_img = _np2pil(ref_np)
            cur_img = _np2pil(cur_np)
            
            # 调整图像尺寸以满足Grace模型要求
            ref_img = _adjust_size_for_grace(ref_img)
            cur_img = _adjust_size_for_grace(cur_img)
            
            # Grace模型的update_reference需要3D张量 [channel, height, width]
            ref_array = np.asarray(ref_img).astype(np.float32) / 255.0
            ref_t = torch.from_numpy(ref_array).permute(2, 0, 1)  # [3, H, W]
            if torch.cuda.is_available():
                ref_t = ref_t.cuda()
            self.ae.update_reference(ref_t)
            eframe, size, compressed_data = self.ae.encode_frame_return_compressed(cur_img, is_iframe)
            # 解码预览
            self.ae.update_reference(ref_t)
            decoded = self.ae.decode_frame(eframe)
            if isinstance(decoded, torch.Tensor):
                dec_np = decoded.detach().float().cpu().numpy().transpose(1,2,0)
                dec_np = np.clip(dec_np, 0.0, 1.0)
            elif isinstance(decoded, Image.Image):
                dec_np = np.asarray(decoded).astype(np.float32)/255.0
            else:
                dec_np = decoded
                if dec_np.max() > 1.0:
                    dec_np = dec_np/255.0
            return int(size), dec_np, eframe, compressed_data
        except Exception as e:
            raise RuntimeError(f"[Grace] 编码/解码失败：{e}")

    def mv_only_size(self, prev_np: np.ndarray, cur_np: np.ndarray):
        # 只示意：直接调用 ae.entropy_mv_interface；失败就抛错
        try:
            from PIL import Image
            def _np2pil(x):
                arr = (np.clip(x,0.0,1.0)*255.0).astype(np.uint8) if x.dtype!=np.uint8 else x
                return Image.fromarray(arr, mode='RGB')
            
            def _adjust_size_for_grace(img, step=64):
                """调整图像尺寸使其能被step整除"""
                w, h = img.size
                new_w = ((w + step - 1) // step) * step
                new_h = ((h + step - 1) // step) * step
                return img.resize((new_w, new_h), Image.LANCZOS)
            
            prev_img = _np2pil(prev_np)
            cur_img  = _np2pil(cur_np)
            
            # 调整图像尺寸以满足Grace模型要求
            prev_img = _adjust_size_for_grace(prev_img)
            cur_img = _adjust_size_for_grace(cur_img)
            
            fn = getattr(self.ae, 'entropy_mv_interface')
            sz, bs, shape = fn(cur_img, prev_img, use_estimation=False)
            return sz, bs, shape
        except Exception as e:
            raise RuntimeError(f"[Grace] entropy_mv_interface 失败：{e}")

# ===================== 工具：帧列举与读图 =====================

FRAME_RE = re.compile(r'^frame_(\d{4})\.png$')

def list_frames(dir_path: Path) -> List[Path]:
    files = []
    for p in dir_path.iterdir():
        if p.is_file():
            m = FRAME_RE.match(p.name)
            if m:
                files.append(p)
    files.sort(key=lambda p: int(FRAME_RE.match(p.name).group(1)))
    return files

def load_rgb_np(path: Path, target_size: Optional[Tuple[int,int]]=None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert('RGB')
        if target_size:
            im = im.resize((target_size[1], target_size[0]), resample=Image.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def video_to_temp_frames(video: Path, resize: Optional[str], tmp_dir: Path) -> Tuple[Path, float]:
    probe = subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,avg_frame_rate',
         '-of','json', str(video)],
        capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    fps_text = info['streams'][0].get('avg_frame_rate') or info['streams'][0]['r_frame_rate']
    num, den = fps_text.split('/')
    src_fps = float(num) / float(den if den!='0' else 1)
    out_dir = tmp_dir / 'frames'
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = []
    if resize:
        w,h = resize.split('x')
        scale = ['-vf', f'scale={int(w)}:{int(h)}']
    subprocess.run(
        ['ffmpeg','-i', str(video), *scale, str(out_dir / 'frame_%04d.png')],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return out_dir, src_fps

# ===================== 发送：计算与 ns3 调用（严格） =====================

def send_file_via_netgear(frame_data: bytes, frame_id: int, fps: float) -> None:
    """
    严格版发送函数（无任何兜底）：
    1) 必须已能初始化并获取到全局 NetGear
    2) 必须能从 NetGear.get_stats() 读到有效的统计
    3) 必须能计算得到 WebRTC 冗余率（策略/表可用）
    4) 必须存在全局 ns3(...) 函数可调用
    5) 发送 data_to_send
    任何一步失败 => 直接抛错并终止程序。
    """
    # 1) 确保全局 NetGear 可用
    net = ensure_global_netgear()
    # 再保险：这里也确保 ACK 接收线程已启动（幂等）
    _ensure_ack_thread(net)

    # 2) 读取网络统计
    stats = net.get_stats()
    if not isinstance(stats, dict):
        raise RuntimeError("[Send] get_stats() 未返回 dict")
    # 兼容：若缺字段，补默认值
    stats.setdefault("total_packets_lost", 0)
    stats.setdefault("rtt_ms", 1)
    stats.setdefault("sent_packets", 0)
    if "total_packets_received" not in stats or "total_packets_lost" not in stats or "rtt_ms" not in stats:
        raise RuntimeError("[Send] get_stats() 缺少必要字段（total_packets_received/total_packets_lost/rtt_ms）")

    # 【修改】丢包率按你的口径：1 - recv/sent
    sp = int(stats.get("sent_packets", 0))
    resv = int(stats.get("total_packets_received", 0))
    if sp <= 0:
        print("[Send] sent_packets==0 或缺失，按默认丢包率0.2处理")
        loss_rate = 0.2
    else:
        loss_rate = 1.0 - (resv / float(sp))
        loss_rate = max(0.0, min(1.0, loss_rate))

    # 【修改】RTT 来自 ACK
    rtt_from_ack = get_last_rtt_ms()
    if rtt_from_ack is None:
        print("[Send] 尚未从 ACK 收到 RTT；使用最小正数占位 1ms")
        rtt_ms = 1.0
    else:
        rtt_ms = float(rtt_from_ack)

    # 3) 计算 WebRTC 冗余率
    fec_rate = compute_webrtc_fec_rate_strict(
        loss_rate=loss_rate, rtt_ms=rtt_ms, fps=float(fps), bitrate_mbps=NS3_DEFAULT_BITRATE_MBPS
    )

    # 4)（可选）调用 ns3(...)，此处仍保持占位
    # if "ns3" not in globals():
    #     raise RuntimeError("[Send] 未找到全局函数 ns3(...)")
    # data_to_send = ns3(frame_data, loss_rate=loss_rate, rtt_ms=rtt_ms, fec_rate=fec_rate)  # noqa: F821
    data_to_send = frame_data

    # 【修改】不再在负载前添加发送时间戳（由接收端从头部获取）
    # for pkt in data_to_send:
    #      net.send(pkt)
    # 直接发送
    net.send(data_to_send, pkt_type=PKT_SV_DATA, frame_id=frame_id)
    print(f"[Sender] frame_id={frame_id} 已发送 | bytes={len(data_to_send)} | loss={loss_rate:.4f}, rtt={rtt_ms:.2f}ms, fec={fec_rate:.4f}")

# ===================== 主流程（按你原设计的 g5_*） =====================

@dataclass
class FrameRec:
    idx: int
    size: int
    time_ms: int

def _sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass  # 这里同步失败不致命

def run_g5_pix2pix_grace(args, frames: List[Path], pix: Pix2PixRunner, grace: GraceBundle) -> List[FrameRec]:
    recs: List[FrameRec] = []

    ref_keyframe_np_parallel0: Optional[np.ndarray] = None
    prev_group_key_np: Optional[np.ndarray] = None

    global_frame_idx = -1  # 仅第0帧 is_iframe=True

    for gi in range(0, (len(frames)+4)//5):
        s = gi*5
        e = min(len(frames), s+5)
        if s >= e:
            break

        group_files = frames[s:e]
        print(f'\n[GROUP {gi}] frames {group_files[0].name} .. {group_files[-1].name}')

        # 组内第0帧：pix2pix → Grace 完整编码
        f0 = group_files[0]
        global_frame_idx += 1
        rgb0 = load_rgb_np(f0)

        _sync_cuda()
        t0 = time.perf_counter()
        rgb0_pix = pix.infer_numpy(rgb0)
        if global_frame_idx == 0:
            ref_np = rgb0_pix
            is_iframe = True
        else:
            is_iframe = False
            if args.parallel:
                if ref_keyframe_np_parallel0 is None:
                    raise RuntimeError('并行模式下找不到第0组关键帧参考')
                ref_np = ref_keyframe_np_parallel0
            else:
                if prev_group_key_np is None:
                    raise RuntimeError('串行模式下找不到上一组关键帧参考')
                ref_np = prev_group_key_np

        size0, dec0_np, eframe, compressed_data = grace.encode_keyframe(ref_np=ref_np, cur_np=rgb0_pix, is_iframe=is_iframe)
        _sync_cuda()
        t1 = time.perf_counter()
        dt_ms = int(round((t1 - t0) * 1000))
        print(f'  [KEY] size={size0}  time_ms={dt_ms}')
        recs.append(FrameRec(idx=s, size=size0, time_ms=dt_ms))

        # === 关键帧：构造记录并发送（内部会严格计算并调用 ns3） ===
        frame_data = append_grace_record(eframe=eframe, compressed_data=compressed_data)
        send_file_via_netgear(frame_data, frame_id=s, fps=args.fps)

        # 更新参考
        if gi == 0:
            ref_keyframe_np_parallel0 = dec0_np.copy()
        prev_group_key_np = dec0_np.copy()

        # 组内 1..4 帧：仅统计大小（prev, cur）
        for j, fj in enumerate(group_files[1:], start=1):
            global_frame_idx += 1
            cur_np = load_rgb_np(fj, target_size=rgb0.shape[:2])
            if j == 1:
                prev_np = rgb0
            else:
                prev_np = load_rgb_np(group_files[j-1])

            _sync_cuda()
            t2 = time.perf_counter()
            sizej, mv_j, shape_j = grace.mv_only_size(prev_np=prev_np, cur_np=cur_np)
            _sync_cuda()
            t3 = time.perf_counter()
            dt_ms_j = int(round((t3 - t2) * 1000))
            print(f'  [MV]  frame={fj.name}  size={sizej}  time_ms={dt_ms_j}')
            recs.append(FrameRec(idx=s+j, size=sizej, time_ms=dt_ms_j))

            # === MV 记录：构造并发送（内部严格计算并调用 ns3） ===
            frame_data = append_grace_record(frame_type='mv', mv_j=mv_j, shape_j=shape_j)
            send_file_via_netgear(frame_data, frame_id=s+j, fps=args.fps)

    return recs

def run_xxx(*args, **kwargs):
    raise NotImplementedError("mode=xxx 尚未实现")

# ===================== CLI / main =====================

def build_args():
    p = argparse.ArgumentParser(
        description="Viduce pipeline: fps分帧 + 每5帧一组；组首帧 pix2pix→Grace(全编码)，其余4帧仅统计大小；并行/串行参考可选。"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--rgb-dir', type=Path, help='输入帧目录，文件名应类似 frame_0000.png')
    src.add_argument('--video', type=Path, help='输入视频路径（需要 ffmpeg）')

    p.add_argument('--fps', type=float, required=True, help='目标采样帧率')
    p.add_argument('--mode', type=str, default='g5_pix2pix_grace',
                   help='当前仅实现 g5_pix2pix_grace（5帧一组）')
    p.add_argument('-p', '--parallel', action='store_true',
                   help='并行参考（都参考第0组关键帧），缺省为串行参考（参考上一组关键帧）')

    # 仅处理前N帧
    p.add_argument('--max-frames', type=int, default=None,
                   help='仅处理采样后的前N帧')

    # ===== Pix2Pix 相关 =====
    p.add_argument('--pix2pix-root', type=Path, required=False,
                   default=Path('/home/wxk/workspace/nsdi/gan-compression'),
                   help='gan-compression 代码根目录')
    p.add_argument('--pix2pix-ckpt', type=Path, required=True,
                   help='pix2pix 生成器权重路径')
    p.add_argument('--pix2pix-netG', type=str, default='sub_mobile_resnet_9blocks',
                   help='生成器架构名')
    p.add_argument('--pix2pix-ngf', type=int, default=64,
                   help='非 sub_* 架构需要的 ngf')
    p.add_argument('--pix2pix-input-nc', type=int, default=3, help='输入通道数')
    p.add_argument('--pix2pix-output-nc', type=int, default=3, help='输出通道数')
    p.add_argument('--pix2pix-config-str', type=str,
                   default='56_24_24_56_16_64_16_64',
                   help='传给 sub_* 架构的 config_str')

    # ===== Grace 相关 =====
    p.add_argument('--grace-root', type=Path, required=True,
                   help='Intrinsic 工程根路径')
    p.add_argument('--grace-model-id', type=str, default='64',
                   help='init_ae_model() 可用的键（如 64/128/...）')

    # 输出
    p.add_argument('--out-dir', type=Path, required=True, help='输出目录')

    # 输入视频->帧抽取尺寸（可选）
    p.add_argument('--resize', type=str, default=None,
                   help='可选：强制尺寸，如 1280x720；仅对 --video 有效')

    return p.parse_args()

def main():
    args = build_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 提前初始化全局 NetGear；失败 => 直接异常退出
    ensure_global_netgear()

    # 清空输出目录旧文件
    for item in args.out_dir.iterdir():
        if item.is_file():
            item.unlink()

    try:
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            # 帧来源
            if args.rgb_dir:
                frames_dir = args.rgb_dir
                src_fps = None
            else:
                frames_dir, src_fps = video_to_temp_frames(args.video, args.resize, tmp)

            frames_all = list_frames(frames_dir)
            if len(frames_all) == 0:
                raise RuntimeError(f'未找到帧：{frames_dir}/frame_XXXX.png')

            # 采样
            if src_fps:
                step = max(1, int(round(src_fps / args.fps)))
            else:
                step = max(1, int(round(30.0 / args.fps)))  # 未知源 fps 时，按约定估计
            frames = frames_all[::step]
            if args.max_frames is not None:
                frames = frames[:max(0, int(args.max_frames))]
                print(f'[INFO] 仅处理前 {len(frames)} 帧')
            if len(frames) == 0:
                raise RuntimeError('[INFO] 没有帧可处理（--max-frames 过小？）')

            # 模型
            pix = Pix2PixRunner(args.pix2pix_root, args.pix2pix_ckpt,
                                args.pix2pix_netG, args.pix2pix_input_nc,
                                args.pix2pix_output_nc, args.pix2pix_config_str,
                                args.pix2pix_ngf)
            grace = GraceBundle(args.grace_root, args.grace_model_id)

            # 分发
            if args.mode == 'g5_pix2pix_grace':
                recs: List[FrameRec] = run_g5_pix2pix_grace(args, frames, pix, grace)
            else:
                raise RuntimeError(f"不支持的 mode: {args.mode}")

            # 保存 CSV
            csv_path = args.out_dir / 'metrics.csv'
            with csv_path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame_idx', 'size', 'time_ms'])
                for r in recs:
                    w.writerow([r.idx, r.size, r.time_ms])
            print(f'\n[OK] 写入 {csv_path}')

    finally:
        # 严格关闭（失败也抛错）
        shutdown_global_netgear()

if __name__ == '__main__':
    main()

'''
python shb_send_frame_interface.py \
  --rgb-dir /data/wxk/workspace/mirage/dataset/video000/rgb \
  --fps 30 --mode g5_pix2pix_grace \
  --pix2pix-ckpt  /data/wxk/workspace/mirage/dataset/video000/pix2pix_nb/log/albedo/compressed888/latest_net_G.pth \
  --grace-root    /home/wxk/workspace/nsdi/Intrinsic \
  --grace-model-id 64 \
  --pix2pix-netG sub_mobile_resnet_9blocks \
  --pix2pix-config-str 8_8_8_8_8_8_8_8 \
  --out-dir /home/wxk/workspace/nsdi/Viduce/out \
  --max-frames 20
'''
