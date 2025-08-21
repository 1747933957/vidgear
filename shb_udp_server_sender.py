# shb_udp_client.py, shb_udp_server_receiver.py, shb_udp_server_sender.py是一组，由shb_udp_server_sender.py发送到shb_udp_client.py再发送到shb_udp_server_receiver.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import collections
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
import struct  # 用于二进制打包/解包
import numpy as np
from PIL import Image
import torch

# 统一入口：保持与其他文件一致的导入
from net.vidgear.vidgear.gears.unified_netgear import NetGearLike as NetGear

# ====== 导入 pkt_type 常量 ======
try:
    from net.vidgear.vidgear.gears.netgear_udp import PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM  # type: ignore
except Exception:
    PKT_DATA, PKT_RES, PKT_SV_DATA, PKT_TERM = 0, 1, 2, 3

# ====== 常量配置 ======
CLIENT_ADDR = "172.27.143.41" # 本机测试："127.0.0.1"
CLIENT_PORT = 5556           # client 绑定的端口

# ------------------------------
# 1) CLI
# ------------------------------
def build_args():
    p = argparse.ArgumentParser(
        description="Viduce pipeline: fps分帧 + 每5帧一组；组首帧 pix2pix→Grace(全编码)，其余4帧仅统计大小；支持串行/并行参考；可限制处理的最大帧数"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--rgb-dir', type=Path, help='输入帧目录，文件名应类似 frame_0000.png')
    src.add_argument('--video', type=Path, help='输入视频路径（需要ffmpeg在PATH中）')

    p.add_argument('--fps', type=float, required=True, help='目标采样帧率（按 1/fps 取帧）')
    p.add_argument('--mode', type=str, default='g5_pix2pix_grace',
                   help='当前仅实现 g5_pix2pix_grace（5帧一组）')
    p.add_argument('-p', '--parallel', action='store_true',
                   help='并行参考（都参考第0组关键帧），缺省为串行参考（参考上一组关键帧）')

    # 仅处理前N帧
    p.add_argument('--max-frames', type=int, default=None,
                   help='仅处理采样后的前N帧；例如 20 表示只处理前20帧')

    # ===== Pix2Pix 相关 =====
    p.add_argument('--pix2pix-root', type=Path, required=False,
                   default=Path('/home/wxk/workspace/nsdi/gan-compression'),
                   help='gan-compression 代码根目录')
    p.add_argument('--pix2pix-ckpt', type=Path, required=True,
                   help='pix2pix 生成器权重路径，例如 checkpoints/latest_net_G.pth')
    p.add_argument('--pix2pix-netG', type=str, default='sub_mobile_resnet_9blocks',
                   help='生成器架构名，如 resnet_9blocks / mobile_resnet_9blocks / sub_mobile_resnet_9blocks 等')
    p.add_argument('--pix2pix-ngf', type=int, default=64,
                   help='对于非 sub_* 架构（resnet_9blocks / mobile_resnet_9blocks）需要的 ngf，默认64')
    p.add_argument('--pix2pix-input-nc', type=int, default=3, help='输入通道数（常见=3）')
    p.add_argument('--pix2pix-output-nc', type=int, default=3, help='输出通道数（常见=3）')
    p.add_argument('--pix2pix-config-str', type=str,
                   default='56_24_24_56_16_64_16_64',
                   help='传给 sub_* 架构的 config_str，默认 56_24_24_56_16_64')

    # ===== Grace 相关 =====
    p.add_argument('--grace-root', type=Path, required=False,
                   default=Path('/home/wxk/workspace/nsdi/Intrinsic'),
                   help='Intrinsic 工程根路径（我会向其相邻目录注入 sys.path）')
    p.add_argument('--grace-model-id', type=str, default='64',
                   help='init_ae_model() 可用的键（如 64/128/...）')

    # 输出
    p.add_argument('--out-dir', type=Path, required=True, help='输出目录（保存中间图与CSV）')

    # 输入视频->帧抽取尺寸（可选）
    p.add_argument('--resize', type=str, default=None,
                   help='可选，强制尺寸，如 1280x720；仅对 --video 有效')

    return p.parse_args()


# ------------------------------
# 2) 工具：列帧、读写、抽帧
# ------------------------------
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
    import subprocess, json
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


# ------------------------------
# 计时辅助：CUDA 同步（若可用）
# ------------------------------
def _sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ------------------------------
# 3) Pix2Pix Runner
# ------------------------------
class Pix2PixRunner:
    def __init__(self, root: Path, ckpt: Path, netG: str, input_nc: int, output_nc: int,
                 config_str: str, ngf: int):
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
            print('[Pix2Pix] model.setup 加载失败，smart_load 兜底：', repr(e))
            self._smart_load(self.model.netG, opt.restore_G_path)

        self.opt = opt
        self.config = self._decode_config(config_str) if (config_str and len(config_str) > 0) else None
        self.device = getattr(self.model, 'device', None)
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _smart_load(self, net, ckpt_path: str):
        import torch, collections
        raw = torch.load(ckpt_path, map_location='cpu')
        if isinstance(raw, dict):
            for k in ['state_dict','model','netG','G']:
                if k in raw and isinstance(raw[k], dict):
                    raw = raw[k]; break
        state = collections.OrderedDict()
        for k, v in raw.items():
            nk = k[7:] if k.startswith('module.') else k
            state[nk] = v
        model_sd = net.state_dict()
        loadable = {}
        mis_shape = []
        unexpected = []
        for k, v in state.items():
            if k in model_sd:
                if tuple(model_sd[k].shape) == tuple(v.shape):
                    loadable[k] = v
                else:
                    mis_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            else:
                unexpected.append(k)
        model_sd.update(loadable)
        net.load_state_dict(model_sd, strict=False)
        print(f'[Pix2Pix] smart_load: matched={len(loadable)} / {len(model_sd)} | '
              f'unexpected={len(unexpected)} | shape_mismatch={len(mis_shape)}')

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
                raise RuntimeError('pix2pix 推理未得到 fake_B')
            if isinstance(fake_np, np.ndarray):
                out = np.clip(fake_np.astype(np.float32), 0.0, 1.0)
            else:
                fake = fake_np
                out = ((fake.clamp(-1,1) + 1.0) / 2.0).detach().cpu().numpy()[0].transpose(1,2,0)
                out = np.clip(out.astype(np.float32), 0.0, 1.0)
            return out


# ------------------------------
# 4) Grace / AE 封装（仅使用 Intrinsic.intrinsic.Grace.ins；自动补齐到步长[若提供]）
# ------------------------------
class GraceBundle:
    """
    仅使用 Intrinsic.intrinsic.Grace.ins.init_ae_model() 创建 AE。
    若 AE 暴露 w_step/h_step，则在编码/估计前按其倍数做右/下 padding，并在解码后裁回原尺寸。
    """
    def __init__(self, project_root: Path, model_id: str):
        for c in {project_root, project_root.parent, project_root / 'Intrinsic', project_root / 'intrinsic'}:
            if c and str(c) not in sys.path:
                sys.path.insert(0, str(c))

        self.ae = None
        # 优先大写路径
        try:
            from Intrinsic.intrinsic.Grace.ins import init_ae_model  # noqa
            models = init_ae_model()
            self.ae = models[model_id]
            print('[GraceBundle] 使用 Intrinsic.intrinsic.Grace.ins.AEModel')
        except Exception as e1:
            print('[GraceBundle] 加载 Intrinsic.intrinsic.Grace.ins 失败，回退小写：', repr(e1))
            from Intrinsic.intrinsic.Grace.ins import init_ae_model  # type: ignore
            models = init_ae_model()
            self.ae = models[model_id]
            print('[GraceBundle] 使用 Intrinsic.intrinsic.Grace.ins.AEModel')

        # 读取步长（可能不存在）
        ws = getattr(self.ae, 'w_step', None)
        hs = getattr(self.ae, 'h_step', None)
        self.w_step = int(ws) if isinstance(ws, (int, np.integer)) and ws > 0 else None
        self.h_step = int(hs) if isinstance(hs, (int, np.integer)) and hs > 0 else None

    @staticmethod
    def _to_pil(rgb_np: np.ndarray) -> Image.Image:
        if rgb_np.dtype != np.uint8:
            arr = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = rgb_np
        return Image.fromarray(arr, mode='RGB')

    @staticmethod
    def _pil_to_chw_tensor(img: Image.Image):
        import torch
        arr = np.asarray(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr.transpose(2,0,1))
        if torch.cuda.is_available():
            ten = ten.cuda()
        return ten

    def _pad_to_multiple_pil(self, img: Image.Image) -> Tuple[Image.Image, Tuple[int,int]]:
        """若提供了 w_step/h_step，则右/下边缘复制补齐；否则不变。返回(新图, 原始(W,H))."""
        if self.w_step is None or self.h_step is None:
            w, h = img.size
            return img, (w, h)
        w, h = img.size
        W = int(math.ceil(w / self.w_step) * self.w_step)
        H = int(math.ceil(h / self.h_step) * self.h_step)
        if W == w and H == h:
            return img, (w, h)
        arr = np.asarray(img)
        pad_right = W - w
        pad_bottom = H - h
        arr_pad = np.pad(arr, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='edge')
        return Image.fromarray(arr_pad, mode='RGB'), (w, h)

    @staticmethod
    def _crop_np_to(arr: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
        th, tw = target_hw
        return arr[:th, :tw, :]

    def _call_encode(self, is_iframe: bool, ref_img: Image.Image, cur_img: Image.Image):
        # 可选 padding
        ref_pad, _ = self._pad_to_multiple_pil(ref_img)
        cur_pad, _ = self._pad_to_multiple_pil(cur_img)

        ref_t = self._pil_to_chw_tensor(ref_pad)

        try:
            self.ae.update_reference(ref_t)
        except Exception as e:
            raise RuntimeError(f'AE.update_reference 失败: {e}')

        try:
            eframe, size, compressed_data = self.ae.encode_frame_return_compressed(cur_pad, is_iframe)
        except Exception as e:
            raise RuntimeError(f'AE.encode_frame 失败: {e}')

        try:
            setattr(eframe, 'tot_size', int(size))
        except Exception:
            pass
        return int(size), eframe, compressed_data

    def _call_decode(self, eframe, ref_img: Image.Image) -> np.ndarray:
        # 解码亦需设置参考；同样先 padding
        ref_pad, _ = self._pad_to_multiple_pil(ref_img)
        ref_t = self._pil_to_chw_tensor(ref_pad)
        try:
            self.ae.update_reference(ref_t)
        except Exception as e:
            raise RuntimeError(f'AE.update_reference(用于解码) 失败: {e}')
        decoded = self.ae.decode_frame(eframe)

        # 转 numpy [H,W,3] in [0,1]
        try:
            import torch
            if isinstance(decoded, torch.Tensor):
                dec = decoded.detach().float().cpu().numpy().transpose(1, 2, 0)
                return np.clip(dec, 0.0, 1.0)
        except Exception:
            pass

        if isinstance(decoded, Image.Image):
            arr = np.asarray(decoded).astype(np.float32) / 255.0
            return arr
        elif isinstance(decoded, np.ndarray):
            arr = decoded.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)
        raise RuntimeError('decode_frame 返回了无法识别的类型')

    # ==== 外部 API ====
    def encode_keyframe(self, ref_np: np.ndarray, cur_np: np.ndarray, is_iframe: bool):
        ref_img = self._to_pil(ref_np)
        cur_img = self._to_pil(cur_np)
        size, eframe, compressed_data_original = self._call_encode(is_iframe, ref_img, cur_img)
        decoded_np = self._call_decode(eframe, ref_img)
        # 裁回当前帧原尺寸（即便曾 padding）
        decoded_np = self._crop_np_to(decoded_np, (cur_np.shape[0], cur_np.shape[1]))
        return int(size), decoded_np, eframe, compressed_data_original

    def mv_only_size(self, prev_np: np.ndarray, cur_np: np.ndarray) -> int:
        prev_img = self._to_pil(prev_np)
        cur_img = self._to_pil(cur_np)

        # 若存在步长约束，则先 padding
        prev_pad, _ = self._pad_to_multiple_pil(prev_img)
        cur_pad, _  = self._pad_to_multiple_pil(cur_img)

        # 1) 优先：接口可能直接挂在 AE 上
        fn = getattr(self.ae, 'entropy_mv_interface', None)
        try:
            sz, bs, shape = fn(cur_pad, prev_pad, use_estimation=False)
            return sz, bs, shape
        except Exception as e1:
            print(f'[GraceBundle] AE.entropy_mv_interface 失败，尝试 grace_coder: {repr(e1)}')


# =========================================================
# 新增功能：I/P/MV 三类记录的二进制写入与读取（可直接复用）
# 记录结构采用 [header | body]，big picture 如下：
# header: magic(4s='SHBV') | ver(u8=1) | ftype(u8:1=I,2=P,3=mv) | reserved(u16=0) | payload_len(u32)
# body:
#   - ftype=3('mv'):  [ mv_len(u32) | mv_bytes ]
#   - ftype=1('I')  : [ frame_id(i32) | shapex(i32) | shapey(i32) | code_len(u32) | code_bytes ]
#   - ftype=2('P')  : [ ip_shapex(i32) | ip_shapey(i32) | ip_off_w(i32) | ip_off_h(i32)
#                       | res_off(u32) | mv_off(u32) | z_off(u32) | ip_off(u32)
#                       | res_len(u32) | mv_len(u32) | z_len(u32) | ip_len(u32)
#                       | [顺序拼接的数据区：res | mv | z | ip_code] ]
#
# 备注：
# - offset 为相对于“数据区起点”的偏移（u32）。
# - 读取时可直接切片拼出四段。
# - eframe_like 与 IPartLike 只是“字段名兼容”的轻量对象，避免 import ins.py。
# =========================================================

MAGIC = b'SHBV'
VERSION = 1
FTYPE_I = 1
FTYPE_P = 2
FTYPE_MV = 3
# === 新增：形状与字节的兼容处理 ===
# === 形状与字节的兼容处理 ===
def _shape_to4(x) -> tuple:
    """
    将 x 规整为 (N, C, H, W) 四个 int：
    - x 为 torch.Size/tuple/list/ndarray：取最后4维（不足则前补1）
    - x 为单 int：视作 (1,1,1,int(x))
    - 其它异常：回退 (1,1,1,0)
    """
    try:
        if hasattr(x, '__iter__') and not isinstance(x, (bytes, bytearray, str)):
            arr = [int(v) for v in list(x)]
        else:
            arr = [int(x)]
    except Exception:
        arr = [0]
    if len(arr) >= 4:
        arr = arr[-4:]
    else:
        arr = [1] * (4 - len(arr)) + arr
    return tuple(int(v) for v in arr)

def _coerce_bytes(maybe_pair):
    """
    输入可能是：
      - bytes / bytearray
      - (bytes, size) 或 [bytes, size]
    统一返回 bytes。
    """
    if isinstance(maybe_pair, (bytes, bytearray)):
        return bytes(maybe_pair)
    if (isinstance(maybe_pair, (tuple, list)) and len(maybe_pair) >= 1):
        return bytes(maybe_pair[0])
    return bytes(maybe_pair)


class IPartLike:
    """
    轻量 I-part 对象：仅保留解码所需字段
    - shapex/shapey：根据你的新约定，均为 int
    """
    def __init__(self, code: bytes, shapex: int, shapey: int, offset_width: int, offset_height: int):
        self.code = code
        self.shapex = int(shapex)         # int
        self.shapey = int(shapey)         # int
        self.offset_width = int(offset_width)
        self.offset_height = int(offset_height)

class EncodedFrameLike:
    """
    轻量 EFrame：I/P 兼容
    - I 帧：shapex/shapey 为 int，code 为 bytes
    - P 帧：shapex/shapey 为 (N,C,H,W)，ipart 为 IPartLike（其中 shapex/shapey 为 int）
            res_stream/mv_stream/z_stream 为 bytes，shapez 为 (N,C,H,W)
    """
    def __init__(self, *, frame_type: str, frame_id: int = 0,
                 code: bytes = None, shapex=None, shapey=None,
                 ipart: IPartLike = None,
                 res_stream: bytes = None, mv_stream: bytes = None, z_stream: bytes = None,
                 shapez=None, mxrange = None):
        self.frame_type = frame_type.upper()  # 'I' / 'P'
        self.frame_id   = int(frame_id)
        self.code       = code                # I: bytes；P: None
        self.shapex     = shapex              # I: int； P: (N,C,H,W)
        self.shapey     = shapey              # I: int； P: (N,C,H,W)
        self.ipart      = ipart               # P: IPartLike
        self.res_stream = res_stream          # P: bytes
        self.mv_stream  = mv_stream           # P: bytes
        self.z_stream   = z_stream            # P: bytes
        self.shapez     = shapez              # P: (N,C,H,W)
        # 可选统计字段
        self.isize = None
        self.tot_size = None
        self.mxrange = mxrange


def append_grace_record(bin_path: Path, *,
                        eframe=None,
                        compressed_data=None,
                        frame_type: str=None,
                        mv_j: bytes=None,
                        shape_j=None):
    """
    追加一条记录（LE 小端）：

    统一头部: magic(4s='SHBV') | ver(u8=4) | ftype(u8:1=I,2=P,3=mv) | reserved(u16=0) | payload_len(u32)

    - ftype=3('mv'): [ shapey4(4*i32) | mv_len(u32) | mv_bytes ]
        * 这里 shapey4 由 shape_j 规整得到（与 P 帧的 shapey 一致）

    - ftype=1('I'):  [ shapex(i32) | shapey(i32) | code_len(u32) | code_bytes ]
        * 不再保存 frame_id
        * 码流优先取 eframe.code；无则回退 compressed_data[0]

    - ftype=2('P'):  [ p_shapex4(4*i32) | p_shapey4(4*i32) | shapez4(4*i32)
                       | ip_shapex(i32) | ip_shapey(i32) | ip_off_w(i32) | ip_off_h(i32)
                       | mxrange_res(f32)                      <-- 新增字段（v4）
                       | res_off(u32) | mv_off(u32) | z_off(u32) | ip_off(u32)
                       | res_len(u32) | mv_len(u32) | z_len(u32) | ip_len(u32)
                       | [顺序拼接的数据区：res | mv | z | ip_code] ]
    """

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open('ab') as fh:
        # === mv 记录 ===
        if (frame_type or (getattr(eframe, 'frame_type', None) is None)) == 'mv':
            if not isinstance(mv_j, (bytes, bytearray)):
                raise TypeError("mv_j 必须是 bytes（压缩后的 MV 流）")
            if shape_j is None:
                raise ValueError("写入 MV 记录时必须提供 shape_j（四元组）")
            syN, syC, syH, syW = _shape_to4(shape_j)
            body = struct.pack('<4iI', syN, syC, syH, syW, len(mv_j)) + mv_j
            header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_MV, 0, len(body))
            fh.write(header); fh.write(body)
            return

        # === I / P 分支 ===
        ftype = str(getattr(eframe, 'frame_type')).upper()

        if ftype == 'I':
            code_bytes = getattr(eframe, 'code', None)
            if code_bytes is None:
                code_bytes = _coerce_bytes(compressed_data[0])
            else:
                if not isinstance(code_bytes, (bytes, bytearray)):
                    code_bytes = bytes(code_bytes)
            sx = int(getattr(eframe, 'shapex', 0))
            sy = int(getattr(eframe, 'shapey', 0))
            body = struct.pack('<iiI', sx, sy, len(code_bytes)) + code_bytes
            header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_I, 0, len(body))
            fh.write(header); fh.write(body)

        elif ftype == 'P':
            # 三段码流 + 新增 mxrange_res(f32)
            res_b = _coerce_bytes(compressed_data[0])
            mv_b  = _coerce_bytes(compressed_data[1])
            z_b   = _coerce_bytes(compressed_data[2])
            mxrange_res = float(compressed_data[3])  # 新增：编码端传入的 mxrange（float）

            # 形状
            psxN, psxC, psxH, psxW = _shape_to4(getattr(eframe, 'shapex', 0))
            psyN, psyC, psyH, psyW = _shape_to4(getattr(eframe, 'shapey', 0))
            if hasattr(eframe, 'z') and hasattr(eframe.z, 'shape'):
                z_shape_src = eframe.z.shape
            elif hasattr(eframe, 'shapez'):
                z_shape_src = getattr(eframe, 'shapez')
            else:
                z_shape_src = 0
            zN, zC, zH, zW = _shape_to4(z_shape_src)

            ip = getattr(eframe, 'ipart', None)
            if ip is None:
                raise ValueError('P 帧必须包含 eframe.ipart')
            ip_code_b = _coerce_bytes(getattr(ip, 'code', b''))
            ip_sx = int(getattr(ip, 'shapex', 0))
            ip_sy = int(getattr(ip, 'shapey', 0))
            off_w = int(getattr(ip, 'offset_width', 0))
            off_h = int(getattr(ip, 'offset_height', 0))

            # 数据区与 offset/len
            data_blob = res_b + mv_b + z_b + ip_code_b
            res_off = 0
            mv_off  = res_off + len(res_b)
            z_off   = mv_off  + len(mv_b)
            ip_off  = z_off   + len(z_b)

            # 表头（v4：新增 f32 mxrange_res）
            table = struct.pack(
                '<4i4i4i i i i i f IIII IIII',
                psxN, psxC, psxH, psxW,
                psyN, psyC, psyH, psyW,
                zN, zC, zH, zW,
                ip_sx, ip_sy, off_w, off_h,
                float(mxrange_res),                 # <--- 新增
                res_off, mv_off, z_off, ip_off,
                len(res_b), len(mv_b), len(z_b), len(ip_code_b)
            )
            body = table + data_blob
            header = struct.pack('<4sBBHI', MAGIC, VERSION, FTYPE_P, 0, len(body))
            fh.write(header); fh.write(body)

        else:
            raise ValueError(f'不支持的 frame_type: {ftype}')

# ------------------------------
# 5) 网络发送相关函数
# ------------------------------
def send_file_via_netgear(file_path: Path, frame_id: int):
    """
    模仿 shb_sender.py 的操作，通过 NetGear 发送生成的文件
    """
    try:
        # 读取文件数据
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # 创建NetGear发送端
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
        
        # 发送文件数据，使用PKT_SV_DATA类型
        net.send(file_data, pkt_type=PKT_SV_DATA, frame_id=frame_id)
        
        print(f"[Sender] Sent file {file_path.name} (frame_id={frame_id}, {len(file_data):,} bytes)")
        
        # 关闭网络连接
        net.close()
        
    except Exception as e:
        print(f"[Sender] Error sending file {file_path}: {e}")

# ------------------------------
# 6) 主流程（5帧分组）——记录结构
# ------------------------------
@dataclass
class FrameRec:
    idx: int
    size: int
    time_ms: int


# ------------------------------
# 7) 封装：g5_pix2pix_grace
# ------------------------------
def run_g5_pix2pix_grace(args, frames: List[Path], pix: Pix2PixRunner, grace: GraceBundle) -> List[FrameRec]:
    recs: List[FrameRec] = []

    # 并行参考需要保留第0组关键帧解码结果
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

        # 组内第0帧：pix2pix → Grace 完整编码（计时仅包含这两步）
        f0 = group_files[0]
        global_frame_idx += 1
        rgb0 = load_rgb_np(f0)

        _sync_cuda()
        t0 = time.perf_counter()
        rgb0_pix = pix.infer_numpy(rgb0)   # 生成后的图

        # 确定参考帧与 is_iframe（不计 IO/保存时间）
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

        # 保存编码后的关键帧
        out_bin_path = args.out_dir / f'grace_stream_{s}.bin'
        try:
            out_bin_path.unlink()  # 新增：删除旧文件，避免混入旧格式/旧记录
        except FileNotFoundError:
            pass
        append_grace_record(out_bin_path, eframe=eframe, compressed_data=compressed_data)
        
        # 发送生成的文件
        send_file_via_netgear(out_bin_path, frame_id=s)


        # 维护参考缓存
        if gi == 0:
            ref_keyframe_np_parallel0 = dec0_np.copy()
        prev_group_key_np = dec0_np.copy()

        # 组内 1..4 帧：仅统计大小（prev, cur），计时仅包含 grace 路径
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


            # 保存mv_j到文件
            out_bin_path = args.out_dir / f'grace_stream_{s+j}.bin'
            try:
                out_bin_path.unlink()  # 新增：删除旧文件，避免混入旧格式/旧记录
            except FileNotFoundError:
                pass
            append_grace_record(out_bin_path, frame_type='mv', mv_j=mv_j, shape_j=shape_j)
            
            # 发送生成的文件
            send_file_via_netgear(out_bin_path, frame_id=s+j)


    return recs


# ------------------------------
# 7) 预留：其它模式占位
# ------------------------------
def run_xxx(args, frames: List[Path], pix: Pix2PixRunner, grace: GraceBundle) -> List[FrameRec]:
    raise NotImplementedError("mode=xxx 尚未实现；请补充 run_xxx(...) 的具体逻辑。")


# ------------------------------
# 8) main：按 mode 分发
# ------------------------------
def main():
    args = build_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_policy = 'parallel' if args.parallel else 'serial'

    # 清空args.out_dir中的内容
    for item in args.out_dir.iterdir():
        if item.is_file():
            item.unlink()

    # 载入源帧
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        if args.rgb_dir:
            frames_dir = args.rgb_dir
            src_fps = None   # 图片目录未知原fps
        else:
            frames_dir, src_fps = video_to_temp_frames(args.video, args.resize, tmp)

        frames_all = list_frames(frames_dir)
        if len(frames_all) == 0:
            raise RuntimeError(f'未找到帧：{frames_dir}/frame_XXXX.png')

        # 采样：根据 fps 子采样
        if src_fps:  # 视频场景
            step = max(1, int(round(src_fps / args.fps)))
        else:        # 目录场景
            step = max(1, int(round(30.0 / args.fps)))  # 若未知，假设原始≈30fps
        frames = frames_all[::step]
        print(f'[INFO] 源帧数={len(frames_all)}，步长={step}，采样后帧数={len(frames)}')

        # 仅处理前 N 帧
        if args.max_frames is not None:
            frames = frames[:max(0, int(args.max_frames))]
            print(f'[INFO] 仅处理前 {len(frames)} 帧（--max-frames 生效）')
        if len(frames) == 0:
            print('[WARN] 没有帧可处理（--max-frames 过小？）')
            return

        # 统一构造 Pix2Pix & Grace
        pix = Pix2PixRunner(args.pix2pix_root, args.pix2pix_ckpt,
                            args.pix2pix_netG, args.pix2pix_input_nc,
                            args.pix2pix_output_nc, args.pix2pix_config_str,
                            args.pix2pix_ngf)
        grace = GraceBundle(args.grace_root, args.grace_model_id)

        # ===== 模式分发 =====
        if args.mode == 'g5_pix2pix_grace':
            recs: List[FrameRec] = run_g5_pix2pix_grace(args, frames, pix, grace)
        elif args.mode == 'xxx':
            recs: List[FrameRec] = run_xxx(args, frames, pix, grace)
        else:
            raise RuntimeError(f"不支持的 mode: {args.mode}")

        # 保存 CSV（文件名包含 ref_policy）
        import csv
        csv_path = args.out_dir / f'metrics_{ref_policy}.csv'
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_idx', 'size', 'time_ms'])
            for r in recs:
                w.writerow([r.idx, r.size, r.time_ms])
        print(f'\n[OK] 写入 {csv_path}')

if __name__ == '__main__':
    main()


'''
python shb_viduce_server_sender.py \
  --rgb-dir /data/wxk/workspace/mirage/dataset/video000/rgb \
  --fps 30 --mode g5_pix2pix_grace \
  --pix2pix-ckpt  /data/wxk/workspace/mirage/dataset/video000/pix2pix_nb/log/albedo/compressed888/latest_net_G.pth \
  --grace-root    /home/wxk/workspace/nsdi/Intrinsic \
  --grace-model-id 64 \
  --pix2pix-netG sub_mobile_resnet_9blocks \
  --pix2pix-config-str 8_8_8_8_8_8_8_8 \
  --out-dir /data/wxk/workspace/mirage/dataset/video000/Viduce/sender_out \
  --max-frames 20
'''