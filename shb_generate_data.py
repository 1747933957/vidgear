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
from vidgear.gears.unified_netgear import NetGearLike as NetGear
# 两层上级：保证能 import 到 shb_viduce_runner.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from shb_viduce_runner import *
# ====== 常量配置 ======
CLIENT_ADDR = "127.0.0.1" # 本机测试："172.27.143.41"
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ------------------------------
# 5) 网络发送相关函数
# ------------------------------
_GLOBAL_TCP_NET = None

# 初始化全局变量
pre_time = time.time()

def _ensure_global_tcp_net() -> NetGear:
    """首次创建并缓存全局 TCP NetGear；随后复用。"""
    global _GLOBAL_TCP_NET
    if _GLOBAL_TCP_NET is not None:
        return _GLOBAL_TCP_NET
    _GLOBAL_TCP_NET = NetGear(
        address=CLIENT_ADDR,
        port=CLIENT_PORT,
        protocol="tcp",
        pattern=1,                # REQ/REP
        receive_mode=False,
        logging=True,
        jpeg_compression=False,   # 关闭JPEG，避免破坏原始字节流
        # 不启用 bidirectional_mode（发送端不需要）
    )
    return _GLOBAL_TCP_NET

def close_global_tcp_net():
    """若需要在程序结束时手动关闭全局连接，可调用本函数。"""
    global _GLOBAL_TCP_NET
    if _GLOBAL_TCP_NET is not None:
        try:
            _GLOBAL_TCP_NET.close()
        finally:
            _GLOBAL_TCP_NET = None

# ------------------------------
# 6) 主流程（5帧分组）——记录结构
# ------------------------------
@dataclass
class FrameRec:
    idx: int
    size: int
    pix2pix_ms: int
    grace_enc_ms: int


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
        t_pix0 = time.perf_counter()
        rgb0_pix = pix.infer_numpy(rgb0)   # 生成后的图
        _sync_cuda()
        t_pix1 = time.perf_counter()
        t_pix_ms = int(round((t_pix1 - t_pix0) * 1000))


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
        
        _sync_cuda()
        t_ge0 = time.perf_counter()
        size0, dec0_np, eframe, compressed_data = grace.encode_keyframe(ref_np=ref_np, cur_np=rgb0_pix, is_iframe=is_iframe)
        _sync_cuda()
        t_ge1 = time.perf_counter()
        t_grace_ms = int(round((t_ge1 - t_ge0) * 1000))
        print(f'  [KEY] size={size0}  pix2pix_ms={t_pix_ms}  grace_enc_ms={t_grace_ms}')
        recs.append(FrameRec(idx=s, size=size0, pix2pix_ms=t_pix_ms, grace_enc_ms=t_grace_ms))

        # 保存编码后的关键帧
        out_bin_path = args.out_dir / f'grace_stream_{(s):04d}.bin'
        try:
            out_bin_path.unlink()  # 新增：删除旧文件，避免混入旧格式/旧记录
        except FileNotFoundError:
            pass
        eframe.ref_id = s-5
        append_grace_record(out_bin_path, eframe=eframe, compressed_data=compressed_data)

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
            t_grace_ms = int(round((t3 - t2) * 1000))

            print(f'  [MV]  frame={fj.name}  size={sizej}  pix2pix_ms=0  grace_enc_ms={t_grace_ms}')
            recs.append(FrameRec(idx=s+j, size=sizej, pix2pix_ms=0, grace_enc_ms=t_grace_ms))


            # 保存mv_j到文件
            out_bin_path = args.out_dir / f'grace_stream_{(s + j):04d}.bin'
            try:
                out_bin_path.unlink()  # 新增：删除旧文件，避免混入旧格式/旧记录
            except FileNotFoundError:
                pass
            append_grace_record(out_bin_path, frame_type='mv', mv_j=mv_j, shape_j=shape_j)

    return recs


# ------------------------------
# 7) 根据pix2pix运行时间，自适应调节关键帧的间隔
# ------------------------------
def run_adaptive_mv_serial(args, frames: List[Path], pix: Pix2PixRunner, grace: GraceBundle) -> List[FrameRec]:
    """
    自适应 mv 间隔：
      - 第 0 帧编码为 I 帧，记其 pix2pix 用时 t0（毫秒）
      - 接下来 ceil(t0/30)-1 帧编码为 MV
      - 再下一帧（记为 x）编码为 P（参考帧固定为 0 号），记其 pix2pix 用时 t1
      - 接下来 ceil(t1/30)-1 帧编码为 MV
      - 再下一帧编码为 P（参考帧改为 x），以此类推，直至耗尽帧
    记录字段 ['frame_idx','size','pix2pix_ms','grace_enc_ms'] 与 g5 方案一致。
    """
    recs: List[FrameRec] = []

    # 缓存关键帧(I/P)的解码图，供后续 P 参考；键为全局帧号
    decoded_cache: dict[int, np.ndarray] = {}

    # 统一尺寸：以第 0 帧为基准，后续读图/推理均按此尺寸
    f0 = frames[0]
    rgb0 = load_rgb_np(f0)                         # 基准读取确定尺寸
    base_h, base_w = rgb0.shape[:2]

    # ===== 第 0 帧：I 帧 =====
    _sync_cuda()
    t0 = time.perf_counter()
    rgb0_pix = pix.infer_numpy(rgb0)               # 第 0 帧先跑 pix2pix
    _sync_cuda()
    t1 = time.perf_counter()
    pix_ms = int(round((t1 - t0) * 1000))

    # I 帧编码（保持与 g5 中 I 帧一致：ref_np 取自身像素图，is_iframe=True）
    _sync_cuda()
    ge0 = time.perf_counter()
    size0, dec0_np, eframe, compressed_data = grace.encode_keyframe(
        ref_np=rgb0_pix, cur_np=rgb0_pix, is_iframe=True
    )
    _sync_cuda()
    ge1 = time.perf_counter()
    grace_ms = int(round((ge1 - ge0) * 1000))

    print(f'  [I  ] frame=0  size={size0}  pix2pix_ms={pix_ms}  grace_enc_ms={grace_ms}')
    recs.append(FrameRec(idx=0, size=size0, pix2pix_ms=pix_ms, grace_enc_ms=grace_ms))

    # 写 bin（I 帧的 ref_id 不参与解码引用，这里可置 0）
    out_bin_path = args.out_dir / f'grace_stream_{0:04d}.bin'
    try:
        out_bin_path.unlink()
    except FileNotFoundError:
        pass
    eframe.ref_id = 0  # 注：仅 P 帧在解码端会读取 ref_id，这里置 0 仅占位
    append_grace_record(out_bin_path, eframe=eframe, compressed_data=compressed_data)

    # 放入参考缓存
    decoded_cache[0] = dec0_np.copy()

    # 依据 t 计算本轮 MV 数量：ceil(t/30)-1（最少为 0）
    def mv_count_for(pix_ms: int) -> int:
        return max(0, int(math.ceil(pix_ms / 30.0)) - 1)

    # 当前参考关键帧（上一关键帧）的全局帧号；初始为 0（I 帧）
    ref_kf_idx = 0
    cur = 1                           # 下一待处理帧号
    last_pix_ms = pix_ms              # 上一次关键帧的 pix2pix 时长（用于推导下一段 MV 数）
    last_mv_prev_np: Optional[np.ndarray] = rgb0  # MV 计算需要 prev_np，初始化为第 0 帧原图

    # 主循环：交替“MV 段 + 1 个 P”
    while cur < len(frames):
        # === 1) 生成一段 MV 帧 ===
        need_mv = mv_count_for(last_pix_ms)
        for _ in range(need_mv):
            if cur >= len(frames):
                break
            # 加载当前帧与前一帧原图（按基准尺寸）
            cur_np  = load_rgb_np(frames[cur],   target_size=(base_h, base_w))
            prev_np = load_rgb_np(frames[cur-1], target_size=(base_h, base_w)) if cur > 0 else last_mv_prev_np

            _sync_cuda()
            t2 = time.perf_counter()
            size_mv, mv_blob, shape_mv = grace.mv_only_size(prev_np=prev_np, cur_np=cur_np)
            _sync_cuda()
            t3 = time.perf_counter()
            t_grace_ms = int(round((t3 - t2) * 1000))

            print(f'  [MV ] frame={cur}  size={size_mv}  pix2pix_ms=0  grace_enc_ms={t_grace_ms}')
            recs.append(FrameRec(idx=cur, size=size_mv, pix2pix_ms=0, grace_enc_ms=t_grace_ms))

            # 写 bin（仅保存 mv）
            out_bin_path = args.out_dir / f'grace_stream_{cur:04d}.bin'
            try:
                out_bin_path.unlink()
            except FileNotFoundError:
                pass
            append_grace_record(out_bin_path, frame_type='mv', mv_j=mv_blob, shape_j=shape_mv)

            last_mv_prev_np = cur_np
            cur += 1

        if cur >= len(frames):
            break

        # === 2) 一个 P 帧，参考 ref_kf_idx ===
        rgb_p = load_rgb_np(frames[cur], target_size=(base_h, base_w))

        _sync_cuda()
        t4 = time.perf_counter()
        rgb_p_pix = pix.infer_numpy(rgb_p)              # P 帧也跑 pix2pix，计时入表
        _sync_cuda()
        t5 = time.perf_counter()
        pix_ms_p = int(round((t5 - t4) * 1000))

        # 参考图：用最近关键帧（ref_kf_idx）的解码图
        if ref_kf_idx not in decoded_cache:
            raise RuntimeError(f'找不到参考关键帧解码图：ref_kf_idx={ref_kf_idx}')
        ref_np = decoded_cache[ref_kf_idx]

        _sync_cuda()
        ge2 = time.perf_counter()
        size_p, dec_p_np, eframe_p, compressed_p = grace.encode_keyframe(
            ref_np=ref_np, cur_np=rgb_p_pix, is_iframe=False
        )
        _sync_cuda()
        ge3 = time.perf_counter()
        grace_ms_p = int(round((ge3 - ge2) * 1000))

        print(f'  [P  ] frame={cur}  ref={ref_kf_idx}  size={size_p}  pix2pix_ms={pix_ms_p}  grace_enc_ms={grace_ms_p}')
        recs.append(FrameRec(idx=cur, size=size_p, pix2pix_ms=pix_ms_p, grace_enc_ms=grace_ms_p))

        # 写 bin：记下参考关键帧 id（供服务端按 PNG 路径查参考）
        out_bin_path = args.out_dir / f'grace_stream_{cur:04d}.bin'
        try:
            out_bin_path.unlink()
        except FileNotFoundError:
            pass
        eframe_p.ref_id = int(ref_kf_idx)               # 关键：P 帧携带 ref_id
        append_grace_record(out_bin_path, eframe=eframe_p, compressed_data=compressed_p)

        # 更新关键帧参考与缓存
        decoded_cache[cur] = dec_p_np.copy()
        ref_kf_idx = cur
        last_pix_ms = pix_ms_p
        last_mv_prev_np = rgb_p
        cur += 1

    return recs

def run_adaptive_mv(args, frames: List[Path], pix: Pix2PixRunner, grace: GraceBundle) -> List[FrameRec]:
    """
    自适应 mv 间隔：
      - 第 0 帧编码为 I 帧，记其 pix2pix 用时 t0（毫秒）
      - 接下来 ceil(t0/30)-1 帧编码为 MV
      - 再下一帧编码为 P（参考帧固定为 0 号），记其 pix2pix 用时 t1
      - 接下来 ceil(t1/30)-1 帧编码为 MV
      - 再下一帧编码为 P（仍参考第 0 帧），以此类推，直至耗尽帧
    记录字段 ['frame_idx','size','pix2pix_ms','grace_enc_ms'] 与 g5 方案一致。
    """
    recs: List[FrameRec] = []

    # 缓存关键帧(I)的解码图，供后续 P 参考
    decoded_cache: dict[int, np.ndarray] = {}

    # 统一尺寸：以第 0 帧为基准
    f0 = frames[0]
    rgb0 = load_rgb_np(f0)
    base_h, base_w = rgb0.shape[:2]

    # ===== 第 0 帧：I 帧 =====
    _sync_cuda()
    t0 = time.perf_counter()
    rgb0_pix = pix.infer_numpy(rgb0)               # 第 0 帧先跑 pix2pix
    _sync_cuda()
    t1 = time.perf_counter()
    pix_ms = int(round((t1 - t0) * 1000))

    # I 帧编码
    _sync_cuda()
    ge0 = time.perf_counter()
    size0, dec0_np, eframe, compressed_data = grace.encode_keyframe(
        ref_np=rgb0_pix, cur_np=rgb0_pix, is_iframe=True
    )
    _sync_cuda()
    ge1 = time.perf_counter()
    grace_ms = int(round((ge1 - ge0) * 1000))

    print(f'  [I  ] frame=0  size={size0}  pix2pix_ms={pix_ms}  grace_enc_ms={grace_ms}')
    recs.append(FrameRec(idx=0, size=size0, pix2pix_ms=pix_ms, grace_enc_ms=grace_ms))

    # 写 bin
    out_bin_path = args.out_dir / f'grace_stream_{0:04d}.bin'
    try:
        out_bin_path.unlink()
    except FileNotFoundError:
        pass
    eframe.ref_id = 0
    append_grace_record(out_bin_path, eframe=eframe, compressed_data=compressed_data)

    # 缓存第 0 帧
    decoded_cache[0] = dec0_np.copy()

    # 依据 t 计算 MV 数量
    def mv_count_for(pix_ms: int) -> int:
        return max(0, int(math.ceil(pix_ms / 30.0)) - 1)

    cur = 1
    last_pix_ms = pix_ms
    last_mv_prev_np: Optional[np.ndarray] = rgb0

    # 主循环
    while cur < len(frames):
        # === 1) 一段 MV 帧 ===
        need_mv = mv_count_for(last_pix_ms)
        for _ in range(need_mv):
            if cur >= len(frames):
                break
            cur_np  = load_rgb_np(frames[cur],   target_size=(base_h, base_w))
            prev_np = load_rgb_np(frames[cur-1], target_size=(base_h, base_w)) if cur > 0 else last_mv_prev_np

            _sync_cuda()
            t2 = time.perf_counter()
            size_mv, mv_blob, shape_mv = grace.mv_only_size(prev_np=prev_np, cur_np=cur_np)
            _sync_cuda()
            t3 = time.perf_counter()
            t_grace_ms = int(round((t3 - t2) * 1000))

            print(f'  [MV ] frame={cur}  size={size_mv}  pix2pix_ms=0  grace_enc_ms={t_grace_ms}')
            recs.append(FrameRec(idx=cur, size=size_mv, pix2pix_ms=0, grace_enc_ms=t_grace_ms))

            out_bin_path = args.out_dir / f'grace_stream_{cur:04d}.bin'
            try:
                out_bin_path.unlink()
            except FileNotFoundError:
                pass
            append_grace_record(out_bin_path, frame_type='mv', mv_j=mv_blob, shape_j=shape_mv)

            last_mv_prev_np = cur_np
            cur += 1

        if cur >= len(frames):
            break

        # === 2) 一个 P 帧，始终参考第 0 帧 ===
        rgb_p = load_rgb_np(frames[cur], target_size=(base_h, base_w))

        _sync_cuda()
        t4 = time.perf_counter()
        rgb_p_pix = pix.infer_numpy(rgb_p)
        _sync_cuda()
        t5 = time.perf_counter()
        pix_ms_p = int(round((t5 - t4) * 1000))

        if 0 not in decoded_cache:
            raise RuntimeError("找不到参考关键帧解码图：frame 0")
        ref_np = decoded_cache[0]

        _sync_cuda()
        ge2 = time.perf_counter()
        size_p, dec_p_np, eframe_p, compressed_p = grace.encode_keyframe(
            ref_np=ref_np, cur_np=rgb_p_pix, is_iframe=False
        )
        _sync_cuda()
        ge3 = time.perf_counter()
        grace_ms_p = int(round((ge3 - ge2) * 1000))

        print(f'  [P  ] frame={cur}  ref=0  size={size_p}  pix2pix_ms={pix_ms_p}  grace_enc_ms={grace_ms_p}')
        recs.append(FrameRec(idx=cur, size=size_p, pix2pix_ms=pix_ms_p, grace_enc_ms=grace_ms_p))

        out_bin_path = args.out_dir / f'grace_stream_{cur:04d}.bin'
        try:
            out_bin_path.unlink()
        except FileNotFoundError:
            pass
        eframe_p.ref_id = 0
        append_grace_record(out_bin_path, eframe=eframe_p, compressed_data=compressed_p)

        decoded_cache[cur] = dec_p_np.copy()
        last_pix_ms = pix_ms_p
        last_mv_prev_np = rgb_p
        cur += 1

    return recs


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
            src_fps = 30   # 图片目录未知原fps
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
        elif args.mode == 'adaptive_mv':
            recs: List[FrameRec] = run_adaptive_mv(args, frames, pix, grace)
        else:
            raise RuntimeError(f"不支持的 mode: {args.mode}")

        # 保存 CSV（文件名包含 ref_policy）
        import csv
        csv_path = args.out_dir / f'metrics_{ref_policy}.csv'
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_idx', 'size', 'pix2pix_ms', 'grace_enc_ms'])
            for r in recs:
                w.writerow([r.idx, r.size, r.pix2pix_ms, r.grace_enc_ms])
        print(f'\n[OK] 写入 {csv_path}')


if __name__ == '__main__':
    main()


'''
python shb_generate_data.py \
  --rgb-dir /data/wxk/workspace/mirage/dataset/video000/rgb \
  --fps 30 --mode adaptive_mv \
  --pix2pix-ckpt  /data/wxk/workspace/mirage/dataset/video000/pix2pix_nb/log/albedo/compressed888/latest_net_G.pth \
  --grace-root    /home/wxk/workspace/nsdi/Intrinsic \
  --grace-model-id 64 \
  --pix2pix-netG sub_mobile_resnet_9blocks \
  --pix2pix-config-str 8_8_8_8_8_8_8_8 \
  --out-dir /data/wxk/workspace/mirage/dataset/video000/Viduce/sender_out \
  --max-frames 20
'''