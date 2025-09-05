# receiver.py
# -*- coding: utf-8 -*-
"""
接收端：解析三类包；按 C++ 客户端“两阶段 XOR 处理”更新统计并完成帧。
对外暴露：
    receive_packet(data: bytes) -> list[int]      # 返回本次新近“完成”的 frame_id 列表
"""
from __future__ import annotations
from typing import Dict, Set, List, Tuple
from .packets import DataPacket, FecPacket, XorPacket, MAGIC, VER, T_DATA, T_FEC, T_XOR
from pathlib import Path
import threading
# =========================
# 单实例接收上下文
# =========================
frame_body_bytes: Dict[int, int] = {}     # fid -> fullLen
fec_body_bytes: Dict[int, int] = {}      # fid -> min(fullLen, max_pay_load)i
fec_body_rcvd: Dict[int, int] = {}      
frame_body_rcvd: Dict[int, int] = {}      # fid -> 已计入字节（含 XOR credit）
frame_pkt_num: Dict[int, int] = {}        # fid -> 总分片数
frame_chunks: Dict[int, Set[int]] = {}    # fid -> 已收到分片 id 集合
played_frames: Set[int] = set()           # 已完成的帧
undecoded_xor_payloads: List[bytes] = []  # 暂不可解的 XOR 原始包（延后再试）
max_pay_load = 1400  # 假设的最大包载荷（可根据实际情况调整）

# === NEW: log config & helper ===
LOG_PATH = Path("receive_packet.txt")
LOG_LOCK = threading.Lock()

def _log_packet(typ_str: str, packet_id: int, frame_id: int | str,pkt_num: int = 0, xor_fids: List[int] | None = None) -> None:
    """
    将单个已解析包记录到 receive_packet.txt
    行格式:
        <packet_id>\t<frame_id>\t<type>[ \tframes=fid1,fid2,...]
    - 对于 XOR 包, frame_id 记为 'NA', 并追加 frames= 列表
    """
    if xor_fids:
        line = f"{packet_id}\t{frame_id}\t{typ_str}\tframes=" + ",".join(map(str, xor_fids)) + "\n"
    else:
        line = f"{packet_id}\t{frame_id}\t{typ_str}\t{pkt_num}\n"
    with LOG_LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)


def _is_frame_complete(fid: int) -> bool:
    if fid in played_frames:
        return True
    n = frame_pkt_num.get(fid)
    if n is not None and len(frame_chunks.get(fid, set())) >= n:
        return True

    return False

def _try_finalize(fid: int, newly: List[int]):
    if fid in played_frames:
        return
    if _is_frame_complete(fid):
        played_frames.add(fid)
        newly.append(fid)

def _process_pending_xor(newly: List[int]):
    """
    复刻 C++ 的 ProcessPendingXorPkts：missingCnt<=1 则可解；
    可解时把 credit 饱和累加到各帧，再尝试将其标记为完成。
    """
    i = 0
    while i < len(undecoded_xor_payloads):
        raw = undecoded_xor_payloads[i]
        try:
            xp = XorPacket.from_bytes(raw)
        except Exception:
            undecoded_xor_payloads.pop(i)
            continue

        items = xp.items
        # 阶段 A：可解性
        missing_cnt = 0
        touched: List[int] = []
        for (fid, full16, cred16) in items:
            if fid not in fec_body_bytes:
                fec_body_bytes.setdefault(fid,min(full16,max_pay_load))
                fec_body_rcvd.setdefault(fid,0)
            if not _is_frame_complete(fid):
                missing_cnt += 1
            touched.append(fid)

        if missing_cnt > 1:
            i += 1
            continue

        # 阶段 B：可解 -> 饱和累加 credit
        for (fid, full16, cred16) in items:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            fec_body_rcvd[fid] = min(full, rcvd + int(cred16))

        # 阶段 C：完成度检查
        for fid in touched:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            if full is not None and rcvd >= full:
                frame_chunks.get(fid, set()).add(len(frame_chunks.get(fid, set())))
            _try_finalize(fid, newly)
        #         frame_chunks.get(xp.frame_id, set()).add(len(frame_chunks.get(xp.frame_id, set())))
        # _try_finalize(fid, newly)

        undecoded_xor_payloads.pop(i)  # 已处理，移除（不自增 i）,继续遍历下一个

def receive_packet(data: bytes) -> List[int]:
    """
    解析任一包（Data/FEC/XOR），更新接收端状态，并返回本次因此完成的 frame_id 列表。
    """
    newly: List[int] = []
    if len(data) < 4 or data[:2] != MAGIC or data[2] != VER:
        print(f"{len(data)} {data[:2]} {data[2]}")
        return newly, 0
    typ = data[3]

    packet_id = 0
    # -------- Data --------
    if typ == T_DATA:
        dp = DataPacket.from_bytes(data)
        packet_id = dp.packet_id
        frame_body_bytes.setdefault(dp.frame_id, dp.frame_len)
        frame_pkt_num.setdefault(dp.frame_id, dp.frame_pkt_num)
        frame_chunks.setdefault(dp.frame_id, set()).add(dp.pkt_id_in_frame)
        _log_packet("data", packet_id=dp.packet_id, frame_id=dp.frame_id,pkt_num=dp.frame_pkt_num)
        _try_finalize(dp.frame_id, newly)
        _process_pending_xor(newly)
        return newly, packet_id

    # -------- FEC（仅作为提示）--------
    if typ == T_FEC:
        fp = FecPacket.from_bytes(data)
        packet_id = fp.packet_id
        _log_packet("fec", packet_id=fp.packet_id, frame_id=fp.frame_id)
        # 不做真实 FEC 恢复，仅尝试触发 XOR 队列再处理
        frame_chunks.get(fp.frame_id, set()).add(len(frame_chunks.get(fp.frame_id, set())))
        _try_finalize(fp.frame_id, newly)
        _process_pending_xor(newly)

        return newly, packet_id

    # -------- XOR --------
    if typ == T_XOR:
        try:
            xp = XorPacket.from_bytes(data)
        except Exception:
            return newly, packet_id

        packet_id = xp.packet_id
        items = xp.items

        xor_fids_for_log = [int(fid) for (fid, _, _) in items]
        _log_packet("xor", packet_id=xp.packet_id, frame_id="NA", xor_fids=xor_fids_for_log)
        
        # 阶段 A：可解性
        missing_cnt = 0
        touched: List[int] = []
        for (fid, full16, cred16) in items:
            if fid not in fec_body_bytes:
                fec_body_bytes.setdefault(fid,min(full16,max_pay_load))
                fec_body_rcvd.setdefault(fid,0)
            if not _is_frame_complete(fid):
                missing_cnt += 1
            touched.append(fid)

        if missing_cnt > 1:
            undecoded_xor_payloads.append(data)
            return newly, packet_id

        # 阶段 B：可解 -> 饱和累加 credit
        for (fid, full16, cred16) in items:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            fec_body_rcvd[fid] = min(full, rcvd + int(cred16))

        # 阶段 C：完成度检查
        for fid in touched:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            if full is not None and rcvd >= full:
                frame_chunks.get(fid, set()).add(len(frame_chunks.get(fid, set())))
            _try_finalize(fid, newly)
        #         frame_chunks.get(xp.frame_id, set()).add(len(frame_chunks.get(xp.frame_id, set())))
        # _try_finalize(fid, newly)
        return newly, packet_id

    return newly, packet_id
