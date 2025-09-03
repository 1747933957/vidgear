# -*- coding: utf-8 -*-
"""
接收端：解析三类包；按 C++ 客户端“两阶段 XOR 处理”更新统计并完成帧。
对外暴露：
    receive_packet(data: bytes) -> List[int] | (List[int], int)
      - 返回本次新近“完成”的 frame_id 列表（兼容已有调用，必要时也返回 packet_id）
"""

from __future__ import annotations
from typing import Dict, Set, List, Tuple
from pathlib import Path
import json

# === 你的工程内的包头与数据结构（保持不变） ===
from .packets import DataPacket, FecPacket, XorPacket, MAGIC, VER, T_DATA, T_FEC, T_XOR

# =========================
# 单实例接收上下文（保持不变）
# =========================
frame_body_bytes: Dict[int, int] = {}     # fid -> fullLen
fec_body_bytes: Dict[int, int] = {}       # fid -> min(fullLen, max_pay_load)
fec_body_rcvd: Dict[int, int] = {}        # fid -> 通过 XOR credit 饱和累加得到的“可恢复字节”
frame_body_rcvd: Dict[int, int] = {}      # fid -> 已计入字节（含 XOR credit）
frame_pkt_num: Dict[int, int] = {}        # fid -> 总分片数
frame_chunks: Dict[int, Set[int]] = {}    # fid -> 已收到分片 id 集合（完成度判定仅看数量）
played_frames: Set[int] = set()           # 已完成的帧
undecoded_xor_payloads: List[bytes] = []  # 暂不可解的 XOR 原始包（延后再试）
max_pay_load = 1400                       # 假设的最大包载荷（可根据实际情况调整)

def _is_frame_complete(fid: int) -> bool:
    if fid in played_frames:
        return True
    n = frame_pkt_num.get(fid)
    if n is not None and len(frame_chunks.get(fid, set())) >= n:
        return True
    return False

def _try_finalize(fid: int, newly: List[int]) -> None:
    if fid in played_frames:
        return
    if _is_frame_complete(fid):
        played_frames.add(fid)
        newly.append(fid)

def _process_pending_xor(newly: List[int]) -> None:
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
                fec_body_bytes.setdefault(fid, min(full16, max_pay_load))
                fec_body_rcvd.setdefault(fid, 0)
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

        undecoded_xor_payloads.pop(i)  # 已处理，移除（不自增 i）

# =========================
# 统计输出 & 丢包跟踪（这里做窗口解耦）
# =========================

# 统计文件：第 1 行是表头；第 n 帧写到第 n+2 行
STATS_TXT_PATH = Path("recv_stats.txt")

# === NEW === 将“记录向量的窗口”和“计算丢包率的窗口”拆分（已存在，保留）
WINDOW_VEC_N  = 5    # 控制写入文件的 0/1 向量长度（例如最近 5 个包）
WINDOW_LOSS_N = 5    # 控制计算丢包率的窗口长度（例如最近 5 个包）

# 全局 packet_id 到达/缺失跟踪
seen_packet_ids: Set[int] = set()
missing_packet_ids: Set[int] = set()
max_packet_id_global: int = -1

# 每帧首个 DATA 的 packet_id（代表“该帧发送本体数据时刻”的起点）
frame_first_data_pid: Dict[int, int] = {}
# 每帧当前“已见到的 DATA/FEC 的最大 packet_id”（作为滑窗基线，确保至少覆盖该帧 DATA+FEC）
latest_df_pid: Dict[int, int] = {}
# 每帧是否已写过一行（用于避免重复回填）
frame_line_written: Dict[int, bool] = {}
# 连续已写入的最大帧 id（用于回填前序缺行）
last_written_fid: int = -1

# === NEW === 仅第一次调用时重置文件头；避免每次读都清空旧内容
_STATS_INITIALIZED: bool = False
def _ensure_stats_file() -> None:
    global _STATS_INITIALIZED
    if not _STATS_INITIALIZED:
        STATS_TXT_PATH.write_text(
            "frame_id\tloss_rate_at_send\tredundancy_rounds\trecentN_vector\n",
            encoding="utf-8"
        )
        _STATS_INITIALIZED = True
    elif not STATS_TXT_PATH.exists():
        # 文件被意外删除时，重建表头
        STATS_TXT_PATH.write_text(
            "frame_id\tloss_rate_at_send\tredundancy_rounds\trecentN_vector\n",
            encoding="utf-8"
        )

def _read_all_lines() -> List[str]:
    _ensure_stats_file()
    with STATS_TXT_PATH.open("r", encoding="utf-8") as f:
        return f.readlines()

def _write_all_lines(lines: List[str]) -> None:
    with STATS_TXT_PATH.open("w", encoding="utf-8") as f:
        f.writelines(lines)

def _write_line_at(fid: int, line_text: str) -> None:
    """
    将 fid 对应的数据写到“第 fid+2 行”（1-based）；支持覆写/扩展。
    """
    lines = _read_all_lines()
    need = fid + 2
    if len(lines) < need:
        lines.extend(["\n"] * (need - len(lines)))
    lines[fid + 1] = line_text.rstrip("\n") + "\n"
    _write_all_lines(lines)
    frame_line_written[fid] = True

def _format_line(fid: int, loss_rate: float | None, redund_rounds: int | None, vec: List[int]) -> str:
    """
    生成一行 TSV 文本：
      - loss_rate: None -> "NA"
      - redund_rounds: None -> "NA"
      - vec: JSON 序列化的 0/1 列表
    """
    lr_str = f"{loss_rate:.6f}" if loss_rate is not None else "NA"
    rr_str = str(redund_rounds) if redund_rounds is not None else "NA"
    return f"{fid}\t{lr_str}\t{rr_str}\t{json.dumps(vec, ensure_ascii=False)}"

def _note_packet(pid: int) -> None:
    """
    记录一个 packet_id 的到达；若出现跳号则把间隔标为 missing（支持乱序）。
    """
    global max_packet_id_global
    if pid is None:
        return
    if pid > max_packet_id_global + 1:
        for mid in range(max_packet_id_global + 1, pid):
            if mid not in seen_packet_ids:
                missing_packet_ids.add(mid)
    if pid > max_packet_id_global:
        max_packet_id_global = pid
    seen_packet_ids.add(pid)
    if pid in missing_packet_ids:
        missing_packet_ids.discard(pid)

# === NEW === 核心：分别用两个窗口计算“向量”和“丢包率”，并强制长度校验
def _compute_vec_and_loss_for_frame(fid: int, baseline_pid: int) -> Tuple[List[int], float]:
    """
    以 baseline_pid 作为窗口右端：
      - 向量窗口长度 = WINDOW_VEC_N
      - 丢包率窗口长度 = WINDOW_LOSS_N
    两者独立计算，互不影响；左端尽量涵盖 [首 DATA pid, baseline]，不足则左侧补 1（虚拟到达）以稳定长度。
    """
    # —— 0/1 向量（长度 WINDOW_VEC_N）——
    first_data_pid = frame_first_data_pid.get(fid, baseline_pid)
    start_vec_natural = min(first_data_pid, baseline_pid - (WINDOW_VEC_N - 1))
    start_vec = max(0, start_vec_natural)

    vec: List[int] = []
    for pid in range(start_vec, baseline_pid):
        if pid in seen_packet_ids:
            vec.append(1)
        elif pid in missing_packet_ids:
            vec.append(0)
        else:
            vec.append(0)  # 未明确标记，保守记 0
    if len(vec) < WINDOW_VEC_N:
        vec = [1] * (WINDOW_VEC_N - len(vec)) + vec  # 左端以“虚拟到达=1”补齐
    # === NEW === 双保险：严格裁剪/补齐到 WINDOW_VEC_N
    if len(vec) > WINDOW_VEC_N:
        vec = vec[-WINDOW_VEC_N:]

    # —— 丢包率（长度 WINDOW_LOSS_N 的窗口）——
    start_loss_natural = min(first_data_pid, baseline_pid - (WINDOW_LOSS_N - 1))
    start_loss = max(0, start_loss_natural)

    loss_vec: List[int] = []
    for pid in range(start_loss, baseline_pid):
        if pid in seen_packet_ids:
            loss_vec.append(1)
        elif pid in missing_packet_ids:
            loss_vec.append(0)
        else:
            loss_vec.append(0)
    if len(loss_vec) < WINDOW_LOSS_N:
        loss_vec = [1] * (WINDOW_LOSS_N - len(loss_vec)) + loss_vec
    # === NEW === 同步裁剪到 WINDOW_LOSS_N 防止越界
    if len(loss_vec) > WINDOW_LOSS_N:
        loss_vec = loss_vec[-WINDOW_LOSS_N:]

    loss_rate = (loss_vec.count(0) / WINDOW_LOSS_N) if WINDOW_LOSS_N > 0 else 0.0
    return vec, loss_rate

def _backfill_missing_lines(upto_fid: int, proxy_baseline_pid: int) -> None:
    """
    当写入 fid 时，如前面存在尚未写入的帧，则用“当前可见历史”回填这些行（初次写入；后续可覆写）。
    这里采用保守策略：
      - 向量使用全 1，长度 = WINDOW_VEC_N；
      - 丢包率使用 0.0。
    """
    global last_written_fid
    start = last_written_fid + 1
    for pfid in range(start, upto_fid):
        if frame_line_written.get(pfid, False):
            continue
        vec = [1] * WINDOW_VEC_N
        _write_line_at(pfid, _format_line(pfid, 0.0, None, vec))
    last_written_fid = max(last_written_fid, upto_fid)

# =========================
# 基于 XOR 的“冗余持续轮次”统计（保持原语义）
# =========================
xor_last_seen_anchor: int = -1                          # 最近一次收到的 XOR anchor
xor_active_first_anchor: Dict[int, int] = {}            # 源帧 fid -> 首次包含它的 anchor
xor_active_last_anchor: Dict[int, int]  = {}            # 源帧 fid -> 最近一次包含它的 anchor
xor_active_running: Set[int] = set()                    # 当前仍在“连续被包含”的源帧集合
redundancy_rounds_final: Dict[int, int] = {}            # 已定格的持续轮次(fid -> 轮数)

MAX_ROUNDS = 5  # 轮次上限

def _finalize_redund_rounds(fid: int, rounds: int, baseline_pid_fallback: int) -> None:
    """将某帧的冗余持续轮次定格并覆写到文件（若尚未写过则先生成丢包率/向量）"""
    redundancy_rounds_final[fid] = max(0, int(rounds))
    baseline = latest_df_pid.get(fid, baseline_pid_fallback)
    vec, loss_rate = _compute_vec_and_loss_for_frame(fid, baseline)
    _write_line_at(fid, _format_line(fid, loss_rate, redundancy_rounds_final[fid], vec))

def _check_finalize_by_threshold(current_anchor: int, baseline_pid_fallback: int) -> None:
    """对活动集中所有帧做“达到上限即定格”的统一检查"""
    to_finish: List[int] = []
    for fid in list(xor_active_running):
        first_a = xor_active_first_anchor.get(fid)
        last_a  = xor_active_last_anchor.get(fid, current_anchor)
        if first_a is None:
            continue
        rounds_now = max(0, (last_a - first_a + 1))
        if rounds_now >= MAX_ROUNDS:
            _finalize_redund_rounds(fid, MAX_ROUNDS, baseline_pid_fallback)
            to_finish.append(fid)
    for fid in to_finish:
        xor_active_running.discard(fid)
        xor_active_first_anchor.pop(fid, None)
        xor_active_last_anchor.pop(fid, None)

# =========================
# 收包入口（Data/FEC/XOR）
# =========================
def receive_packet(data: bytes):
    """
    解析任一包（Data/FEC/XOR），更新接收端状态，并返回本次因此完成的 frame_id 列表。
    一些现有调用会期待 (newly, packet_id)；为兼容，仍按原写法返回。
    """
    # === NEW === 统一在函数开头声明会写入的全局变量，避免作用域报错
    global xor_last_seen_anchor, redundancy_rounds_final

    newly: List[int] = []
    if len(data) < 4 or data[:2] != MAGIC or data[2] != VER:
        # 保留你之前用于 debug 的输出
        try:
            print(f"{len(data)} {data[:2]} {data[2]}")
        except Exception:
            pass
        return newly, 0

    typ = data[3]
    packet_id = 0

    # -------- Data --------
    if typ == T_DATA:
        dp = DataPacket.from_bytes(data)
        packet_id = dp.packet_id

        # 到达/缺失登记
        _note_packet(packet_id)

        # 基本记录
        frame_body_bytes.setdefault(dp.frame_id, dp.frame_len)
        frame_pkt_num.setdefault(dp.frame_id, dp.frame_pkt_num)
        frame_chunks.setdefault(dp.frame_id, set()).add(dp.pkt_id_in_frame)

        # 首个 DATA pid & 回填前序
        if dp.frame_id not in frame_first_data_pid:
            frame_first_data_pid[dp.frame_id] = packet_id
            _backfill_missing_lines(dp.frame_id, proxy_baseline_pid=packet_id)

        # 更新该帧 DATA/FEC 的最大 pid（作为滑窗基线）
        latest_df_pid[dp.frame_id] = max(latest_df_pid.get(dp.frame_id, -1), packet_id)

        # 写/覆写该帧当前行
        fid = dp.frame_id
        baseline = latest_df_pid[fid]
        vec, loss_rate = _compute_vec_and_loss_for_frame(fid, baseline)
        redund_rounds = redundancy_rounds_final.get(fid, None)
        _write_line_at(fid, _format_line(fid, loss_rate, redund_rounds, vec))

        _try_finalize(dp.frame_id, newly)
        _process_pending_xor(newly)

        # === NEW === 无 XOR 兜底：若从未见过任何 XOR，且该帧已完成但未定格轮次，则置为 6
        if fid in played_frames and fid not in redundancy_rounds_final and xor_last_seen_anchor < 0:
            redundancy_rounds_final[fid] = 6

        # 若帧完成，覆写最终行（冗余若未定格则仍为 NA/或上面的兜底值）
        redund_rounds = redundancy_rounds_final.get(dp.frame_id, None)
        _write_line_at(dp.frame_id, _format_line(dp.frame_id, loss_rate, redund_rounds, vec))

        return newly, packet_id

    # -------- FEC --------
    if typ == T_FEC:
        fp = FecPacket.from_bytes(data)
        packet_id = fp.packet_id

        # 到达/缺失登记
        _note_packet(packet_id)

        # 不做真实 FEC 恢复：用“增加分片计数”的方式触发完成度
        frame_chunks.get(fp.frame_id, set()).add(len(frame_chunks.get(fp.frame_id, set())))

        # 更新该帧 DATA/FEC 的最大 pid（保证窗口至少覆盖本帧 DATA+FEC）
        latest_df_pid[fp.frame_id] = max(latest_df_pid.get(fp.frame_id, -1), packet_id)

        # 若该帧已有首 DATA，则按新基线覆写行
        if fp.frame_id in frame_first_data_pid:
            fid = fp.frame_id
            baseline = latest_df_pid[fid]
            vec, loss_rate = _compute_vec_and_loss_for_frame(fid, baseline)
            redund_rounds = redundancy_rounds_final.get(fid, None)
            _write_line_at(fid, _format_line(fid, loss_rate, redund_rounds, vec))

        _try_finalize(fp.frame_id, newly)
        _process_pending_xor(newly)

        # === NEW === 无 XOR 兜底：若从未见过任何 XOR，且该帧已完成但未定格轮次，则置为 6
        fid = fp.frame_id
        if fid in played_frames and fid not in redundancy_rounds_final and xor_last_seen_anchor < 0:
            redundancy_rounds_final[fid] = 6

        # 若帧完成，覆写最终行
        baseline = latest_df_pid.get(fid, packet_id)
        vec, loss_rate = _compute_vec_and_loss_for_frame(fid, baseline)
        redund_rounds = redundancy_rounds_final.get(fid, None)
        _write_line_at(fid, _format_line(fid, loss_rate, redund_rounds, vec))

        return newly, packet_id

    # -------- XOR --------
    if typ == T_XOR:
        try:
            xp = XorPacket.from_bytes(data)
        except Exception:
            return newly, packet_id

        packet_id = xp.packet_id
        _note_packet(packet_id)

        items = xp.items

        # === 基于 anchor 的“冗余持续轮次”统计（与此前一致）===
        anchor = xp.anchor_frame_id
        sources_now: Set[int] = {fid for (fid, _full, _cred) in items}

        # 1) 锚点跳号：在缺失锚点处截断所有活动源帧
        if xor_last_seen_anchor >= 0 and anchor > xor_last_seen_anchor + 1:
            missing_anchor = xor_last_seen_anchor + 1
            to_finish_jump = list(xor_active_running)
            for fid in to_finish_jump:
                first_a = xor_active_first_anchor.get(fid, missing_anchor)
                rounds = max(0, missing_anchor - first_a)  # [first_a, missing_anchor-1]
                _finalize_redund_rounds(fid, rounds, baseline_pid_fallback=packet_id)
                xor_active_running.discard(fid)
                xor_active_first_anchor.pop(fid, None)
                xor_active_last_anchor.pop(fid, None)

        # 2) 已在活动集但当前 XOR 不再包含的源帧：在上个锚点处截断
        to_finish_drop = []
        for fid in list(xor_active_running):
            if fid not in sources_now:
                first_a = xor_active_first_anchor.get(fid, anchor)
                last_a  = xor_active_last_anchor.get(fid, anchor - 1)
                rounds = max(0, (last_a - first_a + 1))
                _finalize_redund_rounds(fid, rounds, baseline_pid_fallback=packet_id)
                to_finish_drop.append(fid)
        for fid in to_finish_drop:
            xor_active_running.discard(fid)
            xor_active_first_anchor.pop(fid, None)
            xor_active_last_anchor.pop(fid, None)

        # 3) 当前 XOR：新增的拉入活动、已有的延长尾巴；到达上限立即定格
        for fid in sources_now:
            if fid not in redundancy_rounds_final:  # 未定格
                if fid not in xor_active_running:
                    xor_active_first_anchor[fid] = anchor
                    xor_active_last_anchor[fid]  = anchor
                    xor_active_running.add(fid)
                else:
                    xor_active_last_anchor[fid] = anchor
                first_a = xor_active_first_anchor[fid]
                last_a  = xor_active_last_anchor[fid]
                rounds_now = max(0, (last_a - first_a + 1))
                if rounds_now >= MAX_ROUNDS:
                    _finalize_redund_rounds(fid, MAX_ROUNDS, baseline_pid_fallback=packet_id)
                    xor_active_running.discard(fid)
                    xor_active_first_anchor.pop(fid, None)
                    xor_active_last_anchor.pop(fid, None)

        xor_last_seen_anchor = max(xor_last_seen_anchor, anchor)
        _check_finalize_by_threshold(current_anchor=xor_last_seen_anchor, baseline_pid_fallback=packet_id)

        # ===== 下方保持原有 XOR credit/完成度逻辑 =====
        missing_cnt = 0
        touched: List[int] = []
        for (fid, full16, cred16) in items:
            if fid not in fec_body_bytes:
                fec_body_bytes.setdefault(fid, min(full16, max_pay_load))
                fec_body_rcvd.setdefault(fid, 0)
            if not _is_frame_complete(fid):
                missing_cnt += 1
            touched.append(fid)

        if missing_cnt > 1:
            undecoded_xor_payloads.append(data)
            return newly, packet_id

        for (fid, full16, cred16) in items:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            fec_body_rcvd[fid] = min(full, rcvd + int(cred16))

        for fid in touched:
            full = int(fec_body_bytes.get(fid))
            rcvd = int(fec_body_rcvd.get(fid))
            if full is not None and rcvd >= full:
                frame_chunks.get(fid, set()).add(len(frame_chunks.get(fid, set())))
            _try_finalize(fid, newly)

        # 若有帧完成/或轮次已定格，覆写最终行
        for fid in list(played_frames):
            redund_rounds = redundancy_rounds_final.get(fid, None)
            baseline = latest_df_pid.get(fid, packet_id)
            vec, loss_rate = _compute_vec_and_loss_for_frame(fid, baseline)
            _write_line_at(fid, _format_line(fid, loss_rate, redund_rounds, vec))

        return newly, packet_id

    # 其它类型：直接返回
    return newly, packet_id
