# sender.py
# -*- coding: utf-8 -*-
"""
发送端：严格复刻 C++ 中 XOR 的“水库 + 令牌桶”更新/发送流程。
对外暴露：
    sendFrame(data: bytes, loss_rate: float, rtt_ms: int, fec_rate: float, max_pay_load: int) -> list[bytes]
约定：
- 不引入 ACK（你之前说过不要 ACK 参与调度），但本文件为“统计/记录”目的，会在 _ack_packet 中更新统计。
- FEC 仅保留“个数控制/摘要”，不做真实纠删（与原实现层次一致）。
- XOR：ComputeAppendInfo / CommitAfterSend / EnqueueTokenPlan 的语义保持不变。

本版变更（重要）：
- 引入双窗口：
  - WINDOW_VEC_N  ：最近包 0/1 向量长度（可视化/模型 recentN）
  - WINDOW_LOSS_N ：计算“发送时刻丢包率”的窗口长度（基于全局已发送包最近 N）
- 每帧 loss_rate_at_send 的计算：以该帧 DATA/FEC 的最大 packet_id 为“基线包号”，
  从全局已发送包中截取“<= 基线”的最近 WINDOW_LOSS_N 个包，按 0/1 估算丢包率。
- === NEW === 基于 ACK 推断 redundancy_rounds（你的规则）：
  - 若首次丢包发生在轮次 r_loss，则 redundancy_rounds = r_loss - r_send - 1，其中 r_send 定义为 0
  - 若前 L 轮全部收到 ACK，则 redundancy_rounds = L
  - 只有确定“首次丢包”或“前 L 轮全 ACK”才定格；否则保持 None/NA
- === NEW === sender 侧生成 recv_stats.txt，且“ACK 到达后实时覆写受影响行”，让丢包率和 recentN_vector 随 ACK 收敛
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math
from pathlib import Path
import json

from .packets import DataPacket, FecPacket, XorPacket
from .slow_module import SlowModule
# sm = SlowModule.load(Path("slow_module.pt"))

# ========= 双窗口常量 =========
WINDOW_VEC_N: int = 5   # recentN 向量长度（用于打印/模型输入）
WINDOW_LOSS_N: int = 5  # 丢包率窗口长度（用于 per-frame loss 估计）

# === 已发送包追踪：{packet_id: {"frame_id": int, "type": "data"/"fec"/"xor", "acked": bool}} ===
g_sent_packets: Dict[int, Dict] = {}

# === 每帧元数据：data/fec/xor 的包列表 + 丢包率 + 冗余 L + 完成标记 + （NEW）ACK 推断字段 ===
g_frame_meta: Dict[int, Dict] = {}

# =========================
# sender 侧统计输出（与 receiver.py 口径一致）
# =========================
SENDER_STATS_TXT_PATH = Path("recv_stats.txt")  # 注意：若 sender 与 receiver 同时写同目录，可能互相覆盖

# 仅首次初始化时写表头，避免每次写清空历史
_SENDER_STATS_INITIALIZED = False
def _ensure_sender_stats_file():
    global _SENDER_STATS_INITIALIZED
    if not _SENDER_STATS_INITIALIZED:
        SENDER_STATS_TXT_PATH.write_text(
            "frame_id\tloss_rate_at_send\tredundancy_rounds\trecentN_vector\n",
            encoding="utf-8",
        )
        _SENDER_STATS_INITIALIZED = True
    elif not SENDER_STATS_TXT_PATH.exists():
        SENDER_STATS_TXT_PATH.write_text(
            "frame_id\tloss_rate_at_send\tredundancy_rounds\trecentN_vector\n",
            encoding="utf-8",
        )
_ensure_sender_stats_file()

def _sender_read_all_lines():
    _ensure_sender_stats_file()
    with SENDER_STATS_TXT_PATH.open("r", encoding="utf-8") as f:
        return f.readlines()

def _sender_write_all_lines(lines):
    with SENDER_STATS_TXT_PATH.open("w", encoding="utf-8") as f:
        f.writelines(lines)

def _sender_write_line_at(fid: int, line_text: str):
    """
    将 fid 对应的数据写到“第 fid+2 行”（1-based）；支持覆写/扩展。
    与 receiver.py 的写入策略保持一致，但不再用纯空行占位——
    会为中间缺失的帧做“回填”（按当前 ACK 状态计算或给出默认值），避免出现空行。
    """
    lines = _sender_read_all_lines()
    need = fid + 2
    if len(lines) < need:
        # 回填每一行
        cur_len = len(lines)
        # 从第 cur_len 到 need-1（注意文件是 1-based：行号 = 帧号+1；0 行是表头）
        for idx in range(cur_len, need):
            pfid = idx - 1  # 还没写过的帧 id
            if pfid < 0:
                # 第 0 行是表头，保持不变
                lines.append("frame_id\tloss_rate_at_send\tredundancy_rounds\trecentN_vector\n")
                continue

            # 有元数据则按当前 ACK 状态重算；没有就给一个合理默认
            meta = g_frame_meta.get(pfid, None)
            if meta is not None:
                baseline = _baseline_pid_for_frame(meta)
                if baseline is None and g_sent_packets:
                    baseline = max(g_sent_packets.keys())
                if baseline is not None:
                    vec, loss_rate = _compute_vec_and_loss_for_frame_sender(pfid, baseline)
                else:
                    vec, loss_rate = ([1] * WINDOW_VEC_N, 0.0)
                redund_rounds = meta.get("redundancy_rounds", None)
                lines.append(_sender_format_line(pfid, loss_rate, redund_rounds, vec) + "\n")
            else:
                # 保守默认：loss_rate=0.0，recentN 全 1，redundancy_rounds=NA
                lines.append(_sender_format_line(pfid, 0.0, None, [1] * WINDOW_VEC_N) + "\n")

    # 写/覆写目标行
    lines[fid + 1] = line_text.rstrip("\n") + "\n"
    _sender_write_all_lines(lines)


def _sender_format_line(fid: int, loss_rate: float | None, redund_rounds: int | None, vec: list[int]) -> str:
    """
    - loss_rate: None -> "NA"
    - redund_rounds: None -> "NA"
    - vec: JSON 序列化的 0/1 列表
    """
    lr_str = f"{loss_rate:.6f}" if loss_rate is not None else "NA"
    rr_str = str(redund_rounds) if redund_rounds is not None else "NA"
    return f"{fid}\t{lr_str}\t{rr_str}\t{json.dumps(vec, ensure_ascii=False)}"

# =========================
# 全局发送侧状态
# =========================
_frame_seq = 0
def _next_frame_id() -> int:
    """生成自增 frame_id。"""
    global _frame_seq
    fid = _frame_seq
    _frame_seq += 1
    return fid

# === 全局包ID计数器 ===
_pkt_seq = 0
def _next_packet_id() -> int:
    """生成自增 packet_id。"""
    global _pkt_seq
    pid = _pkt_seq
    _pkt_seq += 1
    return pid

# 令牌桶：字节
g_token_bucket_bytes: int = 0

@dataclass
class RedundancyPlan:
    """
    水库计划（对应 C++：进入水库后跨 L 轮释放完整重传预算）
    - frame_bytes：完整重传预算，一般取 MIN(max_payload, data_size)
    - schedule：均分前 L-1 轮，最后一轮兜底
    - k：当前运行到的轮次（从 0 开始计数）
    """
    frame_id: int
    frame_bytes: int
    L: int
    k: int = 0
    budget_total: int = 0
    reservoir_left: int = 0
    schedule: List[int] = field(default_factory=list)
    release_this_round: int = 0

def _build_schedule(budget_total: int, L: int) -> List[int]:
    """
    均匀分配，最后一轮收尾（与 C++ BuildSchedule(Uniform) 同语义）。
    """
    if budget_total <= 0 or L <= 0:
        return []
    eff_L = max(1, L)
    avg = budget_total // eff_L
    MIN_BYTES = 50
    sched = []
    given = 0
    for k in range(eff_L):
        alloc = max(avg, MIN_BYTES)
        if k == eff_L - 1:
            alloc = max(0, budget_total - given)
        sched.append(alloc)
        given += alloc
    return sched

# 活跃水库
g_red_plans: List[RedundancyPlan] = []

# === NEW === frame_id -> RedundancyPlan 的映射（用于标注“该帧在某次 XOR 属于第几轮(k)”）
g_plan_by_frame: Dict[int, RedundancyPlan] = {}

@dataclass
class RoundXorInfo:
    """
    本轮 XOR 汇总：
      - xor_bytes: τ（本轮 XOR 预算，字节）
      - sources  : [(frameId, giveBytes)] —— 每个源帧扣账值（用于统计/可视化）
    """
    xor_bytes: int = 0
    sources: List[Tuple[int,int]] = field(default_factory=list)

def _compute_append_info(allowed_tokens: int) -> RoundXorInfo:
    """
    ComputeAppendInfo 一致性版：
      1) candidate[i] = 计划释放（若 k<L 用 schedule[k]，否则用 reservoir_left）
      2) tau = min(max(candidate), allowed_tokens)
      3) 每个活跃帧扣账 min(tau, reservoir_left) 作为本轮分配
    """
    info = RoundXorInfo()
    tau_planned = 0

    for p in g_red_plans:
        p.release_this_round = 0
        if p.reservoir_left <= 0:
            continue
        if p.k < p.L and p.k < len(p.schedule):
            planned = min(p.schedule[p.k], p.reservoir_left)
        else:
            planned = p.reservoir_left   # 超过计划 L：直接倾倒剩余
        tau_planned = max(tau_planned, planned)

    if tau_planned == 0:
        return info

    tau = min(int(min((1<<31)-1, allowed_tokens)), tau_planned) if allowed_tokens > 0 else 0
    info.xor_bytes = tau

    if tau > 0:
        for p in g_red_plans:
            if p.reservoir_left <= 0:
                continue
            deduct = min(tau, p.reservoir_left)
            p.release_this_round = deduct
            if deduct > 0:
                info.sources.append((p.frame_id, deduct))

    return info

def _commit_after_send():
    """
    按 release_this_round 扣减 reservoir，并推进 k；清理空计划。
    """
    i = 0
    while i < len(g_red_plans):
        p = g_red_plans[i]
        if p.release_this_round > 0:
            p.reservoir_left = max(0, p.reservoir_left - p.release_this_round)
        p.k += 1
        p.release_this_round = 0
        if p.reservoir_left == 0:
            g_red_plans.pop(i)
            # === NEW === 同步移除映射
            g_plan_by_frame.pop(p.frame_id, None)
        else:
            i += 1

def _enqueue_token_plan(frame_id: int, frame_bytes: int, L: int):
    """
    新帧进入水库（完整重传预算 = frame_bytes，一般取 MIN(max_payload, data_size)）
    """
    if frame_bytes <= 0 or L <= 0:
        return
    plan = RedundancyPlan(
        frame_id=frame_id,
        frame_bytes=frame_bytes,
        L=L,
        k=0,
        budget_total=frame_bytes,
        reservoir_left=frame_bytes,
        schedule=_build_schedule(frame_bytes, L),
    )
    g_red_plans.append(plan)
    # === NEW === 建立 frame_id -> 计划 的映射，便于标注“该帧在本次 XOR 属于第几轮”
    g_plan_by_frame[frame_id] = plan

# 发送端需要维护的部分元数据（用于 XOR items）
m_frame_body_bytes: Dict[int, int] = {}
m_frame_data_pkt_cnt: Dict[int, int] = {}

# ========= 工具函数 =========
def _recent_vec_global_up_to(baseline_pid: int, N: int) -> List[int]:
    """
    取“全局包序列中 <= baseline_pid 的最近 N 个包”，生成 0/1 向量：
      - 1 表示该包已 ACK（或等价到达）
      - 0 表示未 ACK
    若不足 N 个，则在左端补 1（视作到达）以稳定长度。
    """
    if not g_sent_packets:
        return [1] * N
    # 全局已知包按 pid 排序，并截到 baseline_pid（含）
    pids_sorted = [pid for pid in sorted(g_sent_packets.keys()) if pid <= baseline_pid]
    window = pids_sorted[-N:]
    vec = [1 if g_sent_packets[pid].get("acked", False) else 0 for pid in window]
    if len(vec) < N:
        vec = [1] * (N - len(vec)) + vec
    return vec

def _baseline_pid_for_frame(meta: Dict) -> int | None:
    """
    计算某帧的“基线包号”：该帧所有 DATA/FEC 包的最大 packet_id。
    若该帧目前没有任何 DATA/FEC（极少见），返回 None。
    """
    ids = []
    ids.extend(meta.get("data_pkt_ids", []))
    ids.extend(meta.get("fec_pkt_ids", []))
    if not ids:
        return None
    return max(ids)

# === NEW === 通过 ACK 推断 redundancy_rounds 所需：pid<->(src_fid, round_idx)
g_xor_pid_sources: Dict[int, List[Tuple[int, int]]] = {}  # xor_pid -> [(src_fid, round_idx)]

def _maybe_update_redundancy_rounds_from_ack(xor_pid: int):
    """
    在收到某个 XOR 包的 ACK 时调用：
    - 找到该 XOR 覆盖的所有源帧 (src_fid) 以及“这是该帧的第 r 轮”(round_idx)
    - 对每个源帧，检查其更早轮次是否存在未 ACK 的 XOR 包：
        * 若存在：first_loss_round = 最早未 ACK 的轮次
          redundancy_rounds = first_loss_round - send_round - 1   （send_round 定义为 0）
        * 若不存在且“前 L 轮均已 ACK”：redundancy_rounds = L
    只有当能“确认首个丢包”或“确认无丢包（前 L 轮全 ACK）”时才写入。
    返回：set[int] —— 本次被定格/更新 redundancy_rounds 的 frame_id 集合
    """
    updated = set()
    if xor_pid not in g_xor_pid_sources:
        return updated
    for (src_fid, r_this) in g_xor_pid_sources[xor_pid]:
        meta = g_frame_meta.get(src_fid)
        if not meta:
            continue
        if meta.get("redundancy_rounds", None) is not None:
            continue  # 已定格过就不再改

        rounds_pids: List[int] = meta.get("xor_round_pids", [])
        if not rounds_pids:
            continue
        L = int(meta.get("redundancy_L", 0))
        send_round = int(meta.get("send_round", 0))  # 基准取 0

        # 1) 在 [0, r_this) 范围内寻找“未 ACK”的轮次（这次 ACK 证明前面没来的就算丢）
        first_loss_round = None
        for k in range(0, min(r_this, len(rounds_pids))):
            pid_k = rounds_pids[k]
            if not g_sent_packets.get(pid_k, {}).get("acked", False):
                first_loss_round = k
                break
        if first_loss_round is not None:
            meta["redundancy_rounds"] = max(0, first_loss_round - send_round - 1)
            updated.add(src_fid)
            continue  # 对该帧已定格

        # 2) 如前 L 轮全部存在且全部 ACK，则认为“无丢包”，定格为 L
        if L > 0 and len(rounds_pids) >= L:
            first_L = rounds_pids[:L]
            if all(g_sent_packets.get(pid_i, {}).get("acked", False) for pid_i in first_L):
                meta["redundancy_rounds"] = L
                updated.add(src_fid)

    return updated

# === NEW === 与 receiver 口径一致的窗口计算，用于 sender 侧写文件
def _compute_vec_and_loss_for_frame_sender(fid: int, baseline_pid: int) -> tuple[list[int], float]:
    """
    与 receiver 的窗口口径对齐：
    - recentN_vector 长度 = WINDOW_VEC_N
    - loss_rate_at_send 使用 <= baseline_pid 的最近 WINDOW_LOSS_N 个包
    直接复用 sender 侧全局发包表 g_sent_packets。
    """
    # 向量（长度 WINDOW_VEC_N）
    vec = _recent_vec_global_up_to(baseline_pid, WINDOW_VEC_N)
    if len(vec) > WINDOW_VEC_N:
        vec = vec[-WINDOW_VEC_N:]
    elif len(vec) < WINDOW_VEC_N:
        vec = [1] * (WINDOW_VEC_N - len(vec)) + vec

    # 丢包率（长度 WINDOW_LOSS_N）
    loss_vec = _recent_vec_global_up_to(baseline_pid, WINDOW_LOSS_N)
    if len(loss_vec) > WINDOW_LOSS_N:
        loss_vec = loss_vec[-WINDOW_LOSS_N:]
    elif len(loss_vec) < WINDOW_LOSS_N:
        loss_vec = [1] * (WINDOW_LOSS_N - len(loss_vec)) + loss_vec
    loss_rate = (loss_vec.count(0) / WINDOW_LOSS_N) if WINDOW_LOSS_N > 0 else 0.0

    return vec, loss_rate

# === NEW === 以当前 ACK 状态，重算并覆写某帧的一行
def _sender_rewrite_line_for(fid: int):
    meta = g_frame_meta.get(fid, {})
    baseline = _baseline_pid_for_frame(meta)
    if baseline is None:
        # 退化处理：没有 DATA/FEC，尽量用当前最大 pid
        if g_sent_packets:
            baseline = max(g_sent_packets.keys())
        else:
            vec, loss_rate = ([1] * WINDOW_VEC_N, 0.0)
            line = _sender_format_line(fid, loss_rate, meta.get("redundancy_rounds", None), vec)
            _sender_write_line_at(fid, line)
            return

    vec, loss_rate = _compute_vec_and_loss_for_frame_sender(fid, baseline)
    redund_rounds = meta.get("redundancy_rounds", None)
    line = _sender_format_line(fid, loss_rate, redund_rounds, vec)
    _sender_write_line_at(fid, line)

# === NEW === 给定一个 ack 的 pid，找出所有受影响的帧并覆写（窗口覆盖该 pid 的帧）
def _sender_update_lines_for_ack(acked_pid: int):
    if not g_frame_meta:
        return
    Nmax = max(WINDOW_VEC_N, WINDOW_LOSS_N)
    for fid, meta in g_frame_meta.items():
        baseline = _baseline_pid_for_frame(meta)
        if baseline is None:
            continue
        left = max(0, baseline - Nmax + 1)  # 窗口左端（含）
        if left <= acked_pid <= baseline:
            _sender_rewrite_line_for(fid)

def sendFrame(data: bytes, loss_rate: float, rtt_ms: int, fec_rate: float, max_pay_load: int) -> List[bytes]:
    """
    输入一帧原始 bytes，输出需要“上线传输”的 bytes 包列表（Data/FEC/XOR）。
    - Data：按 max_pay_load 切片（至少一片）
    - FEC ：floor(n * fec_rate) 个摘要包（不做真实纠删）
    - XOR ：严格按“水库 + 令牌桶”策略决定是否在本帧内发送一个 XOR 包（预算 τ）
    """
    assert max_pay_load > 0
    global g_token_bucket_bytes

    data_size = len(data)
    payload_cap = max_pay_load

    # === 在线模型预测（保持原有调用） ===
    pred_loss, pred_dur = predict_next_by_slow_module()
    print(pred_loss, pred_dur)
    L_cur = 5  # 这里保持现状；若要接入 pred_dur，可解注释并做限幅

    # 1) 计算 data 分片数量
    n = data_size // payload_cap + int(data_size % payload_cap != 0)
    if n == 0:
        n = 1  # 保证至少 1 片（允许空帧）

    # 2) 令牌桶进账：当 fec_rate * n 存在小数时，存入 deposit = MIN(max_payload, data_size)
    real_fec_num = fec_rate * float(n)
    frac = abs(real_fec_num - math.floor(real_fec_num))
    if frac > 1e-9:
        deposit = min(payload_cap, data_size)
        g_token_bucket_bytes += deposit

    # 3) 计算本轮 XOR 预算/分摊（用当前桶额度）
    xor_info = _compute_append_info(g_token_bucket_bytes)
    xor_budget_to_send = xor_info.xor_bytes

    # 4) 令牌桶出账（与本轮实际发送一致）
    if xor_budget_to_send > 0:
        use = min(xor_budget_to_send, g_token_bucket_bytes)
        g_token_bucket_bytes -= use

    # 5) 生成 Data 分片
    fid = _next_frame_id()
    m_frame_body_bytes[fid] = data_size
    chunks: List[bytes] = [data[i:i+payload_cap] for i in range(0, data_size, payload_cap)]
    if not chunks:
        chunks = [b""]
    n = len(chunks)
    m_frame_data_pkt_cnt[fid] = n

    data_pkts: List[bytes] = []
    for i, payload in enumerate(chunks):
        pid = _next_packet_id()
        dp = DataPacket(packet_id=pid, frame_id=fid, frame_len=data_size, frame_pkt_num=n, pkt_id_in_frame=i, payload=payload)
        data_pkts.append(dp.to_bytes())
        g_sent_packets[pid] = {"frame_id": fid, "type": "data", "acked": False}
        g_frame_meta.setdefault(fid, {"data_pkt_ids": [], "fec_pkt_ids": [], "xor_pkt_ids": [],
                                      "loss_rate_at_send": None, "redundancy_L": L_cur, "redundancy_done": False})
        g_frame_meta[fid]["data_pkt_ids"].append(pid)
        # === NEW === 初始化 ACK 推断所需字段
        g_frame_meta[fid].setdefault("xor_round_pids", [])  # 保存该帧作为 source 被包含的每一轮 XOR 的 pid
        g_frame_meta[fid].setdefault("send_round", 0)       # 发送基准轮次，定义为 0

    # 6) 生成 FEC（摘要 XOR，不做真实纠删）
    fec_num = int(math.floor(n * fec_rate))
    fec_pkts: List[bytes] = []
    if fec_num > 0:
        max_len = max((len(c) for c in chunks), default=0)
        def xor_many(bufs):
            if not bufs:
                return b""
            padded = [b + b"\x00"*(max_len-len(b)) for b in bufs]
            out = bytearray(padded[0])
            for b in padded[1:]:
                for i in range(max_len):
                    out[i] ^= b[i]
            return bytes(out)
        parity = xor_many(chunks)
        cover_ids = list(range(n))
        for _ in range(fec_num):
            pid = _next_packet_id()
            fp = FecPacket(packet_id=pid, frame_id=fid, covered_ids=cover_ids, parity=parity)
            fec_pkts.append(fp.to_bytes())
            g_sent_packets[pid] = {"frame_id": fid, "type": "fec", "acked": False}
            g_frame_meta[fid]["fec_pkt_ids"].append(pid)
            
    # # 9) 当前帧入队水库（从下一帧开始参与）
    # _enqueue_token_plan(fid, min(payload_cap, data_size), L_cur)

    # # 7) 生成 XOR 包（anchor=当前帧；合并所有源）
    # xor_pkts: List[bytes] = []
    # if xor_budget_to_send > 0 and xor_info.sources:
    #     items: List[Tuple[int,int,int]] = []
    #     budget = xor_budget_to_send
    #     # === NEW === 暂存 (src_fid, round_idx) 以便生成 pid 后建立 g_xor_pid_sources 映射
    #     _src_round_list: List[Tuple[int, int]] = []

    #     for (src_frame_id, give) in xor_info.sources:
    #         length32 = min(1<<31-1, m_frame_body_bytes.get(src_frame_id, 0))
    #         curL = budget             # 信用统一使用 τ（与 C++ 的“同一 XOR 对所有源给同等信用”一致）
    #         items.append((int(src_frame_id), int(length32), int(curL)))

    #         # === NEW === 标注“该源帧此次 XOR 的轮次”：取它当前 plan 的 k（提交前）
    #         round_idx = 0
    #         plan = g_plan_by_frame.get(src_frame_id)
    #         if plan is not None:
    #             round_idx = int(plan.k)  # 提交后才会 +1
    #         # 记录到源帧的“轮次对应 pid 列表”中（先占位 -1，pid 生成后回填）
    #         g_frame_meta.setdefault(src_frame_id, {}).setdefault("xor_round_pids", [])
    #         g_frame_meta[src_frame_id]["xor_round_pids"].append(-1)
    #         _src_round_list.append((src_frame_id, round_idx))

    #     pid = _next_packet_id()
    #     xp = XorPacket(packet_id=pid, anchor_frame_id=fid, items=items, budget=budget)
    #     raw = xp.to_bytes()
    #     g_sent_packets[pid] = {"frame_id": fid, "type": "xor", "acked": False}
    #     g_frame_meta[fid]["xor_pkt_ids"].append(pid)

    #     # === NEW === 回填：把占位 -1 改成真实 pid，并登记 pid -> (src_fid, round_idx)
    #     g_xor_pid_sources[pid] = []
    #     for (src_frame_id, round_idx) in _src_round_list:
    #         lst = g_frame_meta[src_frame_id]["xor_round_pids"]
    #         lst[-1] = pid
    #         g_xor_pid_sources[pid].append((src_frame_id, round_idx))

    #     tail = b'\xff' * int(budget)   # 末尾追加长度为 budget 的全 1 payload（仅用于“占位带宽”，方便对齐 NS-3）
    #     xor_pkts.append(raw + tail)

    # # 8) 提交本轮（推进水库）
    # _commit_after_send()

    # === 在返回前写入 sender 侧的 recv_stats.txt（发送时刻视角；后续 ACK 会覆写收敛） ===
    try:
        # 以“该帧 DATA/FEC 的最大 packet_id”为基线
        meta = g_frame_meta.get(fid, {})
        baseline = _baseline_pid_for_frame(meta)
        if baseline is not None:
            vec, loss_rate = _compute_vec_and_loss_for_frame_sender(fid, baseline)
        else:
            # 极少见：该帧没有 DATA/FEC，退化为“用当前全局最大 pid”估计
            if g_sent_packets:
                baseline = max(g_sent_packets.keys())
                vec, loss_rate = _compute_vec_and_loss_for_frame_sender(fid, baseline)
            else:
                vec, loss_rate = ([1] * WINDOW_VEC_N, 0.0)

        # 写“基于 ACK 的 redundancy_rounds”，未定格则 NA
        redund_rounds = g_frame_meta.get(fid, {}).get("redundancy_rounds", None)

        line = _sender_format_line(fid, loss_rate, redund_rounds, vec)
        _sender_write_line_at(fid, line)
    except Exception as e:
        # 失败不影响原发送流程
        print(f"[sender stats warn] write failed for frame {fid}: {e}")

    return data_pkts + fec_pkts
    # return data_pkts + fec_pkts + xor_pkts


def _ack_packet(packet_id: int):
    """
    在收到 ACK 时调用，更新指定 packet_id 的到达状态，并实时覆写受影响的统计行：
      - 所有窗口覆盖到该 pid 的帧：重算 loss_rate/recentN 并覆写；
      - 若 ACK 到的是 XOR 包：尝试基于 ACK 定格 redundancy_rounds，并覆写相关帧。
    """
    if packet_id in g_sent_packets:
        g_sent_packets[packet_id]["acked"] = True

        # 若 ACK 的是 XOR，先做基于 ACK 的轮次推断，并把被定格的帧行覆写
        if g_sent_packets[packet_id].get("type") == "xor":
            updated_fids = _maybe_update_redundancy_rounds_from_ack(packet_id)
            for fid in updated_fids:
                _sender_rewrite_line_for(fid)

        # 无论 ack 的是什么包，所有“窗口覆盖到该 pid 的帧”都要重算一行（loss_rate 和 recentN 都可能变化）
        _sender_update_lines_for_ack(packet_id)


def _report_stats():
    """
    统计并打印：
    1) 每帧发送时的丢包率（窗口长度 = WINDOW_LOSS_N；以该帧 DATA/FEC 最大 pid 为基线）
    2) 每帧冗余包的持续轮次（等冗余全部发送完才能确定）
    3) 最近 WINDOW_VEC_N 个包的接收/丢失 0-1 向量
    """
    # 1) 丢包率（按“最近 WINDOW_LOSS_N 个全局包”口径）
    for fid, meta in g_frame_meta.items():
        if meta["loss_rate_at_send"] is None:
            baseline = _baseline_pid_for_frame(meta)
            if baseline is None:
                continue
            loss_vec = _recent_vec_global_up_to(baseline, WINDOW_LOSS_N)
            loss_rate = loss_vec.count(0) / float(WINDOW_LOSS_N) if WINDOW_LOSS_N > 0 else 0.0
            meta["loss_rate_at_send"] = float(loss_rate)
            print(f"[Frame {fid}] loss_rate_at_send(win={WINDOW_LOSS_N})={loss_rate:.3f}")

    # 2) 冗余包持续轮次（这里用“是否所有 fec/xor 包已发送完/可视为结束”的近似逻辑）
    for fid, meta in g_frame_meta.items():
        if not meta["redundancy_done"]:
            # 简化：若所有 FEC/XOR 包都已经“发送出列”（发送端本地视角），就认为持续轮次已定格
            all_sent = True  # 发送端视角恒为 True
            if all_sent:
                meta["redundancy_done"] = True
                print(f"[Frame {fid}] redundancy_L={meta['redundancy_L']}")

    # 3) 最近 WINDOW_VEC_N 个包的 0/1 向量（用于可视化/模型 recentN）
    if g_sent_packets:
        recent_pids = sorted(g_sent_packets.keys())[-WINDOW_VEC_N:]
        vec = [1 if g_sent_packets[pid]["acked"] else 0 for pid in recent_pids]
        if len(vec) < WINDOW_VEC_N:
            vec = [1] * (WINDOW_VEC_N - len(vec)) + vec
    else:
        vec = [1] * WINDOW_VEC_N
    print(f"Recent {WINDOW_VEC_N} packets status vector: {vec}")


def predict_next_by_slow_module(H: int = 5):
    """
    使用 SlowModule 进行下一帧的预测。
    变更：
      - recentN 向量长度改为 WINDOW_VEC_N（不再由形参 N 控制）
      - rate_hist 的每帧丢包率来自 meta['loss_rate_at_send']（该值由 WINDOW_LOSS_N 的窗口计算）
    其他保持一致：
      - rate_hist 长度 H，不足左端补 0
      - dur_hist 使用已完成帧的 redundancy_L，不足左端补 5
    返回:
      - pred_loss (float): 预测下一轮丢包率（0~1）
      - pred_dur  (int)  : 预测下一帧冗余包持续轮次（四舍五入到非负整数）
    """
    # === 1) 最近 WINDOW_VEC_N 个包的 0/1 向量（不足左端补 1） ===
    if g_sent_packets:
        recent_pids = sorted(g_sent_packets.keys())[-WINDOW_VEC_N:]
        recent_vec = [1 if g_sent_packets[pid]["acked"] else 0 for pid in recent_pids]
        if len(recent_vec) < WINDOW_VEC_N:
            recent_vec = [1] * (WINDOW_VEC_N - len(recent_vec)) + recent_vec
    else:
        recent_vec = [1] * WINDOW_VEC_N

    # === 2) 历史丢包率 rate_hist（长度 H，不足左端补 0） ===
    frame_ids_sorted = sorted(g_frame_meta.keys())
    last_H_fids_for_rate = frame_ids_sorted[-H:] if len(frame_ids_sorted) >= H else frame_ids_sorted
    rate_hist = []
    for fid in last_H_fids_for_rate:
        lr = g_frame_meta[fid].get("loss_rate_at_send", None)
        if lr is None:
            data_ids = g_frame_meta[fid].get("data_pkt_ids", [])
            if data_ids:
                recv = sum(1 for pid in data_ids if g_sent_packets.get(pid, {}).get("acked", False))
                lr = 1.0 - (recv / len(data_ids))
            else:
                lr = 0.0
        rate_hist.append(float(max(0.0, min(1.0, lr))))
    if len(rate_hist) < H:
        rate_hist = [0.0] * (H - len(rate_hist)) + rate_hist
    rate_hist = rate_hist[-H:]

    # === 3) 历史冗余持续轮次 dur_hist（仅采用已完成帧；不足左端补 5） ===
    dur_done_vals = []
    for fid in frame_ids_sorted:
        meta = g_frame_meta[fid]
        if meta.get("redundancy_done", False):
            dur_val = int(meta.get("redundancy_L", 0))
            dur_done_vals.append(max(0, dur_val))
    dur_hist = dur_done_vals[-H:]
    if len(dur_hist) < H:
        dur_hist = [5] * (H - len(dur_hist)) + dur_hist
    dur_hist = dur_hist[-H:]

    # === 4) 调用 SlowModule 预测 ===
    #pred_loss, pred_dur = sm.predict(rate_hist, dur_hist, recent_vec)
    return 0, 0
    return pred_loss, pred_dur
