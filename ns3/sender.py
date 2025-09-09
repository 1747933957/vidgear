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

# 发送端需要维护的部分元数据（用于 XOR items）
m_frame_body_bytes: Dict[int, int] = {}
m_frame_data_pkt_cnt: Dict[int, int] = {}

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

    data_pkts: List[bytes] = []
    for i, payload in enumerate(chunks):
        pid = _next_packet_id()
        dp = DataPacket(packet_id=pid, frame_id=fid, frame_len=data_size, frame_pkt_num=n, pkt_id_in_frame=i, payload=payload)
        data_pkts.append(dp.to_bytes())

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

    return data_pkts + fec_pkts

def sendFrame_xor(data: bytes, loss_rate: float, rtt_ms: float, fec_rate: float, max_pay_load: int) -> List[bytes]:
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
    # pred_loss, pred_dur = predict_next_by_slow_module()
    # print(pred_loss, pred_dur)
    # L_cur是现在的ddl/rtt和1/冗余率上取整的最小值
    ddl = 100.0
    if fec_rate <= 0.000001:
        fec_rate = 0.000001
    if rtt_ms <= 0.000001:
        rtt_ms = 30.0
    L_cur = math.ceil(min(ddl/rtt_ms, 1/fec_rate))

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

    data_pkts: List[bytes] = []
    for i, payload in enumerate(chunks):
        pid = _next_packet_id()
        dp = DataPacket(packet_id=pid, frame_id=fid, frame_len=data_size, frame_pkt_num=n, pkt_id_in_frame=i, payload=payload)
        data_pkts.append(dp.to_bytes())

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
            
    # 9) 当前帧入队水库（从下一帧开始参与）
    _enqueue_token_plan(fid, min(payload_cap, data_size), L_cur)

    # 7) 生成 XOR 包（anchor=当前帧；合并所有源）
    xor_pkts: List[bytes] = []
    if xor_budget_to_send > 0 and xor_info.sources:
        items: List[Tuple[int,int,int]] = []
        budget = xor_budget_to_send
        # === NEW === 暂存 (src_fid, round_idx) 以便生成 pid 后建立 g_xor_pid_sources 映射
        _src_round_list: List[Tuple[int, int]] = []

        for (src_frame_id, give) in xor_info.sources:
            length32 = min(1<<31-1, m_frame_body_bytes.get(src_frame_id, 0))
            curL = budget             # 信用统一使用 τ（与 C++ 的“同一 XOR 对所有源给同等信用”一致）
            items.append((int(src_frame_id), int(length32), int(curL)))

            # === NEW === 标注“该源帧此次 XOR 的轮次”：取它当前 plan 的 k（提交前）
            round_idx = 0
            plan = g_plan_by_frame.get(src_frame_id)
            if plan is not None:
                round_idx = int(plan.k)  # 提交后才会 +1
            # 记录到源帧的“轮次对应 pid 列表”中（先占位 -1，pid 生成后回填）
            _src_round_list.append((src_frame_id, round_idx))

        pid = _next_packet_id()
        xp = XorPacket(packet_id=pid, anchor_frame_id=fid, items=items, budget=budget)
        raw = xp.to_bytes()

        tail = b'\xff' * int(budget)   # 末尾追加长度为 budget 的全 1 payload（仅用于“占位带宽”，方便对齐 NS-3）
        xor_pkts.append(raw + tail)

    # 8) 提交本轮（推进水库）
    _commit_after_send()

    return data_pkts + fec_pkts + xor_pkts