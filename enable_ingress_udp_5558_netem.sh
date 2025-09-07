#!/usr/bin/env bash
# 对进入服务器的所有 UDP（IPv4+IPv6）施加 netem：loss 8% 25%（突发丢）
# 用法：./enable_ingress_udp_all_netem.sh [DEV]   # DEV 省略则自动探测
set -euo pipefail

DEV="${1:-$(ip route show default 0.0.0.0/0 2>/dev/null | awk '{print $5}' | head -n1)}"
DEV="${DEV:-eth0}"
echo "[*] Using ingress device: ${DEV}"

# 准备 ifb0
sudo modprobe ifb numifbs=1 || true
ip link show ifb0 >/dev/null 2>&1 || sudo ip link add ifb0 type ifb
sudo ip link set up dev ifb0

# 幂等清理
sudo tc qdisc del dev "${DEV}" ingress 2>/dev/null || true
sudo tc qdisc del dev ifb0 root 2>/dev/null || true

# 把 ${DEV} 的 IPv4+IPv6 入站镜像到 ifb0
sudo tc qdisc add dev "${DEV}" handle ffff: ingress
# IPv4
sudo tc filter add dev "${DEV}" parent ffff: protocol ip u32 match u32 0 0 \
  action mirred egress redirect dev ifb0
# IPv6
sudo tc filter add dev "${DEV}" parent ffff: protocol ipv6 u32 match u32 0 0 \
  action mirred egress redirect dev ifb0

# ifb0：prio 根 + 在 band3 挂 netem
sudo tc qdisc add dev ifb0 root handle 1: prio
sudo tc qdisc add dev ifb0 parent 1:3 handle 30: netem loss 5%

# 只让 UDP 进入 band3（分别匹配 IPv4/IPv6）
# IPv4 UDP（17）
sudo tc filter add dev ifb0 protocol ip parent 1: prio 3 u32 \
  match ip protocol 17 0xff flowid 1:3
# IPv6 UDP（nexthdr=17）
sudo tc filter add dev ifb0 protocol ipv6 parent 1: prio 3 u32 \
  match ip6 nexthdr 17 0xff flowid 1:3

echo "[+] Enabled netem on ingress UDP (IPv4+IPv6) -> ifb0: loss 5%"
echo "[i] Verify counters:"
sudo tc -s qdisc show dev ifb0 | sed -n '1,50p'
sudo tc -s filter show dev ifb0 parent 1:
sudo tc -s filter show dev "${DEV}" parent ffff:
