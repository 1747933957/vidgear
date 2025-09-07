#!/usr/bin/env bash
# 取消 enable_ingress_udp_all_netem.sh 设置的所有 netem 配置
# 用法：./disable_ingress_udp_all_netem.sh [DEV]   # DEV 省略则自动探测
set -euo pipefail

DEV="${1:-$(ip route show default 0.0.0.0/0 2>/dev/null | awk '{print $5}' | head -n1)}"
DEV="${DEV:-eth0}"
echo "[*] Using ingress device: ${DEV}"

# 删除 ${DEV} 上的 ingress qdisc
sudo tc qdisc del dev "${DEV}" ingress 2>/dev/null || true

# 删除 ifb0 上的 root qdisc
sudo tc qdisc del dev ifb0 root 2>/dev/null || true

# 删除 ifb0 接口（可选，如果不再需要）
if ip link show ifb0 >/dev/null 2>&1; then
  sudo ip link set dev ifb0 down
  sudo ip link delete ifb0 type ifb 2>/dev/null || true
fi

echo "[+] Disabled netem on ingress UDP (IPv4+IPv6)"
