#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RAY_PORT="${RAY_PORT:-6379}"

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "hostfile not found: ${HOSTFILE}" >&2
    exit 1
fi

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
if (( ${#HOSTS[@]} == 0 )); then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

HEAD_IP="${HOSTS[0]}"
for index in "${!HOSTS[@]}"; do
    ip="${HOSTS[$index]}"
    echo "Starting Ray on ${ip}..."
    if (( index == 0 )); then
        ssh -o StrictHostKeyChecking=no "root@${ip}" \
            "ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --dashboard-host=0.0.0.0"
    else
        ssh -o StrictHostKeyChecking=no "root@${ip}" \
            "ray stop --force >/dev/null 2>&1 || true; ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${ip}"
    fi
done

echo "Ray cluster started. Jobs API: http://${HEAD_IP}:8265"
