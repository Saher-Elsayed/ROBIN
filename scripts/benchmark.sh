#!/usr/bin/env bash
# Sweep all baselines on one design.
set -euo pipefail

DESIGN=${1:-GEMM-systolic}
DEVICE=${2:-VE2302}
RTL=${3:-./benchmarks/${DESIGN}/src}
XDC=${4:-./benchmarks/${DESIGN}/${DESIGN}.xdc}

OUT_DIR="runs/${DESIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

for METHOD in Default PerfExplore_or_HighEffort RandomSearch TuRBO DRiLLS-style ROBIN-FPGA; do
    echo "===== ${METHOD} on ${DESIGN} ====="
    python scripts/train.py \
        --design "${DESIGN}" \
        --rtl-path "${RTL}" \
        --xdc "${XDC}" \
        --device "${DEVICE}" \
        --episodes 1200 \
        --seeds 5 \
        --corners 4 \
        --out "${OUT_DIR}/${METHOD}"
done

python scripts/evaluate.py --archive "${OUT_DIR}/aggregated.zip"
