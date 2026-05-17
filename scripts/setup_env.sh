#!/usr/bin/env bash
# Initialise the Vivado/Quartus environment for ROBIN runs.
set -euo pipefail

if [[ -z "${VIVADO_DIR:-}" ]]; then
    echo "Set VIVADO_DIR=/opt/Xilinx/Vivado/2024.2 (or similar)"
    exit 1
fi
if [[ ! -f "${VIVADO_DIR}/settings64.sh" ]]; then
    echo "Vivado settings file not found at ${VIVADO_DIR}/settings64.sh"
    exit 1
fi
# shellcheck disable=SC1091
source "${VIVADO_DIR}/settings64.sh"

if [[ -n "${QUARTUS_ROOTDIR:-}" ]]; then
    export PATH="${QUARTUS_ROOTDIR}/bin:${PATH}"
fi

vivado -version | head -1
if command -v quartus_sh >/dev/null; then
    quartus_sh --version | head -1
fi

echo "ROBIN environment initialised."
