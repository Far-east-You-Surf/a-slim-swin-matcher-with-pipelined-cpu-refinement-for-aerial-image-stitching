#!/usr/bin/env bash
# Rekam kondisi lingkungan satu sesi -> results/<sesi>/env.txt
# Pakai: bash scripts/env_capture.sh <sesi>   (dijalankan dari cloud_e14/)
set -u
S=${1:?nama sesi}; mkdir -p results/$S; O=results/$S/env.txt
{
  echo "# sesi $S  $(date -Is)  host $(hostname)"
  echo "## nvidia-smi"; nvidia-smi
  echo "## nvidia-smi -q -d CLOCK,POWER"; nvidia-smi -q -d CLOCK,POWER
  echo "## proses compute"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "## CUDA"; (nvcc --version 2>/dev/null || echo "nvcc tidak ada") | tail -2
  echo "## TensorRT (python)"; ${PY:-python3} -c "import tensorrt as t; print(t.__version__)" 2>&1
  echo "## trtexec"; (command -v trtexec && trtexec --version 2>&1 | head -3) || echo "trtexec tidak ada"
  echo "## python"; ${PY:-python3} --version
  echo "## OS"; uname -a; (. /etc/os-release && echo "$PRETTY_NAME")
} > $O 2>&1
echo "ditulis: $O"
