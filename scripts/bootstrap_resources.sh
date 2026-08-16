#!/usr/bin/env bash
# Clone official models used by this repo. Run from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

clone() {
  local url="$1" dir="$2"
  if [ -d "${TP}/${dir}/.git" ]; then
    echo "[skip] ${dir}"
  else
    echo "[clone] ${dir}"
    git clone --depth 1 "${url}" "${TP}/${dir}"
  fi
}

clone https://github.com/wuji-technology/wuji-description.git wuji-description
clone https://github.com/engineai-robotics/engineai_robotics_native_sdk.git engineai-native-sdk

echo "optional (not required for unit tests):"
echo "  git clone --depth 1 https://github.com/wuji-technology/mujoco-sim.git ${TP}/wuji-mujoco-sim"
echo "  git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git ${TP}/GR00T-WholeBodyControl"
