#!/usr/bin/env bash
# CountSense Edge - one-time setup on the Raspberry Pi 5 or Pi 4 (64-bit Raspberry Pi OS, Python 3.13).
# Installs everything OFFLINE from the wheel files copied from the laptop (~/pi_wheels),
# checks the install, then runs one real detection as proof.
#
# usage (on the Pi, in the project folder):
#     bash scripts/setup_pi.sh                  # wheels in ~/pi_wheels
#     bash scripts/setup_pi.sh /path/to/wheels  # wheels somewhere else
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS="${1:-$HOME/pi_wheels}"
VENV="$PROJECT_DIR/.venv"
PY="$VENV/bin/python"
cd "$PROJECT_DIR"

step() { echo; echo "=== $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

step "1/5 Checking the Pi"
arch="$(dpkg --print-architecture 2>/dev/null || uname -m)"
[[ "$arch" == "arm64" || "$arch" == "aarch64" ]] || fail "needs 64-bit Raspberry Pi OS (found '$arch')."
pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" \
    || fail "python3 not found."
free_mb="$(df -Pm "$PROJECT_DIR" | awk 'NR == 2 {print $4}')"
# Which board this is, and its memory page size (Pi 5 kernel: 16384, Pi 4 kernel: 4096).
board="$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || true)"
echo "board: ${board:-unknown} | page size: $(getconf PAGESIZE)"
echo "architecture: $arch | python3: $pyver | free space: ${free_mb} MB | wheels: $WHEELS"
[[ "$pyver" == "3.13" ]] || fail "the wheels are built for Python 3.13, this Pi has $pyver."
[[ -d "$WHEELS" ]] || fail "wheel folder $WHEELS not found. From the laptop: scp -r pi_wheels test@192.168.50.2:"
(( free_mb >= 2000 )) || fail "need at least 2000 MB free, have ${free_mb} MB."

step "2/5 Creating the Python environment in $VENV"
if [[ -x "$PY" ]]; then
    echo "already there, reusing it"
elif python3 -m venv "$VENV"; then
    echo "created"
else
    # python3-venv is missing (and there is no internet to apt-get it): create the
    # venv without pip, then run pip straight from its wheel file to install itself.
    echo "python3-venv is missing, using the fallback"
    rm -rf "$VENV"
    python3 -m venv --without-pip "$VENV"
    pip_wheel="$(ls "$WHEELS"/pip-*.whl 2>/dev/null | head -n 1)" || true
    [[ -n "$pip_wheel" ]] || fail "no pip-*.whl in $WHEELS for the fallback."
    "$PY" "$pip_wheel/pip" install --no-index --find-links "$WHEELS" pip
fi

step "3/5 Installing packages offline from $WHEELS"
# torch is named here, not in requirements-pi.txt: it comes from a different index (see that file).
"$PY" -m pip install --no-index --find-links "$WHEELS" torch==2.14.1+cpu -r requirements-pi.txt

step "4/5 Checking the install"
# opencv-python needs the system library libGL. If this OS doesn't have it, switch to
# opencv-python-headless: the same OpenCV without GUI windows, which we never open.
cv2_error="$("$PY" -c 'import cv2' 2>&1)" || true
if [[ "$cv2_error" == *"libGL.so.1"* ]]; then
    echo "libGL.so.1 is missing: switching to opencv-python-headless 5.0.0.93"
    "$PY" -m pip uninstall -y opencv-python
    # --force-reinstall: both packages own the same cv2 files, so the uninstall above also
    # removes them when the headless one is already installed (setup run a second time).
    "$PY" -m pip install --no-index --find-links "$WHEELS" --force-reinstall --no-deps opencv-python-headless==5.0.0.93
fi
"$PY" - <<'EOF'
import importlib.metadata as md
import platform

import cv2
import numpy
import torch
from executorch.runtime import Runtime

xnnpack = Runtime.get().backend_registry.is_available("XnnpackBackend")
print("arch:", platform.machine(), "| python:", platform.python_version())
print("torch:", torch.__version__, "| executorch:", md.version("executorch"))
print("numpy:", numpy.__version__, "| opencv:", cv2.__version__)
print("XNNPACK available:", xnnpack)
if not xnnpack:
    raise SystemExit("ERROR: the XNNPACK backend is not available")
EOF

step "5/5 Smoke test: detecting objects in data/bus.jpg (expect 5: 1 bus + 4 persons)"
"$PY" run_poc.py --source data/bus.jpg --model yolo11n_320 --out outputs/setup_check

step "Setup complete. Run with:  .venv/bin/python run_poc.py --source data/<clip>.mp4 --preset traffic"
