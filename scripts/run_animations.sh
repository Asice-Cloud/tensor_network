#!/usr/bin/env zsh
# Run particle_in_well presets with embedded parameters
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT_DIR/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "Python executable not found in $VENV. Make sure .venv exists and is created."
  exit 1
fi

cd "$ROOT_DIR"

# ----- Parameters (edit here) -----
# box width
a=1.0
# spatial resolution (number of x points)
Nx=1200
# number of eigenmodes to project onto
N=400
# presets to generate, choose from: 2x piecewise gauss eigen1 eigen2 superpos12
PRESETS=(2x piecewise gauss eigen1 eigen2 superpos12)
# animation framing mode: either provide explicit frame list via TIMES or use FRAMES/TSTART/TEND
# if TIMES is non-empty it will be preferred (list of time points)
TIMES=()
# or use frames/tstart/tend/fps for smoother animations
FRAMES=400
TSTART=0.0
TEND=5.0
FPS=12
# output directory
OUT_DIR="sim"
# whether to animate (true/false)
ANIMATE=true
# smoke mode: if true the script will run a quick lightweight job (overrides Nx,N,FRAMES)
SMOKE=false
# ----------------------------------

# Simple CLI override: allow USER to pass environment variables to change behavior, e.g.
# SMOKE=true ./scripts/run_animations.sh

mkdir -p "$OUT_DIR"

echo "Using python: $PY"
echo "Generating animations for presets: ${PRESETS[*]}"

# Apply smoke overrides if requested
if [ "$SMOKE" = true ]; then
  echo "SMOKE mode: overriding Nx -> 200, N -> 50, FRAMES -> 20 for quick runs"
  Nx=200
  N=50
  FRAMES=20
  FPS=10
  TEND=1.0
fi

for preset in "${PRESETS[@]}"; do
  # choose output extension: default MP4 for animated runs to avoid huge GIFs
  if [ "$ANIMATE" = true ]; then
    out_file="$OUT_DIR/pw_${preset}_long.mp4"
  else
    out_file="$OUT_DIR/pw_${preset}.png"
  fi

  echo "\n--- Running preset: $preset -> $out_file ---"

  # build base command
  cmd=("$PY" "sim/particle_in_well.py" "--init" "$preset" "--a" "$a" "--Nx" "$Nx" "--N" "$N")

  # timing args: prefer explicit TIMES if provided
  if [ ${#TIMES[@]} -gt 0 ]; then
    cmd+=("--times" "${TIMES[@]}")
  else
    cmd+=("--frames" "$FRAMES" "--tstart" "$TSTART" "--tend" "$TEND" "--fps" "$FPS")
  fi

  if [ "$ANIMATE" = true ]; then
    cmd+=("--animate" "--out" "$out_file")
  else
    cmd+=("--out" "$out_file")
  fi

  # execute
  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
done

echo "\nAnimations generated in $OUT_DIR (mp4/png)"
