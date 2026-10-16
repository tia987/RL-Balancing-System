#!/bin/bash
# Install for the `main` branch: Isaac Lab 3.0 (submodule) + Isaac Sim 6.1 on Python 3.12.
# Works on the workstation and on Euler. Run it from anywhere; it works in its own folder.
#
# On Euler it loads the needed modules itself. Fetch the Isaac Lab submodule on the LOGIN node
# first (git over SSH may not work from compute nodes), then run this script, on the login node
# or as a batch job (install.sbatch).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$REPO_ROOT"

# --- cluster modules (Euler); skipped automatically on machines without Lmod ----
# `bash installation.sh` runs a non-login shell where `module` may be undefined: load Lmod first.
if ! type module >/dev/null 2>&1 && [ -n "${LMOD_PKG:-}" ] && [ -f "$LMOD_PKG/init/bash" ]; then
  source "$LMOD_PKG/init/bash"
fi
if type module >/dev/null 2>&1; then
  # gcc: compiler for packages built from source; eth_proxy: internet on compute nodes
  module load stack/2025-06 gcc/12.2.0 eth_proxy
fi

# --- uv settings ------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_NO_CONFIG=1 # Euler: /cluster/scratch/uv.toml is unreadable
if [ -n "$SCRATCH" ]; then
  export UV_CACHE_DIR="$SCRATCH/.uv-cache" # Euler: keep the ~20 GB cache out of $HOME
fi
unset VIRTUAL_ENV # ignore whatever venv the calling shell had active

# --- Isaac Lab sources (pinned submodule commit, no history) -----------------
if [ ! -f IsaacLab/isaaclab.sh ]; then
  git submodule update --init --depth 1 IsaacLab
fi

# --- Python 3.12 venv + Isaac Sim 6.1 ----------------------------------------
uv venv --python 3.12 --allow-existing .env_isaaclab
source .env_isaaclab/bin/activate
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

uv pip install "isaacsim[all,extscache]==6.1.0.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match

# --- Isaac Lab (provides the `isaaclab` command) -------------------------------
export OMNI_KIT_ACCEPT_EULA=YES # no interactive license prompt
export TERM=xterm-256color      # isaaclab.sh exits on unknown terminals (kitty)
# isaaclab.sh tries `sudo apt-get install cmake build-essential` when cmake is missing.
# There is no sudo on Euler, so put a cmake into the venv instead (pip wheel, no root needed).
# A C/C++ compiler is also needed (on Euler it comes from the gcc module loaded above).
command -v cmake >/dev/null || uv pip install cmake
command -v gcc >/dev/null || {
  echo "ERROR: no C compiler found (gcc)" >&2
  exit 1
}
# The pip cmake is CMake 4, which refuses projects declaring cmake_minimum_required < 3.5.
# egl-probe 1.0.2 (pulled in by robomimic for the mimic extension) is one of them.
# Isaac Lab sets this same variable itself, but only on ARM machines.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
(cd IsaacLab && ./isaaclab.sh --install)

# --- project + extras ------------------------------------------------------------
uv pip install --no-deps -e TwoWheelBalance
uv pip install "moviepy<2.0" tensorboard # moviepy 2.x breaks video recording here

# --- sanity check ------------------------------------------------------------------
command -v isaaclab >/dev/null || {
  echo "ERROR: isaaclab command missing" >&2
  exit 1
}
python - <<'EOF'
import sys, importlib.metadata as m
print("python  ", sys.version.split()[0])
for p in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib"):
    try:
        print(f"{p:10s}", m.version(p))
    except m.PackageNotFoundError:
        print(f"{p:10s}", "MISSING")
EOF
echo "Done. Activate with: source $REPO_ROOT/.env_isaaclab/bin/activate"
