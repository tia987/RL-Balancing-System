#!/bin/bash
# Install for the `main` branch: Isaac Lab 3.0 (submodule) + Isaac Sim 6.1 on Python 3.12.
# Works on the workstation and on Euler. Run it from anywhere; it works in its own folder.
#
# On Euler, fetch the Isaac Lab submodule on the LOGIN node first (git over SSH may not work
# from compute nodes), then run this script, ideally as a batch job (install.sbatch).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$REPO_ROOT"

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
