#!/bin/bash

# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv through python3.11 for isaac lab compatibility
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install Isaacs Sym
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
