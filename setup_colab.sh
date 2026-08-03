#!/bin/bash
# Setup CausalVLA on Google Colab
# Usage: !bash setup_colab.sh
set -e

echo "=== CausalVLA Colab Setup ==="

# 1. Install LeRobot
echo "[1/5] Installing LeRobot..."
pip install "lerobot[training]" -q

# 2. Clone CausalVLA repo (if not already)
if [ ! -d "causalvla" ]; then
    echo "[2/5] Cloning CausalVLA..."
    git clone https://github.com/phawitb/causalvla.git causalvla
else
    echo "[2/5] Updating CausalVLA..."
    cd causalvla && git pull && cd ..
fi

# 3. Install causal_aug package
echo "[3/5] Installing causal_aug..."
pip install -e causalvla/causal_aug -q

# 4. Copy causal_vla policy into lerobot
echo "[4/5] Installing causal_vla policy..."
LEROBOT_POLICIES=$(python -c "import lerobot.policies; import os; print(os.path.dirname(lerobot.policies.__file__))")
cp -r causalvla/lerobot_patches/causal_vla "$LEROBOT_POLICIES/"

# Register in __init__.py
if ! grep -q "causal_vla" "$LEROBOT_POLICIES/__init__.py"; then
    # Add import after ACTConfig line
    sed -i 's/from .act.configuration_act import ACTConfig as ACTConfig/from .act.configuration_act import ACTConfig as ACTConfig\nfrom .causal_vla.configuration_causal_vla import CausalVLAConfig as CausalVLAConfig/' "$LEROBOT_POLICIES/__init__.py"
    echo "  Registered CausalVLAConfig in policies/__init__.py"
fi

# 5. Patch SmolVLA with forward_with_latent()
echo "[5/5] Patching SmolVLA..."
SMOLVLA_DIR=$(python -c "import lerobot.policies.smolvla.modeling_smolvla as m; import os; print(os.path.dirname(m.__file__))")
SMOLVLA_FILE="$SMOLVLA_DIR/modeling_smolvla.py"

if ! grep -q "forward_with_latent" "$SMOLVLA_FILE"; then
    # Apply the patch
    LEROBOT_ROOT=$(python -c "import lerobot; import os; print(os.path.dirname(os.path.dirname(lerobot.__file__)))")
    cd "$LEROBOT_ROOT" && git apply --whitespace=fix "$(cd - > /dev/null && pwd)/causalvla/lerobot_patches/lerobot_causalvla.patch" 2>/dev/null || {
        echo "  Patch failed (may already be applied or version mismatch)"
        echo "  Trying manual patch..."
        python -c "
import inspect
import lerobot.policies.smolvla.modeling_smolvla as m
path = inspect.getfile(m)
with open(path) as f: src = f.read()
if 'forward_with_latent' not in src:
    with open('$(pwd)/causalvla/lerobot_patches/forward_with_latent.py') as f: patch = f.read()
    src = src.replace('    def sample_actions(', patch + '\n    def sample_actions(')
    with open(path, 'w') as f: f.write(src)
    print('  Manual patch applied')
else:
    print('  Already patched')
"
    }
    cd - > /dev/null
else
    echo "  Already patched"
fi

# Verify
echo ""
echo "=== Verification ==="
python -c "
from causal_aug import CausalAugmenter
from lerobot.policies.causal_vla.configuration_causal_vla import CausalVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
import torch

print('causal_aug:', 'OK')
print('CausalVLAConfig:', 'OK')
print('forward_with_latent:', hasattr(VLAFlowMatching, 'forward_with_latent'))

aug = CausalAugmenter(K=3)
out = aug(torch.randn(2, 3, 256, 256))
print(f'CausalAugmenter output: {out.shape}')
print()
print('Setup complete!')
"
