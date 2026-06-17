#!/usr/bin/env bash
# Applies patches to onnxruntime-genai 0.11.4 builder to support
# transformers 5.12+ where Gemma 3's RoPE config moved into rope_parameters dict.
#
# Run after installing the builder venv:
#   source /tmp/genai-builder-venv310/bin/activate
#   bash patches/apply.sh

set -e

VENV="${1:-/tmp/genai-builder-venv310}"
BUILDERS="$VENV/lib/python3.10/site-packages/onnxruntime_genai/models/builders"

if [ ! -d "$BUILDERS" ]; then
  echo "Error: builders directory not found at $BUILDERS"
  echo "Usage: bash patches/apply.sh [venv-path]"
  exit 1
fi

python3 - <<EOF
import glob, re

# --- Patch 1: gemma.py (rope_local_theta) ---
f = "$BUILDERS/gemma.py"
c = open(f).read()
p = re.sub(
    r'self\.rope_local_theta\s*=\s*config\.rope_local_base_freq',
    '''if hasattr(config, "rope_local_base_freq"):
            self.rope_local_theta = config.rope_local_base_freq
        elif hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_local_theta = config.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)
        else:
            self.rope_local_theta = 10000.0''',
    c
)
open(f, 'w').write(p)
print("gemma.py  ->", "patched" if p != c else "WARNING: pattern not found (already patched or version mismatch)")

# --- Patch 2: base.py (rope_theta fallback) ---
f = "$BUILDERS/base.py"
c = open(f).read()
old = 'config.rope_theta if hasattr(config, "rope_theta") else config.rope_embedding_base if hasattr(config, "rope_embedding_base") else 10000'
new = 'config.rope_theta if hasattr(config, "rope_theta") else config.rope_embedding_base if hasattr(config, "rope_embedding_base") else config.rope_parameters.get("full_attention", {}).get("rope_theta", 10000) if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict) else 10000'
p = c.replace(old, new)
open(f, 'w').write(p)
print("base.py   ->", "patched" if p != c else "WARNING: pattern not found (already patched or version mismatch)")
EOF

echo "Done. You can now run the model builder."
