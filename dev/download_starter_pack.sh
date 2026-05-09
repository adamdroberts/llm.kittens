#!/bin/bash
# Download GPT-2 124M reference checkpoints and tokenizer for llm.kittens.
# Mirrors what llm.c expects in its root directory.
#
# Files fetched (matching llm.c's expected names and layout):
#   gpt2_tokenizer.bin              — GPT-2 BPE vocab table (used by sampler.h)
#   gpt2_124M.bin                   — fp32 weights, magic 20240326 v3
#   gpt2_124M_bf16.bin              — bf16 weights, magic 20240326 v5
#   gpt2_124M_debug_state.bin       — B=4 T=64 forward+backward reference
#                                      tensors for test_gpt2.cu numerical parity
#
# The .bin format is exactly the same as llm.c's (parameter ordering is
# preserved across the port), so the same files load directly.

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer huggingface-hub if available (resumable, parallel), fall back to curl.
fetch() {
    local url="$1"
    local out="$2"
    if [ -f "$out" ]; then
        echo "✓ $out already present, skipping"
        return 0
    fi
    echo "→ fetching $out"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
    else
        echo "Need curl or wget on PATH" >&2
        exit 1
    fi
}

# Karpathy's HF mirror (same one llm.c documents)
BASE="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main"

fetch "$BASE/gpt2_tokenizer.bin"           gpt2_tokenizer.bin
fetch "$BASE/gpt2_124M.bin"                gpt2_124M.bin
fetch "$BASE/gpt2_124M_bf16.bin"           gpt2_124M_bf16.bin
fetch "$BASE/gpt2_124M_debug_state.bin"    gpt2_124M_debug_state.bin

echo
echo "✓ starter pack downloaded into $(pwd)"
ls -lh gpt2_tokenizer.bin gpt2_124M*.bin
