"""Parity test: llm.kittens attention_gqa_forward vs PyTorch reference (GQA + RoPE + causal MHA)."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


def _rope_apply(x, cos, sin):
    """RoPE in the same paired-halves layout the kernel uses."""
    import torch
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@pytest.mark.kernel
def test_parity_attention_gqa_forward(kernel_runner, iodir):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this host")
    F = torch.nn.functional

    # GQA shapes that hit the supported TK fast path on H100 and a sane
    # CUDA fallback elsewhere. NH=8, NKVH=2 → 4-to-1 grouping.
    B, T, C, NH, NKVH = 1, 256, 512, 8, 2
    HS = C // NH
    C_kv = NKVH * HS
    half = HS // 2

    rng = np.random.default_rng(57)
    # Packed input layout: (B, T, C + 2*C_kv) — Q, K, V concatenated on the channel axis.
    inp = rng.uniform(-1.0, 1.0, size=(B, T, C + 2 * C_kv)).astype(np.float32)
    inv_freq = 1.0 / (10000 ** (np.arange(0, half).astype(np.float32) / half))
    freqs = np.outer(np.arange(T, dtype=np.float32), inv_freq)
    cos_v = np.cos(freqs).astype(np.float32)
    sin_v = np.sin(freqs).astype(np.float32)

    save_bf16(iodir / "inp.npy", inp.reshape(-1))
    save_bf16(iodir / "cos.npy", cos_v.reshape(-1))
    save_bf16(iodir / "sin.npy", sin_v.reshape(-1))
    save_shape(iodir / "shape.npy", B, T, C, NH, NKVH)

    r = kernel_runner.parity_run("probe_attention_gqa", str(iodir))
    assert r.exit_code == 0, f"probe:\n{r.stdout}\n{r.stderr}"

    out_tk = load_bf16(iodir / "tk" / "out.npy").reshape(B, T, C)

    # Torch reference: split QKV → reshape into heads → RoPE Q/K → repeat KV
    # to NH heads → causal scaled dot-product attention → reshape back.
    inp_t = torch.from_numpy(inp).bfloat16().cuda().float()
    q = inp_t[..., :C].view(B, T, NH, HS).transpose(1, 2)              # (B, NH, T, HS)
    k = inp_t[..., C:C + C_kv].view(B, T, NKVH, HS).transpose(1, 2)    # (B, NKVH, T, HS)
    v = inp_t[..., C + C_kv:].view(B, T, NKVH, HS).transpose(1, 2)
    cos_t = torch.from_numpy(cos_v).bfloat16().cuda().float().view(1, 1, T, half)
    sin_t = torch.from_numpy(sin_v).bfloat16().cuda().float().view(1, 1, T, half)
    q = _rope_apply(q, cos_t, sin_t)
    k = _rope_apply(k, cos_t, sin_t)
    n_rep = NH // NKVH
    k = k.unsqueeze(2).expand(B, NKVH, n_rep, T, HS).reshape(B, NH, T, HS)
    v = v.unsqueeze(2).expand(B, NKVH, n_rep, T, HS).reshape(B, NH, T, HS)
    ref = F.scaled_dot_product_attention(
        q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float(),
        attn_mask=None, is_causal=True
    )
    out_ref = ref.transpose(1, 2).contiguous().view(B, T, C).bfloat16().float().cpu().numpy()

    diff = max_abs_diff(out_tk, out_ref)
    print(f"attention_gqa forward max_abs_diff = {diff:.6f}")
    # bf16 attention through softmax — looser tolerance than other ops.
    assert diff <= 0.2, f"attention_gqa drift: {diff}"
