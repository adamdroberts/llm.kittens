"""Parity test: llm.kittens rope forward+backward vs PyTorch reference."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


def _torch_rope_apply(x, cos, sin, *, inverse=False):
    """Apply RoPE by pairing adjacent halves and rotating each pair."""
    import torch
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    if inverse:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    else:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


@pytest.mark.kernel
@pytest.mark.parametrize("B,H,T,HS", [
    (1, 1, 16, 64),    # active_warps = 1
    (1, 1, 128, 64),   # active_warps = 8 (full)
    (1, 4, 64, 64),    # active_warps = 4 (original failing case)
])
def test_parity_rope(kernel_runner, iodir, B, H, T, HS):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not initialized for this pytest process")

    rng = np.random.default_rng(31)
    x    = rng.uniform(-1.0, 1.0, size=(B, H, T, HS)).astype(np.float32)
    dout = rng.uniform(-0.5, 0.5, size=(B, H, T, HS)).astype(np.float32)
    # Standard RoPE freqs at base 10000.
    half = HS // 2
    inv_freq = 1.0 / (10000 ** (np.arange(0, half).astype(np.float32) / half))
    pos = np.arange(T, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)            # (T, half)
    cos_v = np.cos(freqs).astype(np.float32)
    sin_v = np.sin(freqs).astype(np.float32)

    save_bf16(iodir / "x.npy",    x.reshape(-1))
    save_bf16(iodir / "cos.npy",  cos_v.reshape(-1))
    save_bf16(iodir / "sin.npy",  sin_v.reshape(-1))
    save_bf16(iodir / "dout.npy", dout.reshape(-1))
    save_shape(iodir / "shape.npy", B, H, T, HS)

    r = kernel_runner.parity_run("probe_rope", str(iodir))
    assert r.exit_code == 0, f"probe:\n{r.stdout}\n{r.stderr}"

    out_tk = load_bf16(iodir / "tk" / "out.npy").reshape(B, H, T, HS)
    dx_tk  = load_bf16(iodir / "tk" / "dx.npy").reshape(B, H, T, HS)

    # Torch reference (fp32, then bf16 quantize).
    x_t    = torch.from_numpy(x).bfloat16().cuda().float()
    cos_t  = torch.from_numpy(cos_v).bfloat16().cuda().float().view(1, 1, T, half)
    sin_t  = torch.from_numpy(sin_v).bfloat16().cuda().float().view(1, 1, T, half)
    dout_t = torch.from_numpy(dout).bfloat16().cuda().float()
    out_ref = _torch_rope_apply(x_t,    cos_t, sin_t, inverse=False).bfloat16().float().cpu().numpy()
    dx_ref  = _torch_rope_apply(dout_t, cos_t, sin_t, inverse=True ).bfloat16().float().cpu().numpy()

    out_diff = max_abs_diff(out_tk, out_ref)
    dx_diff  = max_abs_diff(dx_tk,  dx_ref)
    print(f"out max_abs_diff = {out_diff:.6f}")
    print(f"dx  max_abs_diff = {dx_diff:.6f}")
    assert out_diff <= 0.05, f"rope fwd drift: {out_diff}"
    assert dx_diff  <= 0.05, f"rope bwd drift: {dx_diff}"
