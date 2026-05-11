"""Parity test: llm.kittens SwiGLU vs PyTorch reference (Family B).

llm.c has no SwiGLU, so we compute the reference in torch (`silu(g) * u` for
forward; `dgate = dout * up * dsilu(g)`, `dup = dout * silu(g)` for backward).
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_swiglu(kernel_runner, iodir):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this host (torch reference needs a GPU "
                    "to mirror the kernel's bf16 quantization order)")

    N = 4096
    rng = np.random.default_rng(123)
    gate = rng.uniform(-1.0, 1.0, size=(N,)).astype(np.float32)
    up   = rng.uniform(-1.0, 1.0, size=(N,)).astype(np.float32)
    dout = rng.uniform(-0.5, 0.5, size=(N,)).astype(np.float32)
    save_bf16(iodir / "gate.npy", gate)
    save_bf16(iodir / "up.npy",   up)
    save_bf16(iodir / "dout.npy", dout)
    save_shape(iodir / "shape.npy", N)

    r = kernel_runner.parity_run("probe_swiglu", str(iodir))
    assert r.exit_code == 0, f"probe:\n{r.stdout}\n{r.stderr}"

    out_tk   = load_bf16(iodir / "tk" / "out.npy")
    dgate_tk = load_bf16(iodir / "tk" / "dgate.npy")
    dup_tk   = load_bf16(iodir / "tk" / "dup.npy")

    # Torch reference — operate in bf16 on GPU so quantization order matches
    # the kernel's intermediate precision (silu in fp32, multiply in bf16).
    g_t = torch.from_numpy(gate).bfloat16().cuda()
    u_t = torch.from_numpy(up).bfloat16().cuda()
    do_t = torch.from_numpy(dout).bfloat16().cuda()
    sig = torch.sigmoid(g_t.float())
    silu_g = (g_t.float() * sig).bfloat16()
    out_ref = (silu_g * u_t).float().cpu().numpy()
    dsilu_g = sig * (1.0 + g_t.float() * (1.0 - sig))
    dgate_ref = (do_t.float() * u_t.float() * dsilu_g).bfloat16().float().cpu().numpy()
    dup_ref   = (do_t.float() * silu_g.float()).bfloat16().float().cpu().numpy()

    out_diff   = max_abs_diff(out_tk, out_ref)
    dgate_diff = max_abs_diff(dgate_tk, dgate_ref)
    dup_diff   = max_abs_diff(dup_tk, dup_ref)
    print(f"out   max_abs_diff = {out_diff:.6f}")
    print(f"dgate max_abs_diff = {dgate_diff:.6f}")
    print(f"dup   max_abs_diff = {dup_diff:.6f}")
    # bf16 rel precision ~3e-3; with values near unity, ~1e-2 absolute is the
    # quantization floor.
    assert out_diff   <= 0.02, f"swiglu out drift: {out_diff}"
    assert dgate_diff <= 0.02, f"swiglu dgate drift: {dgate_diff}"
    assert dup_diff   <= 0.02, f"swiglu dup drift: {dup_diff}"
