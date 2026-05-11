"""Parity test: llm.kittens gelu vs llm.c gelu (forward + backward-inplace)."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_gelu(kernel_runner, iodir):
    N = 8192  # multiple of 512*8 (fwd) and 128*8 (bwd) for bf16 x128
    rng = np.random.default_rng(11)
    inp = rng.uniform(-3.0, 3.0, size=(N,)).astype(np.float32)
    dout = rng.uniform(-1.0, 1.0, size=(N,)).astype(np.float32)
    save_bf16(iodir / "inp.npy", inp)
    save_bf16(iodir / "dout.npy", dout)
    save_shape(iodir / "shape.npy", N)

    r_ref = kernel_runner.parity_run("probe_gelu_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_gelu_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    out_ref = load_bf16(iodir / "ref" / "out.npy")
    out_tk  = load_bf16(iodir / "tk"  / "out.npy")
    dinp_ref = load_bf16(iodir / "ref" / "dinp.npy")
    dinp_tk  = load_bf16(iodir / "tk"  / "dinp.npy")

    out_diff = max_abs_diff(out_tk, out_ref)
    dinp_diff = max_abs_diff(dinp_tk, dinp_ref)
    print(f"out  max_abs_diff = {out_diff:.6f}")
    print(f"dinp max_abs_diff = {dinp_diff:.6f}")
    assert out_diff <= 1e-3, f"gelu fwd drift: {out_diff}"
    assert dinp_diff <= 1e-3, f"gelu bwd drift: {dinp_diff}"
