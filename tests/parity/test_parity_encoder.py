"""Parity test: llm.kittens encoder_forward vs llm.c encoder_forward."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_encoder(kernel_runner, iodir):
    B, T, C, V = 2, 64, 128, 256
    rng = np.random.default_rng(7)
    inp = rng.integers(0, V, size=(B * T,), dtype=np.int32)
    wte = rng.uniform(-0.1, 0.1, size=(V, C)).astype(np.float32)
    wpe = rng.uniform(-0.05, 0.05, size=(T, C)).astype(np.float32)
    np.save(iodir / "inp.npy", inp)
    save_bf16(iodir / "wte.npy", wte)
    save_bf16(iodir / "wpe.npy", wpe)
    save_shape(iodir / "shape.npy", B, T, C, V)

    r_ref = kernel_runner.parity_run("probe_encoder_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_encoder_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    out_ref = load_bf16(iodir / "ref" / "out.npy")
    out_tk  = load_bf16(iodir / "tk"  / "out.npy")
    diff = max_abs_diff(out_tk, out_ref)
    print(f"encoder fwd max_abs_diff = {diff:.6f}")
    assert diff <= 1e-3
