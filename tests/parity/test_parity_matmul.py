"""Parity test: llm.kittens matmul_forward (TK GEMM) vs llm.c matmul_forward_cublaslt."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_matmul_forward(kernel_runner, iodir):
    # Square shape that hits the default TK matmul template and a normal cuBLAS
    # path. M=B*T=1024, N=OC=1024, K=C=1024.
    B, T, C, OC = 1, 1024, 1024, 1024
    rng = np.random.default_rng(42)
    inp = rng.uniform(-1.0, 1.0, size=(B, T, C)).astype(np.float32)
    weight = rng.uniform(-1.0, 1.0, size=(OC, C)).astype(np.float32)  # llm.c layout
    bias = rng.uniform(-0.1, 0.1, size=(OC,)).astype(np.float32)
    save_bf16(iodir / "inp.npy", inp)
    save_bf16(iodir / "weight.npy", weight)
    save_bf16(iodir / "bias.npy", bias)
    save_shape(iodir / "shape.npy", B, T, C, OC)

    r_ref = kernel_runner.parity_run("probe_matmul_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_matmul_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    out_ref = load_bf16(iodir / "ref" / "out.npy")
    out_tk  = load_bf16(iodir / "tk"  / "out.npy")
    diff = max_abs_diff(out_tk, out_ref)
    print(f"matmul forward max_abs_diff = {diff:.6f}")
    # bf16 GEMM with K=1024 — accumulation error ~sqrt(K)*eps ~0.1; cuBLASLt
    # vs TK use different reduction orders. Loose tolerance.
    assert diff <= 0.5, f"matmul drift: {diff}"
