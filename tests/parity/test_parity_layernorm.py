"""
Parity test: llm.kittens layernorm_forward vs llm.c layernorm_forward.

Both probes consume identical bf16 inputs from the same iodir and write their
outputs to subdirs `tk/` and `ref/`. We compare element-wise. On SM120 the TK
path falls back to the same CUDA kernel as llm.c (sources match), so the test
passes trivially. On H100 the TK fast path runs and this is a real comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    bf16_bits_to_fp32,
    load_bf16,
    max_abs_diff,
    save_bf16,
    save_shape,
)


@pytest.mark.kernel
def test_parity_layernorm(kernel_runner, iodir):
    B, T, C = 4, 64, 768
    rng = np.random.default_rng(7)
    inp = rng.uniform(-1.0, 1.0, size=(B * T, C)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, size=(C,)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, size=(C,)).astype(np.float32)

    save_bf16(iodir / "inp.npy", inp)
    save_bf16(iodir / "weight.npy", weight)
    save_bf16(iodir / "bias.npy", bias)
    save_shape(iodir / "shape.npy", B, T, C)

    r_ref = kernel_runner.parity_run("probe_layernorm_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref probe failed:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_layernorm_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk probe failed:\n{r_tk.stdout}\n{r_tk.stderr}"

    out_ref = load_bf16(iodir / "ref" / "out.npy")
    out_tk = load_bf16(iodir / "tk" / "out.npy")
    mean_ref = np.load(iodir / "ref" / "mean.npy")
    mean_tk = np.load(iodir / "tk" / "mean.npy")
    rstd_ref = np.load(iodir / "ref" / "rstd.npy")
    rstd_tk = np.load(iodir / "tk" / "rstd.npy")

    out_diff = max_abs_diff(out_tk, out_ref)
    mean_diff = max_abs_diff(mean_tk, mean_ref)
    rstd_diff = max_abs_diff(rstd_tk, rstd_ref)

    # Tight bf16 tolerance — these are the same input and (on SM120) the same
    # kernel. On H100 the TK fast path computes the sum in bf16 (warp::sum on
    # sv_bf<D>) and only converts to fp32 for the divide; llm.c's reference
    # sums in fp32 throughout. For D=768 the mean/rstd drift is ~5e-4 / ~7e-3,
    # one bf16 ULP scaled by D. Tolerances here are calibrated for that path;
    # the SM120 CUDA fallback hits the original tighter bounds and still
    # passes within them.
    out_tol = 0.05
    mean_tol = 1e-3
    rstd_tol = 1e-2

    print(f"out  max_abs_diff = {out_diff:.6f} (tol {out_tol})")
    print(f"mean max_abs_diff = {mean_diff:.6f} (tol {mean_tol})")
    print(f"rstd max_abs_diff = {rstd_diff:.6f} (tol {rstd_tol})")

    assert out_diff <= out_tol, f"layernorm out drift: {out_diff} > {out_tol}"
    assert mean_diff <= mean_tol, f"layernorm mean drift: {mean_diff} > {mean_tol}"
    assert rstd_diff <= rstd_tol, f"layernorm rstd drift: {rstd_diff} > {rstd_tol}"
