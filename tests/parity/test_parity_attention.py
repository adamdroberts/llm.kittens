"""Parity test: llm.kittens attention_forward vs llm.c attention_forward.

Forward only — backward signatures and scratch layouts diverge between the
two repos enough that a meaningful side-by-side needs more wrapping.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_attention_forward(kernel_runner, iodir):
    # T=192 hits the TK-fast forward path on H100 and also a clean shape on
    # llm.c's CUDA path. C=768 / NH=12 → HS=64, in the {64, 128} TK supported set.
    B, T, C, NH = 1, 192, 768, 12
    rng = np.random.default_rng(17)
    inp = rng.uniform(-1.0, 1.0, size=(B, T, 3 * C)).astype(np.float32)
    save_bf16(iodir / "inp.npy", inp)
    save_shape(iodir / "shape.npy", B, T, C, NH)

    r_ref = kernel_runner.parity_run("probe_attention_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_attention_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    out_ref = load_bf16(iodir / "ref" / "out.npy")
    out_tk  = load_bf16(iodir / "tk"  / "out.npy")
    diff = max_abs_diff(out_tk, out_ref)
    print(f"attention forward max_abs_diff = {diff:.6f}")
    # llm.c uses cuBLASLt for QK^T and att-V; TK uses fused attention on H100,
    # CUDA fallback elsewhere. Reduction order differs → loose bf16 tolerance.
    assert diff <= 0.1, f"attention drift: {diff}"
