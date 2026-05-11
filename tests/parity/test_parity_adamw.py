"""Parity test: llm.kittens adamw_update vs llm.c adamw_update (one step)."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import bf16_bits_to_fp32, load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_adamw(kernel_runner, iodir):
    N = 4096
    rng = np.random.default_rng(99)
    master = rng.uniform(-0.1, 0.1, size=(N,)).astype(np.float32)
    grad = rng.uniform(-0.05, 0.05, size=(N,)).astype(np.float32)
    m = np.zeros(N, dtype=np.float32)
    v = np.zeros(N, dtype=np.float32)

    save_bf16(iodir / "param.npy", master)
    np.save(iodir / "master.npy", master)
    save_bf16(iodir / "grad.npy", grad)
    np.save(iodir / "m.npy", m)
    np.save(iodir / "v.npy", v)
    save_shape(iodir / "shape.npy", N)

    r_ref = kernel_runner.parity_run("probe_adamw_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_adamw_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    master_ref = np.load(iodir / "ref" / "master.npy")
    master_tk  = np.load(iodir / "tk"  / "master.npy")
    m_ref = np.load(iodir / "ref" / "m.npy"); m_tk = np.load(iodir / "tk" / "m.npy")
    v_ref = np.load(iodir / "ref" / "v.npy"); v_tk = np.load(iodir / "tk" / "v.npy")
    param_ref = load_bf16(iodir / "ref" / "param.npy")
    param_tk  = load_bf16(iodir / "tk"  / "param.npy")

    print(f"master max_abs_diff = {max_abs_diff(master_tk, master_ref):.3e}")
    print(f"m      max_abs_diff = {max_abs_diff(m_tk, m_ref):.3e}")
    print(f"v      max_abs_diff = {max_abs_diff(v_tk, v_ref):.3e}")
    print(f"param  max_abs_diff = {max_abs_diff(param_tk, param_ref):.3e}")

    # FP32 master/m/v should be bit-exact (same arithmetic, same order).
    assert max_abs_diff(master_tk, master_ref) <= 1e-6
    assert max_abs_diff(m_tk, m_ref) <= 1e-6
    assert max_abs_diff(v_tk, v_ref) <= 1e-6
    # bf16 param goes through stochastic rounding with the same seed in both
    # — should also match bit-exact. Allow 1 ULP just in case.
    assert max_abs_diff(param_tk, param_ref) <= 1e-2
