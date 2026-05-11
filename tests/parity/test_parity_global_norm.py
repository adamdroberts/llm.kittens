"""Parity test: llm.kittens global_norm_squared vs llm.c global_norm_squared."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import save_bf16, save_shape


@pytest.mark.kernel
def test_parity_global_norm(kernel_runner, iodir):
    N = 4096 * 32
    rng = np.random.default_rng(31)
    vals = rng.uniform(-0.5, 0.5, size=(N,)).astype(np.float32)
    save_bf16(iodir / "vals.npy", vals)
    save_shape(iodir / "shape.npy", N)

    r_ref = kernel_runner.parity_run("probe_global_norm_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_global_norm_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    norm_ref = float(np.load(iodir / "ref" / "norm.npy")[0])
    norm_tk  = float(np.load(iodir / "tk"  / "norm.npy")[0])
    rel = abs(norm_tk - norm_ref) / max(norm_ref, 1e-12)
    print(f"|norm_ref|={norm_ref:.4f} |norm_tk|={norm_tk:.4f} rel_diff={rel:.3e}")
    # The reductions are in fp32 so this should be exact bitwise, but allow a
    # tiny floating-rounding slack just in case the block-grid layout changes.
    assert rel <= 1e-5, f"global_norm drift: rel={rel}"
