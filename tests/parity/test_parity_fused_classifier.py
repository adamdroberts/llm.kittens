"""Parity test: llm.kittens fused_classifier vs llm.c fused_classifier.

Both probes skip cleanly on non-Hopper (the kernel deadlocks on RTX 5090's
1536-thread SM with __launch_bounds__(1024, 2)). On SM90 / H100 they run.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
@pytest.mark.parametrize("B,T,V,P", [
    (1, 1, 1024, 1024),   # smallest aligned case: 1 block, V==P (no tail loop)
    (2, 8, 1003, 1024),   # original failing case: V<P, unaligned tail
])
def test_parity_fused_classifier(kernel_runner, iodir, B, T, V, P):
    rng = np.random.default_rng(555)
    logits = rng.uniform(-2.0, 2.0, size=(B * T, P)).astype(np.float32)
    targets = rng.integers(0, V, size=(B * T,), dtype=np.int32)
    save_bf16(iodir / "logits.npy", logits)
    np.save(iodir / "targets.npy", targets)
    save_shape(iodir / "shape.npy", B, T, V, P)

    r_ref = kernel_runner.parity_run("probe_fused_classifier_ref", str(iodir))
    assert r_ref.exit_code == 0, f"ref:\n{r_ref.stdout}\n{r_ref.stderr}"
    r_tk = kernel_runner.parity_run("probe_fused_classifier_tk", str(iodir))
    assert r_tk.exit_code == 0, f"tk:\n{r_tk.stdout}\n{r_tk.stderr}"

    if "SKIPPED" in r_ref.stdout or "SKIPPED" in r_tk.stdout:
        pytest.skip("fused_classifier kernel needs sm_90; both probes skipped.")

    loss_ref = np.load(iodir / "ref" / "loss.npy")
    loss_tk  = np.load(iodir / "tk"  / "loss.npy")
    dlog_ref = load_bf16(iodir / "ref" / "dlogits.npy")
    dlog_tk  = load_bf16(iodir / "tk"  / "dlogits.npy")

    loss_diff = max_abs_diff(loss_tk, loss_ref)
    dlog_diff = max_abs_diff(dlog_tk, dlog_ref)
    print(f"loss    max_abs_diff = {loss_diff:.6f}")
    print(f"dlogits max_abs_diff = {dlog_diff:.6f}")
    # Both probes use byte-identical kernel sources — should be bit-exact.
    assert loss_diff <= 1e-5, f"loss drift: {loss_diff}"
    assert dlog_diff <= 1e-5, f"dlogits drift: {dlog_diff}"
