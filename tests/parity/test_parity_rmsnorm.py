"""Parity test: llm.kittens rmsnorm_forward vs PyTorch reference."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import load_bf16, max_abs_diff, save_bf16, save_shape


@pytest.mark.kernel
def test_parity_rmsnorm(kernel_runner, iodir):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this host")

    N, C = 256, 768
    eps = 1e-5
    rng = np.random.default_rng(13)
    inp = rng.uniform(-1.0, 1.0, size=(N, C)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, size=(C,)).astype(np.float32)
    save_bf16(iodir / "inp.npy", inp)
    save_bf16(iodir / "weight.npy", weight)
    save_shape(iodir / "shape.npy", N, C)

    r = kernel_runner.parity_run("probe_rmsnorm", str(iodir))
    assert r.exit_code == 0, f"probe:\n{r.stdout}\n{r.stderr}"

    out_tk  = load_bf16(iodir / "tk" / "out.npy").reshape(N, C)
    rstd_tk = np.load(iodir / "tk" / "rstd.npy")

    # Torch reference — feed the SAME bf16-quantized input the kernel saw, so
    # the only source of drift is fp32 reduction order in rstd.
    from .conftest import bf16_bits_to_fp32
    inp_q = bf16_bits_to_fp32(np.load(iodir / "inp.npy")).reshape(N, C)
    w_q   = bf16_bits_to_fp32(np.load(iodir / "weight.npy"))
    inp_t = torch.from_numpy(inp_q).cuda()
    w_t   = torch.from_numpy(w_q).cuda()
    rstd_ref = torch.rsqrt(inp_t.pow(2).mean(dim=1) + eps)
    out_ref  = (inp_t * rstd_ref.unsqueeze(1) * w_t).bfloat16().float().cpu().numpy()
    rstd_ref = rstd_ref.cpu().numpy()

    out_diff  = max_abs_diff(out_tk, out_ref)
    rstd_diff = max_abs_diff(rstd_tk, rstd_ref)
    print(f"out  max_abs_diff = {out_diff:.6f}")
    print(f"rstd max_abs_diff = {rstd_diff:.6f}")
    assert out_diff  <= 0.05, f"rmsnorm out drift: {out_diff}"
    # rstd is fp32 but reductions happen in different order on GPU vs torch.
    # Bound is conservative — rstd magnitudes ≈ 1/sqrt(mean(x^2)) ~ 1.7 for
    # x∈[-1,1], so 1e-2 is ~0.6% relative.
    assert rstd_diff <= 1e-2, f"rmsnorm rstd drift: {rstd_diff}"
