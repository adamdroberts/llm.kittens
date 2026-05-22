#!/usr/bin/env python3
"""Torch SM120 runtime-family timing prototype for GPT-2 shapes."""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
import torch.nn.functional as F


def event_time_us(fn, *, warmup: int, repeats: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


def print_result(name: str, shape: str, us: float) -> None:
    print(f"{name:<30} | {shape:<28} | {'Torch':<12} | {us:9.3f} us")


def release_cuda_cache() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def gelu_backward_tanh(dout: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    x = inp.float()
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_out = torch.tanh(tanh_arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    return (local_grad * dout.float()).to(torch.bfloat16)


def bench_bias_and_gelu(*, repeats: int, warmup: int) -> None:
    bt = 64 * 1024
    hidden_c = 768
    qkv_c = 2304
    fc_c = 3072

    torch.manual_seed(120)
    hidden = torch.randn((bt, hidden_c), device="cuda", dtype=torch.bfloat16)
    qkv = torch.randn((bt, qkv_c), device="cuda", dtype=torch.bfloat16)
    fc = torch.randn((bt, fc_c), device="cuda", dtype=torch.bfloat16)
    fc_aux = torch.randn((bt, fc_c), device="cuda", dtype=torch.bfloat16)
    bias_hidden = torch.randn((hidden_c,), device="cuda", dtype=torch.bfloat16)
    bias_fc = torch.randn((fc_c,), device="cuda", dtype=torch.bfloat16)

    print_result(
        "bias_add",
        "BT=65536 OC=768",
        event_time_us(lambda: hidden + bias_hidden, warmup=warmup, repeats=repeats, iters=50),
    )
    print_result(
        "bias_add",
        "BT=65536 OC=3072",
        event_time_us(lambda: fc + bias_fc, warmup=warmup, repeats=repeats, iters=25),
    )
    print_result(
        "gelu_forward",
        "BT=65536 C=3072",
        event_time_us(lambda: F.gelu(fc_aux, approximate="tanh"), warmup=warmup, repeats=repeats, iters=25),
    )
    print_result(
        "gelu_backward_inplace",
        "BT=65536 C=3072",
        event_time_us(lambda: gelu_backward_tanh(fc, fc_aux), warmup=warmup, repeats=repeats, iters=10),
    )
    print_result(
        "bias_grad_reduce",
        "BT=65536 OC=768",
        event_time_us(lambda: hidden.float().sum(dim=0).to(torch.bfloat16), warmup=warmup, repeats=repeats, iters=10),
    )
    print_result(
        "bias_grad_reduce",
        "BT=65536 OC=2304",
        event_time_us(lambda: qkv.float().sum(dim=0).to(torch.bfloat16), warmup=warmup, repeats=repeats, iters=10),
    )
    print_result(
        "bias_grad_reduce",
        "BT=65536 OC=3072",
        event_time_us(lambda: fc.float().sum(dim=0).to(torch.bfloat16), warmup=warmup, repeats=repeats, iters=10),
    )


def bench_optimizer_and_norm(*, repeats: int, warmup: int) -> None:
    params_count = 124_475_904
    torch.manual_seed(120)
    params = torch.nn.Parameter(torch.ones((params_count,), device="cuda", dtype=torch.bfloat16))
    grad = torch.ones_like(params)
    params.grad = grad
    optimizer = torch.optim.AdamW(
        [params],
        lr=0.0006,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
        fused=True,
    )
    optimizer.step()
    torch.cuda.synchronize()
    state = optimizer.state[params]
    exp_avg_dtype = state["exp_avg"].dtype if torch.is_tensor(state.get("exp_avg")) else "unknown"
    exp_avg_sq_dtype = state["exp_avg_sq"].dtype if torch.is_tensor(state.get("exp_avg_sq")) else "unknown"
    print(f"Torch fused AdamW state dtypes: exp_avg={exp_avg_dtype}; exp_avg_sq={exp_avg_sq_dtype}")

    print_result(
        "cuda_memset",
        f"grad_elems={params_count}",
        event_time_us(lambda: grad.zero_(), warmup=warmup, repeats=repeats, iters=20),
    )
    grad.fill_(1.0)

    print_result(
        "global_norm_squared",
        "params=124475904",
        event_time_us(lambda: torch.sum(grad.float() * grad.float()), warmup=warmup, repeats=repeats, iters=5),
    )

    def adamw_step() -> None:
        params.grad = grad
        optimizer.step()

    print_result(
        "adamw_update_bf16_state",
        "params=124475904 no-master",
        event_time_us(adamw_step, warmup=warmup, repeats=repeats, iters=5),
    )

    del optimizer, params, state
    release_cuda_cache()

    params_fp32_state = torch.ones((params_count,), device="cuda", dtype=torch.bfloat16)
    grad_fp32_state = torch.ones_like(params_fp32_state)
    m = torch.zeros((params_count,), device="cuda", dtype=torch.float32)
    v = torch.zeros((params_count,), device="cuda", dtype=torch.float32)

    def adamw_fp32_state_step() -> None:
        grad_f = grad_fp32_state.float()
        m.mul_(0.9).add_(grad_f, alpha=0.1)
        v.mul_(0.95).addcmul_(grad_f, grad_f, value=0.05)
        update = m / (v.sqrt() + 1.0e-8)
        params_fp32_state.mul_(1.0 - 0.0006 * 0.1)
        params_fp32_state.add_(update.to(torch.bfloat16), alpha=-0.0006)

    print_result(
        "adamw_update",
        "params=124475904 no-master fp32-state",
        event_time_us(adamw_fp32_state_step, warmup=warmup, repeats=repeats, iters=1),
    )


def bench_encoder_and_memory(*, repeats: int, warmup: int) -> None:
    batch = 64
    seq = 1024
    channels = 768
    padded_vocab = 50304
    hidden_elems = batch * seq * channels
    torch.manual_seed(120)
    wte = torch.randn((padded_vocab, channels), device="cuda", dtype=torch.bfloat16)
    wpe = torch.randn((seq, channels), device="cuda", dtype=torch.bfloat16)
    tokens = torch.zeros((batch, seq), device="cuda", dtype=torch.long)
    hidden = torch.empty((batch, seq, channels), device="cuda", dtype=torch.bfloat16)
    hidden_copy = torch.empty_like(hidden)

    def encoder_forward() -> torch.Tensor:
        token_emb = wte.index_select(0, tokens.reshape(-1)).view(batch, seq, channels)
        return token_emb + wpe.view(1, seq, channels)

    print_result(
        "encoder_forward",
        "B=64 T=1024 C=768",
        event_time_us(encoder_forward, warmup=warmup, repeats=repeats, iters=20),
    )
    print_result(
        "cuda_memset",
        f"hidden_elems={hidden_elems}",
        event_time_us(lambda: hidden.zero_(), warmup=warmup, repeats=repeats, iters=20),
    )
    print_result(
        "cuda_copy_d2d",
        f"hidden_elems={hidden_elems}",
        event_time_us(lambda: hidden_copy.copy_(hidden), warmup=warmup, repeats=repeats, iters=20),
    )


def bench_logits_memory(*, repeats: int, warmup: int) -> None:
    logits_elems = 64 * 1024 * 50304
    logits = torch.empty((logits_elems,), device="cuda", dtype=torch.bfloat16)
    logits_copy = torch.empty_like(logits)
    print_result(
        "cuda_memset",
        f"logits_elems={logits_elems}",
        event_time_us(lambda: logits.zero_(), warmup=warmup, repeats=repeats, iters=1),
    )
    print_result(
        "cuda_copy_d2d",
        f"logits_elems={logits_elems}",
        event_time_us(lambda: logits_copy.copy_(logits), warmup=warmup, repeats=repeats, iters=1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch SM120 GPT-2 runtime-family benchmark prototype")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PyTorch CUDA context is not initialized in this process; rerun inside the target benchmark context.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch runtime device: {device_name}; capability=sm_{capability[0]}{capability[1]}")
    bench_bias_and_gelu(repeats=args.repeats, warmup=args.warmup)
    release_cuda_cache()
    bench_optimizer_and_norm(repeats=args.repeats, warmup=args.warmup)
    release_cuda_cache()
    bench_encoder_and_memory(repeats=args.repeats, warmup=args.warmup)
    release_cuda_cache()
    bench_logits_memory(repeats=args.repeats, warmup=args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
