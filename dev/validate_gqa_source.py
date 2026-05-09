#!/usr/bin/env python3
"""Source-level guards for the custom Llama GQA/RoPE attention path."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "llmc" / "attention_gqa.cuh"
TK_KERNEL = ROOT / "llmc" / "tk" / "attention_gqa_h100.cuh"
TK_BWD_KERNEL = ROOT / "llmc" / "tk" / "attention_h100.cuh"
SMOKE = ROOT / "dev" / "cuda" / "test_attention_gqa.cu"
REFERENCE = ROOT / "dev" / "validate_attention_gqa_reference.py"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def extract_function(text: str, marker: str) -> str:
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"missing function marker {marker!r}")
    body_start = text.find("{", start)
    if body_start == -1:
        raise AssertionError(f"missing function body for {marker!r}")
    depth = 1
    i = body_start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unterminated function {marker!r}")
    return text[body_start + 1 : i - 1]


def require_contains(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{context} missing {needle!r}")


def require_all(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        require_contains(text, needle, context, failures)


def require_order(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    cursor = -1
    for needle in needles:
        offset = text.find(needle, cursor + 1)
        if offset == -1:
            failures.append(f"{context} missing {needle!r}")
            continue
        cursor = offset


def validate_wrapper(wrapper: str, failures: list[str]) -> None:
    helper = extract_function(wrapper, "inline bool attention_gqa_uses_tk_tile_rope")
    require_all(
        helper,
        [
            "if (cos == nullptr || sin == nullptr || tk_workspace == nullptr)",
            "if (NH <= 0 || C % NH != 0)",
            "return llmk::attention_gqa::has_tk_backward(T, C / NH, NH, NKVH);",
        ],
        f"{rel(WRAPPER)} tile-RoPE gate",
        failures,
    )

    forward = extract_function(wrapper, "inline void attention_gqa_forward")
    require_order(
        forward,
        [
            "bool tk_tile_rope = attention_gqa_uses_tk_tile_rope",
            "if (tk_tile_rope) {",
            "gqa_permute_q_kernel",
            "gqa_permute_kv_kernel",
            "} else if (cos != nullptr && sin != nullptr) {",
            "gqa_permute_q_rope_kernel",
            "gqa_permute_kv_rope_kernel",
            "llmk::attention_gqa::has_tk_forward",
            "llmk::attention_gqa::launch_forward_causal_rope",
            "llmk::attention_gqa::launch_forward_causal(\n",
            "gqa_attention_forward_kernel",
        ],
        f"{rel(WRAPPER)} forward routing",
        failures,
    )

    backward = extract_function(wrapper, "inline void attention_gqa_backward")
    require_contains(
        wrapper,
        "bool qkvr_uses_tk_tile_rope = false",
        f"{rel(WRAPPER)} backward signature",
        failures,
    )
    require_all(
        backward,
        [
            "llmk::attention_gqa::has_tk_backward",
            "qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(cos)) : nullptr",
            "qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(sin)) : nullptr",
            "gqa_float_grads_to_bf16_kernel",
            "if (cos != nullptr && sin != nullptr)",
            "gqa_permute_backward_rope_kernel",
            "gqa_permute_backward_kernel",
        ],
        f"{rel(WRAPPER)} backward routing",
        failures,
    )
    require_order(
        backward,
        [
            "llmk::attention_gqa::launch_backward_causal(",
            "qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(cos)) : nullptr",
            "qkvr_uses_tk_tile_rope ? llmk::to_bf16(const_cast<floatX*>(sin)) : nullptr",
            "gqa_float_grads_to_bf16_kernel",
            "if (cos != nullptr && sin != nullptr)",
            "gqa_permute_backward_rope_kernel",
            "gqa_permute_backward_kernel",
        ],
        f"{rel(WRAPPER)} backward tile-RoPE gradient routing",
        failures,
    )


def validate_tk_kernel(tk_kernel: str, failures: list[str]) -> None:
    require_all(
        tk_kernel,
        [
            "int q_head_idx = static_cast<int>(blockIdx.y);",
            "int kv_head_idx = q_head_idx / g.hr;",
            "globals g{q_arg, k_arg, v_arg, l_arg, o_arg, cos, sin, T, NH / NKVH};",
            "assert(NH % NKVH == 0);",
            "if (n_q_heads % n_kv_heads != 0)",
            "if (T % fwd_sequence_granularity() != 0)",
            "if (head_dim == 64)",
            "if (head_dim == 128)",
            "return T % llmk::attention::bwd_sequence_granularity() == 0;",
            "rope_tile_forward(q_smem[warpgroupid]",
            "rope_tile_forward(k_smem[kv_idx % K::stages]",
            "launch_forward_causal_rope",
            "launch_backward_causal_gqa",
        ],
        f"{rel(TK_KERNEL)} GQA/TK contract",
        failures,
    )


def validate_tk_backward_kernel(tk_bwd_kernel: str, failures: list[str]) -> None:
    require_all(
        tk_bwd_kernel,
        [
            "globals g{q_arg, k_arg, v_arg, og_arg, qg_arg, kg_arg, vg_arg, l_arg, d_arg, cos, sin, T, NH / NKVH};",
            "rope_tile_forward(q_smem[tic]",
            "rope_tile_forward(k_smem[0]",
            "rope_tile_forward(k_smem[1]",
            "launch_backward_causal_gqa",
        ],
        f"{rel(TK_BWD_KERNEL)} GQA backward contract",
        failures,
    )


def validate_smoke_and_reference(smoke: str, reference: str, failures: list[str]) -> None:
    require_all(
        smoke,
        [
            "run_case(128, false",
            "run_case(256, true",
            'printf("GQA case T=%d backward=%s OK',
            "test_attention_gqa smoke OK",
        ],
        rel(SMOKE),
        failures,
    )
    require_all(
        reference,
        [
            "def materialized_repeat_attention",
            "def grouped_tile_rope_attention",
            "repeat_interleave(shape.n_rep",
            "q_group = q[:, kvh * shape.n_rep",
            "Shape(batch=1, seq_len=128, query_heads=4, kv_heads=2, head_dim=128)",
            "Shape(batch=1, seq_len=256, query_heads=4, kv_heads=2, head_dim=128)",
        ],
        rel(REFERENCE),
        failures,
    )


def main() -> None:
    failures: list[str] = []
    validate_wrapper(WRAPPER.read_text(), failures)
    validate_tk_kernel(TK_KERNEL.read_text(), failures)
    validate_tk_backward_kernel(TK_BWD_KERNEL.read_text(), failures)
    validate_smoke_and_reference(SMOKE.read_text(), REFERENCE.read_text(), failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("GQA source guards OK")


if __name__ == "__main__":
    main()
