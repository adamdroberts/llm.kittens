#!/usr/bin/env python3
"""Source-level guards for CUDA/NCCL/ZeRO contracts that are easy to regress."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "llmc" / "zero.cuh",
    ROOT / "train_gpt2.cu",
    ROOT / "train_llama3.cu",
]
TRAINERS = [
    ROOT / "train_gpt2.cu",
    ROOT / "train_llama3.cu",
]


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def split_top_level_args(args: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(args):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            result.append(args[start:i].strip())
            start = i + 1
    result.append(args[start:].strip())
    return result


def iter_calls(text: str, name: str) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    cursor = 0
    needle = f"{name}("
    while True:
        start = text.find(needle, cursor)
        if start == -1:
            return calls
        args_start = start + len(needle)
        depth = 1
        i = args_start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            raise AssertionError(f"unterminated {name} call near byte {start}")
        calls.append((start, text[args_start : i - 1]))
        cursor = i


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def extract_function_block(text: str, name: str) -> str:
    start = text.find(name)
    if start == -1:
        raise AssertionError(f"missing function {name}")
    brace = text.find("{", start)
    if brace == -1:
        raise AssertionError(f"missing body for function {name}")
    depth = 1
    i = brace + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unterminated function {name}")
    return text[brace + 1 : i - 1]


def extract_function_definition_block(text: str, name: str) -> str:
    cursor = 0
    needle = f"{name}("
    while True:
        start = text.find(needle, cursor)
        if start == -1:
            raise AssertionError(f"missing function definition {name}")
        args_start = start + len(needle)
        depth = 1
        i = args_start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            raise AssertionError(f"unterminated {name} signature near byte {start}")
        j = i
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 1
            k = j + 1
            while k < len(text) and depth > 0:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            if depth != 0:
                raise AssertionError(f"unterminated function {name}")
            return text[j + 1 : k - 1]
        cursor = i


def require_compact_snippet(text: str, snippet: str, rel: Path, description: str) -> str | None:
    if compact(snippet) not in compact(text):
        return f"{rel} missing {description}: {snippet}"
    return None


def validate_nccl_allreduce_counts() -> None:
    failures: list[str] = []
    for path in SOURCES:
        text = path.read_text()
        rel = path.relative_to(ROOT)
        for offset, args_text in iter_calls(text, "ncclAllReduce"):
            args = split_top_level_args(args_text)
            if len(args) != 7:
                failures.append(f"{rel}:{line_for_offset(text, offset)} expected 7 ncclAllReduce args, found {len(args)}")
                continue
            count_arg = args[2]
            if "sizeof" in count_arg:
                failures.append(
                    f"{rel}:{line_for_offset(text, offset)} uses byte size as ncclAllReduce count: {count_arg}"
                )
    if failures:
        raise AssertionError("\n".join(failures))


def validate_nccl_allgather_stream_sync() -> None:
    failures: list[str] = []
    for path in SOURCES:
        text = path.read_text()
        rel = path.relative_to(ROOT)
        for offset, _args_text in iter_calls(text, "ncclAllGather"):
            sync_window = text[max(0, offset - 500) : offset]
            if "multi_gpu_sync_nccl_stream_from_compute" not in sync_window:
                failures.append(
                    f"{rel}:{line_for_offset(text, offset)} ncclAllGather lacks preceding compute->NCCL stream sync"
                )
    if failures:
        raise AssertionError("\n".join(failures))


def validate_zero3_runtime_guard_order() -> None:
    failures: list[str] = []
    for path in TRAINERS:
        text = path.read_text()
        rel = path.relative_to(ROOT)
        guard_calls = [
            (offset, args)
            for offset, args in iter_calls(text, "validate_zero_runtime_request")
            if args.strip() == "zero_stage"
        ]
        init_calls = iter_calls(text, "multi_gpu_config_init")
        dry_offset = text.find("max_steps == 0")

        if len(guard_calls) != 1:
            failures.append(
                f"{rel} expected exactly one validate_zero_runtime_request(zero_stage) call, found {len(guard_calls)}"
            )
            continue
        if len(init_calls) != 1:
            failures.append(f"{rel} expected exactly one multi_gpu_config_init call, found {len(init_calls)}")
            continue
        if dry_offset == -1:
            failures.append(f"{rel} lacks max_steps == 0 host-only dry-run branch before runtime init")
            continue

        guard_offset = guard_calls[0][0]
        init_offset = init_calls[0][0]
        if guard_offset < dry_offset:
            failures.append(
                f"{rel}:{line_for_offset(text, guard_offset)} ZeRO runtime guard blocks host-only dry-run branch"
            )
        if guard_offset > init_offset:
            failures.append(
                f"{rel}:{line_for_offset(text, guard_offset)} ZeRO runtime guard must run before multi_gpu_config_init"
            )

    if failures:
        raise AssertionError("\n".join(failures))


def validate_zero3_runtime_guard_contract() -> None:
    path = ROOT / "llmc" / "zero.cuh"
    text = path.read_text()
    rel = path.relative_to(ROOT)
    block = extract_function_block(text, "validate_zero_runtime_request")
    failures = [
        failure
        for failure in [
            require_compact_snippet(
                block,
                "validate_zero_stage_request(zero_stage);",
                rel,
                "unsupported-stage validation inside validate_zero_runtime_request",
            ),
            require_compact_snippet(
                text,
                "ZeRO Stage 3: parameter shards + runtime all-gather compute layout",
                rel,
                "ZeRO-3 runtime support banner",
            ),
        ]
        if failure is not None
    ]
    forbidden = [
        "ZeRO-3 host-only layout validation is available with -x 0",
        "Refusing to launch a broken ZeRO-3 runtime.",
    ]
    for snippet in forbidden:
        if snippet in text:
            failures.append(f"{rel} still contains stale ZeRO-3 fail-fast text: {snippet}")
    if failures:
        raise AssertionError("\n".join(failures))


def validate_zero3_parameter_shard_contract() -> None:
    required_snippets = {
        ROOT / "llmc" / "zero.cuh": [
            (
                "void zero_all_gather_parameter_shards_to_full",
                "shared ZeRO-3 parameter all-gather helper",
            ),
            (
                "zero_copy_strided_kernel",
                "strided BF16 shard copy helper",
            ),
            (
                "ncclAllGather(src_shards + l * shard_stride",
                "ZeRO-3 NCCL all-gather from parameter shards",
            ),
        ],
        ROOT / "train_gpt2.cu": [
            (
                "void* param_shards_memory; // ZeRO-3 authoritative local parameter shard, BF16",
                "GPT ZeRO-3 authoritative shard buffer",
            ),
            (
                "gpt2_allocate_zero3_parameter_shards(&model, &multi_gpu_config);",
                "GPT runtime initializes ZeRO-3 parameter shards",
            ),
            (
                "? (floatX*)model->param_shards_memory + local_offset_partial",
                "GPT AdamW updates local ZeRO-3 parameter shard",
            ),
            (
                "floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;",
                "GPT AdamW reads reduce-scattered gradient shard in full gradient layout",
            ),
            (
                "zero_all_gather_parameter_shards_to_full(",
                "GPT all-gathers ZeRO-3 parameter shards back to full compute layout",
            ),
        ],
        ROOT / "train_llama3.cu": [
            (
                "void* param_shards_memory; // ZeRO-3 authoritative local parameter shard, BF16",
                "Llama ZeRO-3 authoritative shard buffer",
            ),
            (
                "llama_allocate_zero3_parameter_shards(&model, &multi_gpu_config);",
                "Llama runtime initializes ZeRO-3 parameter shards",
            ),
            (
                "? (floatX*)model->param_shards_memory + local_offset_partial",
                "Llama AdamW updates local ZeRO-3 parameter shard",
            ),
            (
                "floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;",
                "Llama AdamW reads reduce-scattered gradient shard in full gradient layout",
            ),
            (
                "zero_all_gather_parameter_shards_to_full(",
                "Llama all-gathers ZeRO-3 parameter shards back to full compute layout",
            ),
        ],
    }
    failures: list[str] = []
    for path, snippets in required_snippets.items():
        text = path.read_text()
        rel = path.relative_to(ROOT)
        for snippet, description in snippets:
            failure = require_compact_snippet(text, snippet, rel, description)
            if failure is not None:
                failures.append(failure)
    if failures:
        raise AssertionError("\n".join(failures))


def validate_zero3_update_sync_contract() -> None:
    update_functions = {
        ROOT / "train_gpt2.cu": "gpt2_update",
        ROOT / "train_llama3.cu": "llama_update",
    }
    failures: list[str] = []
    for path, function_name in update_functions.items():
        text = path.read_text()
        rel = path.relative_to(ROOT)
        block = extract_function_definition_block(text, function_name)
        gather_offset = block.find("zero_all_gather_parameter_shards_to_full(")
        sync_offset = block.rfind("cudaCheck(cudaDeviceSynchronize());")
        if gather_offset == -1:
            failures.append(f"{rel}:{function_name} missing ZeRO-3 parameter all-gather")
            continue
        if sync_offset == -1:
            failures.append(f"{rel}:{function_name} missing post-update cudaDeviceSynchronize")
            continue
        if sync_offset < gather_offset:
            failures.append(
                f"{rel}:{function_name} synchronizes before ZeRO-3 all-gather instead of after it"
            )
    if failures:
        raise AssertionError("\n".join(failures))


def main() -> None:
    validate_nccl_allreduce_counts()
    validate_nccl_allgather_stream_sync()
    validate_zero3_runtime_guard_order()
    validate_zero3_runtime_guard_contract()
    validate_zero3_parameter_shard_contract()
    validate_zero3_update_sync_contract()
    print("NCCL/ZeRO source guards OK")


if __name__ == "__main__":
    main()
