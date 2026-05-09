#!/usr/bin/env python3
"""Validate prepared training and eval data artifacts without CUDA.

The C++ DataLoader/EvalLoader validate headers at runtime. This script catches
bad magic/version pairs, truncated files, bad token widths, and malformed
HellaSwag-style examples earlier in the H100 validation flow.
"""

import argparse
import glob
import struct
import tempfile
from pathlib import Path


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4

TRAIN_FORMATS = {
    20240520: {
        "name": "gpt-2",
        "version": 1,
        "token_bytes": 2,
        "max_token": 50256,
    },
    20240801: {
        "name": "llama-3",
        "version": 7,
        "token_bytes": 4,
        "max_token": 128255,
    },
}

EVAL_FORMATS = {
    20240522: {
        "name": "gpt-2-eval",
        "version": 1,
        "token_bytes": 2,
        "start_example": 2**16 - 1,
        "max_token": 50256,
    },
    20240802: {
        "name": "llama-3-eval",
        "version": 7,
        "token_bytes": 4,
        "start_example": 2**32 - 1,
        "max_token": 128255,
    },
}

DEFAULT_PATTERNS = (
    "dev/data/tinyshakespeare/*.bin",
    "dev/data/tinystories/*.bin",
    "dev/data/fineweb10B/*.bin",
    "dev/data/fineweb100B/*.bin",
    "dev/data/edu_fineweb10B/*.bin",
    "dev/data/edu_fineweb100B/*.bin",
    "dev/data/hellaswag/*.bin",
    "fineweb100B/*.bin",
    "edu_fineweb100B/*.bin",
)


def read_header(path):
    with path.open("rb") as f:
        data = f.read(HEADER_BYTES)
    if len(data) != HEADER_BYTES:
        raise ValueError(f"{path} is too small to contain a {HEADER_BYTES}-byte header")
    return list(struct.unpack("<256i", data))


def token_struct(token_bytes):
    if token_bytes == 2:
        return "<H"
    if token_bytes == 4:
        return "<I"
    raise ValueError(f"unsupported token width: {token_bytes}")


def unpack_tokens(data, token_bytes):
    if len(data) % token_bytes != 0:
        raise ValueError(f"token payload byte count {len(data)} is not divisible by {token_bytes}")
    count = len(data) // token_bytes
    if count == 0:
        return ()
    fmt = f"<{count}{'H' if token_bytes == 2 else 'I'}"
    return struct.unpack(fmt, data)


def sample_indices(count, sample_tokens):
    if count <= 0 or sample_tokens <= 0:
        return []
    if count <= sample_tokens * 3:
        return list(range(count))
    windows = [
        range(0, sample_tokens),
        range(max(0, count // 2 - sample_tokens // 2), min(count, count // 2 + sample_tokens // 2)),
        range(max(0, count - sample_tokens), count),
    ]
    return sorted(set(idx for window in windows for idx in window))


def check_token_range(path, tokens, max_token, label):
    for idx, token in enumerate(tokens):
        if token > max_token:
            raise ValueError(f"{path} {label}[{idx}] token {token} exceeds max token {max_token}")


def validate_train(path, header, fmt, args):
    version = header[1]
    if version != fmt["version"]:
        raise ValueError(f"{path} has {fmt['name']} version {version}, expected {fmt['version']}")

    num_tokens = header[2]
    if num_tokens <= 0:
        raise ValueError(f"{path} has invalid token count {num_tokens}")

    token_bytes = fmt["token_bytes"]
    expected_size = HEADER_BYTES + num_tokens * token_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} size mismatch: got {actual_size}, expected {expected_size}")

    if args.min_train_tokens and num_tokens < args.min_train_tokens:
        raise ValueError(
            f"{path} has {num_tokens} tokens, fewer than --min-train-tokens {args.min_train_tokens}"
        )
    if args.batch_size and args.sequence_length and args.processes:
        required = args.batch_size * args.sequence_length * args.processes + 1
        if num_tokens < required:
            raise ValueError(
                f"{path} has {num_tokens} tokens, fewer than B*T*processes+1={required}"
            )

    validate_train_tokens(path, num_tokens, fmt, args)
    print(
        f"validated train {path}: format={fmt['name']} tokens={num_tokens} "
        f"token_bytes={token_bytes} bytes={actual_size}"
    )


def validate_train_tokens(path, num_tokens, fmt, args):
    token_bytes = fmt["token_bytes"]
    max_token = fmt["max_token"]
    unpack_one = struct.Struct(token_struct(token_bytes)).unpack
    with path.open("rb") as f:
        if args.full_token_scan:
            f.seek(HEADER_BYTES)
            remaining = num_tokens
            chunk_tokens = max(1, args.chunk_tokens)
            offset = 0
            while remaining:
                count = min(remaining, chunk_tokens)
                data = f.read(count * token_bytes)
                if len(data) != count * token_bytes:
                    raise ValueError(f"{path} ended while scanning tokens")
                check_token_range(path, unpack_tokens(data, token_bytes), max_token, f"token@{offset}")
                offset += count
                remaining -= count
        else:
            for idx in sample_indices(num_tokens, args.sample_tokens):
                f.seek(HEADER_BYTES + idx * token_bytes)
                data = f.read(token_bytes)
                if len(data) != token_bytes:
                    raise ValueError(f"{path} ended while sampling token {idx}")
                token = unpack_one(data)[0]
                if token > max_token:
                    raise ValueError(f"{path} token[{idx}]={token} exceeds max token {max_token}")


def validate_eval(path, header, fmt, args):
    version = header[1]
    if version != fmt["version"]:
        raise ValueError(f"{path} has {fmt['name']} version {version}, expected {fmt['version']}")

    num_examples = header[2]
    longest_example_bytes = header[3]
    if num_examples <= 0:
        raise ValueError(f"{path} has invalid example count {num_examples}")
    if longest_example_bytes <= 0:
        raise ValueError(f"{path} has invalid longest_example_bytes {longest_example_bytes}")

    token_bytes = fmt["token_bytes"]
    prefix_bytes = token_bytes * 2
    max_seen_bytes = 0
    with path.open("rb") as f:
        f.seek(HEADER_BYTES)
        for expected_index in range(num_examples):
            prefix = f.read(prefix_bytes)
            if len(prefix) != prefix_bytes:
                raise ValueError(f"{path} ended before example {expected_index}")
            start_example, example_bytes = unpack_tokens(prefix, token_bytes)
            if start_example != fmt["start_example"]:
                raise ValueError(
                    f"{path} example {expected_index} has bad delimiter {start_example}, "
                    f"expected {fmt['start_example']}"
                )
            if example_bytes <= prefix_bytes or example_bytes % token_bytes != 0:
                raise ValueError(f"{path} example {expected_index} has bad byte count {example_bytes}")
            if example_bytes > longest_example_bytes:
                raise ValueError(
                    f"{path} example {expected_index} is {example_bytes} bytes, "
                    f"longer than header longest {longest_example_bytes}"
                )
            rest = f.read(example_bytes - prefix_bytes)
            if len(rest) != example_bytes - prefix_bytes:
                raise ValueError(f"{path} ended inside example {expected_index}")
            tokens = (start_example, example_bytes) + unpack_tokens(rest, token_bytes)
            validate_eval_example(path, expected_index, tokens, fmt, args)
            max_seen_bytes = max(max_seen_bytes, example_bytes)
        trailing = f.read(1)
        if trailing:
            raise ValueError(f"{path} has trailing bytes after {num_examples} examples")

    if max_seen_bytes != longest_example_bytes:
        raise ValueError(
            f"{path} header longest_example_bytes={longest_example_bytes}, "
            f"but max parsed example is {max_seen_bytes}"
        )
    print(
        f"validated eval {path}: format={fmt['name']} examples={num_examples} "
        f"longest_example_bytes={longest_example_bytes} bytes={path.stat().st_size}"
    )


def validate_eval_example(path, expected_index, tokens, fmt, args):
    token_bytes = fmt["token_bytes"]
    example_bytes = tokens[1]
    if len(tokens) * token_bytes != example_bytes:
        raise ValueError(f"{path} example {expected_index} token count does not match example_bytes")
    if len(tokens) < 10:
        raise ValueError(f"{path} example {expected_index} is too short")
    if tokens[2] != expected_index:
        raise ValueError(f"{path} example index mismatch: got {tokens[2]}, expected {expected_index}")

    label = tokens[3]
    num_completions = tokens[4]
    if num_completions <= 0:
        raise ValueError(f"{path} example {expected_index} has no completions")
    if args.expected_completions and num_completions != args.expected_completions:
        raise ValueError(
            f"{path} example {expected_index} has {num_completions} completions, "
            f"expected {args.expected_completions}"
        )
    if label >= num_completions:
        raise ValueError(f"{path} example {expected_index} label {label} outside {num_completions} completions")

    pos = 5
    ctx_len = tokens[pos]
    pos += 1
    if ctx_len <= 0:
        raise ValueError(f"{path} example {expected_index} has empty context")
    if pos + ctx_len > len(tokens):
        raise ValueError(f"{path} example {expected_index} context overruns example")
    check_token_range(path, tokens[pos : pos + ctx_len], fmt["max_token"], f"example {expected_index} ctx")
    pos += ctx_len

    for completion_index in range(num_completions):
        if pos >= len(tokens):
            raise ValueError(f"{path} example {expected_index} ended before completion {completion_index}")
        completion_len = tokens[pos]
        pos += 1
        if completion_len <= 0:
            raise ValueError(f"{path} example {expected_index} completion {completion_index} is empty")
        if pos + completion_len > len(tokens):
            raise ValueError(f"{path} example {expected_index} completion {completion_index} overruns example")
        check_token_range(
            path,
            tokens[pos : pos + completion_len],
            fmt["max_token"],
            f"example {expected_index} completion {completion_index}",
        )
        pos += completion_len

    if pos != len(tokens):
        raise ValueError(f"{path} example {expected_index} has {len(tokens) - pos} unused tokens")


def validate_file(path, args):
    header = read_header(path)
    magic = header[0]
    if magic in TRAIN_FORMATS:
        validate_train(path, header, TRAIN_FORMATS[magic], args)
    elif magic in EVAL_FORMATS:
        validate_eval(path, header, EVAL_FORMATS[magic], args)
    else:
        expected = sorted(TRAIN_FORMATS) + sorted(EVAL_FORMATS)
        raise ValueError(f"{path} has unknown data magic {magic}; expected one of {expected}")


def pack_header(magic, version, count, extra=0):
    header = [0] * HEADER_INTS
    header[0] = magic
    header[1] = version
    header[2] = count
    header[3] = extra
    return struct.pack("<256i", *header)


def pack_tokens(tokens, token_bytes):
    if token_bytes == 2:
        return struct.pack(f"<{len(tokens)}H", *tokens)
    if token_bytes == 4:
        return struct.pack(f"<{len(tokens)}I", *tokens)
    raise ValueError(f"unsupported token width: {token_bytes}")


def write_train_artifact(path, magic, tokens):
    fmt = TRAIN_FORMATS[magic]
    path.write_bytes(
        pack_header(magic, fmt["version"], len(tokens))
        + pack_tokens(tokens, fmt["token_bytes"])
    )


def write_eval_artifact(path, magic, examples):
    fmt = EVAL_FORMATS[magic]
    token_bytes = fmt["token_bytes"]
    encoded = bytearray()
    longest = 0
    for example in examples:
        stream = list(example)
        stream[0] = fmt["start_example"]
        stream[1] = len(stream) * token_bytes
        longest = max(longest, stream[1])
        encoded.extend(pack_tokens(stream, token_bytes))
    path.write_bytes(pack_header(magic, fmt["version"], len(examples), longest) + encoded)


def self_test_args():
    return argparse.Namespace(
        min_train_tokens=2,
        batch_size=0,
        sequence_length=0,
        processes=0,
        full_token_scan=True,
        chunk_tokens=3,
        sample_tokens=2,
        expected_completions=4,
    )


def expect_failure(label, func):
    try:
        func()
    except ValueError:
        return
    raise AssertionError(f"{label} was accepted unexpectedly")


def run_self_test():
    args = self_test_args()
    with tempfile.TemporaryDirectory(prefix="llmkittens_data_artifacts_") as tmp:
        tmpdir = Path(tmp)
        gpt_train = tmpdir / "gpt_train.bin"
        llama_train = tmpdir / "llama_train.bin"
        gpt_eval = tmpdir / "gpt_eval.bin"
        llama_eval = tmpdir / "llama_eval.bin"
        bad_train = tmpdir / "bad_train.bin"
        bad_eval = tmpdir / "bad_eval.bin"

        write_train_artifact(gpt_train, 20240520, [10, 11, 12, 13, 14, 15])
        write_train_artifact(llama_train, 20240801, [128000, 128001, 42, 43, 44, 45])

        gpt_example = [
            0, 0, 0, 2, 4,
            2, 100, 101,
            1, 102,
            1, 103,
            1, 104,
            1, 105,
        ]
        llama_example = [
            0, 0, 0, 1, 4,
            2, 128000, 128001,
            1, 128002,
            1, 128003,
            1, 128004,
            1, 128005,
        ]
        write_eval_artifact(gpt_eval, 20240522, [gpt_example])
        write_eval_artifact(llama_eval, 20240802, [llama_example])

        for path in (gpt_train, llama_train, gpt_eval, llama_eval):
            validate_file(path, args)

        write_train_artifact(bad_train, 20240520, [10, 11, 60000])
        expect_failure("out-of-range train token", lambda: validate_file(bad_train, args))

        bad_example = list(gpt_example)
        bad_example[3] = 4
        write_eval_artifact(bad_eval, 20240522, [bad_example])
        expect_failure("bad eval label", lambda: validate_file(bad_eval, args))

    print("Data artifact self-test OK")


def expand_patterns(patterns, strict_missing):
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            message = f"pattern matched no files: {pattern}"
            if strict_missing:
                raise FileNotFoundError(message)
            print(f"skipping {message}")
            continue
        for match in matches:
            path = Path(match)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Validate llm.kittens data .bin artifacts without CUDA")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic pass/fail artifact validation smoke")
    parser.add_argument("--pattern", action="append", default=[], help="Additional glob pattern to validate")
    parser.add_argument("--no-defaults", action="store_true", help="Only validate --pattern globs")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any glob pattern matches no files")
    parser.add_argument("--sample-tokens", type=int, default=64, help="Tokens to sample from start/middle/end of train shards")
    parser.add_argument("--full-token-scan", action="store_true", help="Scan every training token for vocabulary range")
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000, help="Tokens per chunk during --full-token-scan")
    parser.add_argument("--min-train-tokens", type=int, default=2, help="Minimum token count for each training shard")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional DataLoader B for shard-size validation")
    parser.add_argument("--sequence-length", type=int, default=0, help="Optional DataLoader T for shard-size validation")
    parser.add_argument("--processes", type=int, default=0, help="Optional DataLoader process count for shard-size validation")
    parser.add_argument("--expected-completions", type=int, default=4, help="Expected eval completions per example; 0 disables")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    patterns = []
    if not args.no_defaults:
        patterns.extend(DEFAULT_PATTERNS)
    patterns.extend(args.pattern)
    if not patterns:
        raise ValueError("no patterns to validate")

    paths = expand_patterns(patterns, args.strict_missing)
    if not paths:
        print("No data artifacts found; run dev/data preprocessing or pass --pattern to validate a specific file.")
        return

    for path in paths:
        validate_file(path, args)
    print(f"Data artifact metadata OK: files={len(paths)}")


if __name__ == "__main__":
    main()
