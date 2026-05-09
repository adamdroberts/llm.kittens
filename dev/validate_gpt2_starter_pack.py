#!/usr/bin/env python3
"""Host-side validation for GPT-2 starter-pack artifacts.

This catches corrupt or mismatched reference files before the GPU-only
`gpt2_validate` and `test_gpt2cu` gates read them. It does not execute model
code and is not a substitute for the H100 forward/parity checks.
"""

import argparse
import math
import struct
import tempfile
from pathlib import Path


GPT2_MODEL_MAGIC = 20240326
GPT2_STATE_MAGIC = 20240327
GPT2_TOKENIZER_MAGIC = 20240328
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
FP32_VERSION = 3
BF16_VERSION = 5
DEBUG_STATE_VERSION = 2
BF16_BYTES = 2
FP32_BYTES = 4


def read_int_header(path):
    path = Path(path)
    with path.open("rb") as f:
        data = f.read(HEADER_BYTES)
    if len(data) != HEADER_BYTES:
        raise ValueError(f"{path} is too small to contain a {HEADER_BYTES}-byte header")
    return list(struct.unpack("<256i", data))


def gpt2_parameter_elements(max_seq_len, padded_vocab_size, num_layers, channels):
    c = channels
    l = num_layers
    return [
        padded_vocab_size * c,
        max_seq_len * c,
        l * c,
        l * c,
        l * (3 * c) * c,
        l * (3 * c),
        l * c * c,
        l * c,
        l * c,
        l * c,
        l * (4 * c) * c,
        l * (4 * c),
        l * c * (4 * c),
        l * c,
        c,
        c,
    ]


def validate_model(path, expected_version=None, expected_config=None):
    path = Path(path)
    header = read_int_header(path)
    if header[0] != GPT2_MODEL_MAGIC:
        raise ValueError(f"bad GPT-2 model magic for {path}: got {header[0]}, expected {GPT2_MODEL_MAGIC}")
    version = header[1]
    if version not in (FP32_VERSION, BF16_VERSION):
        raise ValueError(f"bad GPT-2 model version for {path}: got {version}")
    if expected_version is not None and version != expected_version:
        raise ValueError(f"bad GPT-2 model version for {path}: got {version}, expected {expected_version}")

    config = {
        "max_seq_len": header[2],
        "vocab_size": header[3],
        "num_layers": header[4],
        "num_heads": header[5],
        "channels": header[6],
        "padded_vocab_size": header[7],
    }
    for key, value in config.items():
        if value <= 0:
            raise ValueError(f"invalid GPT-2 model config {key}={value} in {path}")
    if config["channels"] % config["num_heads"] != 0:
        raise ValueError(f"channels must divide heads in {path}: {config}")
    if config["padded_vocab_size"] < config["vocab_size"]:
        raise ValueError(f"padded_vocab_size must be >= vocab_size in {path}: {config}")
    if expected_config is not None and config != expected_config:
        raise ValueError(f"config mismatch for {path}: got {config}, expected {expected_config}")

    param_elements = gpt2_parameter_elements(
        config["max_seq_len"],
        config["padded_vocab_size"],
        config["num_layers"],
        config["channels"],
    )
    num_parameters = sum(param_elements)
    element_bytes = BF16_BYTES if version == BF16_VERSION else FP32_BYTES
    expected_bytes = HEADER_BYTES + num_parameters * element_bytes
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"bad GPT-2 model size for {path}: got {actual_bytes}, expected {expected_bytes}")

    return {
        "path": path,
        "version": version,
        "config": config,
        "param_elements": param_elements,
        "num_parameters": num_parameters,
        "bytes": actual_bytes,
    }


def validate_tokenizer(path, expected_vocab_size):
    path = Path(path)
    with path.open("rb") as f:
        header_data = f.read(HEADER_BYTES)
        if len(header_data) != HEADER_BYTES:
            raise ValueError(f"{path} is too small to contain a tokenizer header")
        header = struct.unpack("<256I", header_data)
        if header[0] != GPT2_TOKENIZER_MAGIC:
            raise ValueError(f"bad tokenizer magic for {path}: got {header[0]}, expected {GPT2_TOKENIZER_MAGIC}")
        version = header[1]
        vocab_size = header[2]
        if version not in (1, 2):
            raise ValueError(f"bad tokenizer version for {path}: got {version}")
        if vocab_size != expected_vocab_size:
            raise ValueError(f"tokenizer vocab mismatch for {path}: got {vocab_size}, expected {expected_vocab_size}")
        if version == 2 and header[3] >= vocab_size:
            raise ValueError(f"tokenizer EOT token out of range for {path}: {header[3]}")

        total_token_bytes = 0
        for token_id in range(vocab_size):
            length_data = f.read(1)
            if len(length_data) != 1:
                raise ValueError(f"tokenizer ended before token {token_id}")
            length = length_data[0]
            if length <= 0:
                raise ValueError(f"tokenizer token {token_id} has invalid length {length}")
            token = f.read(length)
            if len(token) != length:
                raise ValueError(f"tokenizer ended inside token {token_id}")
            total_token_bytes += length
        trailing = f.read(1)
        if trailing:
            raise ValueError(f"tokenizer {path} has trailing bytes after {vocab_size} tokens")

    print(
        f"validated {path}: version={version} vocab={vocab_size} "
        f"token_bytes={total_token_bytes} bytes={path.stat().st_size}"
    )


def read_float_at(f, offset):
    f.seek(offset)
    data = f.read(4)
    if len(data) != 4:
        raise ValueError(f"could not read float at offset {offset}")
    return struct.unpack("<f", data)[0]


def validate_float_samples(f, base_offset, count, label):
    if count == 0:
        return
    for idx in sorted({0, count // 2, count - 1}):
        value = read_float_at(f, base_offset + idx * FP32_BYTES)
        if not math.isfinite(value):
            raise ValueError(f"{label} sample {idx} is not finite: {value}")


def validate_debug_state(path, model_info):
    path = Path(path)
    header = read_int_header(path)
    if header[0] != GPT2_STATE_MAGIC:
        raise ValueError(f"bad GPT-2 debug-state magic for {path}: got {header[0]}, expected {GPT2_STATE_MAGIC}")
    if header[1] != DEBUG_STATE_VERSION:
        raise ValueError(f"bad GPT-2 debug-state version for {path}: got {header[1]}, expected {DEBUG_STATE_VERSION}")

    b = header[2]
    t = header[3]
    config = model_info["config"]
    vocab_size = config["vocab_size"]
    if b <= 0 or t <= 0:
        raise ValueError(f"invalid debug-state batch/sequence values: B={b} T={t}")
    if t > config["max_seq_len"]:
        raise ValueError(f"debug-state sequence length T={t} exceeds model max_seq_len={config['max_seq_len']}")

    bt = b * t
    x_offset = HEADER_BYTES
    y_offset = x_offset + bt * 4
    logits_offset = y_offset + bt * 4
    loss_offset = logits_offset + bt * vocab_size * FP32_BYTES
    grads_offset = loss_offset + FP32_BYTES
    expected_bytes = grads_offset + model_info["num_parameters"] * FP32_BYTES
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"bad GPT-2 debug-state size for {path}: got {actual_bytes}, expected {expected_bytes}")

    with path.open("rb") as f:
        f.seek(x_offset)
        token_data = f.read(bt * 4 * 2)
        if len(token_data) != bt * 4 * 2:
            raise ValueError(f"could not read debug-state x/y tokens from {path}")
        tokens = struct.unpack(f"<{bt * 2}i", token_data)
        for token_idx, token in enumerate(tokens):
            if token < 0 or token >= vocab_size:
                group = "x" if token_idx < bt else "y"
                pos = token_idx if token_idx < bt else token_idx - bt
                raise ValueError(f"debug-state {group}[{pos}] token {token} outside vocab {vocab_size}")

        expected_loss = read_float_at(f, loss_offset)
        if not math.isfinite(expected_loss) or expected_loss <= 0.0:
            raise ValueError(f"invalid debug-state expected_loss: {expected_loss}")
        validate_float_samples(f, logits_offset, bt * vocab_size, "debug-state logits")
        validate_float_samples(f, grads_offset, model_info["num_parameters"], "debug-state gradients")

    print(
        f"validated {path}: B={b} T={t} logits={bt * vocab_size} "
        f"grad_floats={model_info['num_parameters']} expected_loss={expected_loss:.9f} bytes={actual_bytes}"
    )


def pack_int_header(values, unsigned=False):
    header = [0] * HEADER_INTS
    for idx, value in enumerate(values):
        header[idx] = value
    code = "I" if unsigned else "i"
    return struct.pack(f"<{HEADER_INTS}{code}", *header)


def write_model(path, version, config):
    param_elements = gpt2_parameter_elements(
        config["max_seq_len"],
        config["padded_vocab_size"],
        config["num_layers"],
        config["channels"],
    )
    num_parameters = sum(param_elements)
    element_bytes = BF16_BYTES if version == BF16_VERSION else FP32_BYTES
    header = pack_int_header(
        [
            GPT2_MODEL_MAGIC,
            version,
            config["max_seq_len"],
            config["vocab_size"],
            config["num_layers"],
            config["num_heads"],
            config["channels"],
            config["padded_vocab_size"],
        ]
    )
    Path(path).write_bytes(header + bytes(num_parameters * element_bytes))


def write_tokenizer(path, vocab_size):
    header = pack_int_header([GPT2_TOKENIZER_MAGIC, 2, vocab_size, vocab_size - 1], unsigned=True)
    tokens = bytearray()
    for token_id in range(vocab_size):
        token = bytes([65 + token_id % 26])
        tokens.append(len(token))
        tokens.extend(token)
    Path(path).write_bytes(header + tokens)


def write_debug_state(path, model_info, *, bad_token=False):
    b = 1
    t = min(2, model_info["config"]["max_seq_len"])
    bt = b * t
    vocab_size = model_info["config"]["vocab_size"]
    x = list(range(bt))
    y = list(range(1, bt + 1))
    if bad_token:
        y[-1] = vocab_size
    token_payload = struct.pack(f"<{bt * 2}i", *(x + y))
    logits = struct.pack(f"<{bt * vocab_size}f", *([0.0] * (bt * vocab_size)))
    loss = struct.pack("<f", 1.25)
    grads = struct.pack(f"<{model_info['num_parameters']}f", *([0.0] * model_info["num_parameters"]))
    header = pack_int_header([GPT2_STATE_MAGIC, DEBUG_STATE_VERSION, b, t])
    Path(path).write_bytes(header + token_payload + logits + loss + grads)


def expect_failure(label, func):
    try:
        func()
    except ValueError:
        return
    raise AssertionError(f"{label} was accepted unexpectedly")


def run_self_test():
    config = {
        "max_seq_len": 4,
        "vocab_size": 8,
        "num_layers": 1,
        "num_heads": 2,
        "channels": 4,
        "padded_vocab_size": 8,
    }
    with tempfile.TemporaryDirectory(prefix="llmkittens_gpt2_starter_") as tmp:
        tmpdir = Path(tmp)
        bf16_path = tmpdir / "gpt2_tiny_bf16.bin"
        fp32_path = tmpdir / "gpt2_tiny_fp32.bin"
        tokenizer_path = tmpdir / "gpt2_tiny_tokenizer.bin"
        debug_path = tmpdir / "gpt2_tiny_debug_state.bin"
        bad_debug_path = tmpdir / "gpt2_bad_debug_state.bin"

        write_model(bf16_path, BF16_VERSION, config)
        write_model(fp32_path, FP32_VERSION, config)
        write_tokenizer(tokenizer_path, config["vocab_size"])

        bf16 = validate_model(bf16_path, expected_version=BF16_VERSION)
        fp32 = validate_model(fp32_path, expected_version=FP32_VERSION, expected_config=bf16["config"])
        if fp32["num_parameters"] != bf16["num_parameters"]:
            raise AssertionError("synthetic fp32/bf16 parameter counts differ")
        validate_tokenizer(tokenizer_path, bf16["config"]["vocab_size"])

        write_debug_state(debug_path, bf16)
        validate_debug_state(debug_path, bf16)

        write_debug_state(bad_debug_path, bf16, bad_token=True)
        expect_failure("out-of-range debug token", lambda: validate_debug_state(bad_debug_path, bf16))
        expect_failure("tokenizer vocab mismatch", lambda: validate_tokenizer(tokenizer_path, config["vocab_size"] + 1))

    print("GPT-2 starter-pack self-test OK")


def main():
    parser = argparse.ArgumentParser(description="Validate GPT-2 starter-pack files without CUDA")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic pass/fail starter-pack validation smoke")
    parser.add_argument("--tokenizer", default="gpt2_tokenizer.bin")
    parser.add_argument("--fp32-model", default="gpt2_124M.bin")
    parser.add_argument("--bf16-model", default="gpt2_124M_bf16.bin")
    parser.add_argument("--debug-state", default="gpt2_124M_debug_state.bin")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    bf16 = validate_model(args.bf16_model, expected_version=BF16_VERSION)
    fp32 = validate_model(args.fp32_model, expected_version=FP32_VERSION, expected_config=bf16["config"])
    if fp32["num_parameters"] != bf16["num_parameters"]:
        raise ValueError("fp32 and bf16 GPT-2 checkpoints disagree on parameter count")
    validate_tokenizer(args.tokenizer, bf16["config"]["vocab_size"])
    validate_debug_state(args.debug_state, bf16)
    print(
        "GPT-2 starter-pack metadata OK: "
        f"params={bf16['num_parameters']} vocab={bf16['config']['vocab_size']} "
        f"padded_vocab={bf16['config']['padded_vocab_size']}"
    )


if __name__ == "__main__":
    main()
