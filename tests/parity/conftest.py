"""
Shared helpers for the per-kernel parity tier.

Inputs and outputs cross the Python/CUDA boundary as .npy files. We carry bf16
through NumPy as uint16 (raw bf16 bits) — NumPy has no native bf16 dtype, so
this avoids a fp32-roundtrip that would silently quantize differently than
the kernel does. The probe binaries reinterpret the uint16 buffer as
__nv_bfloat16 on the device.

Both Family A (probes vs llm.c reference probes) and Family B (probes vs
PyTorch reference) live under tests/parity/. They share the `iodir` fixture
that gives each test a clean tmp dir to stage inputs and outputs.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest


@pytest.fixture
def iodir(tmp_path) -> pathlib.Path:
    """Per-test scratch directory for staging .npy inputs and probe outputs."""
    return tmp_path


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even cast from float32 to the 16-bit bf16 bit pattern.

    Returns a uint16 array with the same shape as ``x``. The exponent matches
    fp32; mantissa is truncated from 23 to 7 bits with bankers' rounding.
    Subnormals and NaN/Inf round-trip correctly because we leave the high
    16 bits of fp32 untouched once rounded.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    bits = x.view(np.uint32)
    # round-to-nearest-even: bias = 0x7FFF + LSB of the kept exponent+mantissa
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    """Inverse of fp32_to_bf16_bits — exact (zero-padding the low mantissa)."""
    upcast = (bits.astype(np.uint32) << 16).view(np.float32)
    return upcast.copy()


def save_bf16(path: pathlib.Path, x: np.ndarray) -> None:
    np.save(path, fp32_to_bf16_bits(x))


def save_shape(path: pathlib.Path, *dims: int) -> None:
    np.save(path, np.asarray(dims, dtype=np.int32))


def load_bf16(path: pathlib.Path) -> np.ndarray:
    """Load uint16 .npy and return the bf16-decoded fp32 view."""
    bits = np.load(path)
    if bits.dtype != np.uint16:
        raise TypeError(f"{path} is dtype {bits.dtype}, expected uint16 (bf16-bits)")
    return bf16_bits_to_fp32(bits)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
