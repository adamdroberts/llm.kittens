"""
Common interface for kernel-test runners.

Each backend (local subprocess, remote runpod-flash) exposes:

    build_all()        — compile every test_* binary listed in KERNEL_BINARIES
                          via `make -j test-kernels`. Called once per session.

    run(name)          — execute one binary and return a RunResult. Also writes
                          `<name>.log` to the repo root so the existing bash
                          validation harness (scripts/validate_goal_h100.sh)
                          can replay via SMOKE_VALIDATE_ONLY=1.

    close()            — release resources (a no-op for local; for flash this
                          allows the worker to scale down).
"""
from __future__ import annotations

import abc
import pathlib
from dataclasses import dataclass


# Single source of truth: every binary the harness knows how to drive. Must
# match the `test-kernels` aggregate target in the Makefile.
KERNEL_BINARIES: tuple[str, ...] = (
    "test_matmul",
    "test_attention",
    "test_attention_gqa",
    "test_layernorm",
    "test_rmsnorm",
    "test_rope",
    "test_swiglu",
    "test_gelu",
    "test_fused_classifier",
    "test_encoder",
    "test_adamw",
    "test_global_norm",
)

# Repo root resolves the same on both backends because tests/ lives at the top.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class RunResult:
    name: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and f"{self.name} smoke OK" in self.stdout


class KernelRunner(abc.ABC):
    @abc.abstractmethod
    def build_all(self) -> None: ...

    @abc.abstractmethod
    def run(self, name: str) -> RunResult: ...

    @abc.abstractmethod
    def parity_run(self, name: str, *args: str) -> RunResult:
        """Run a parity probe binary with extra argv (typically an iodir path).

        Distinct from `run()` because parity probes need argv (the iodir to
        read inputs from / write outputs to) and, on the flash backend, also
        need round-tripping of .npy files in/out of the worker.
        """
        ...

    def close(self) -> None:
        return None

    @staticmethod
    def write_log(name: str, stdout: str, stderr: str) -> None:
        # Mirror the binary stdout into <name>.log at the repo root so that
        # scripts/validate_goal_h100.sh's SMOKE_VALIDATE_ONLY=1 / SMOKE_LOG_DIR
        # replay path can grep for "<name> smoke OK" without re-executing.
        log_path = REPO_ROOT / f"{name}.log"
        log_path.write_text(stdout + (("\n--- STDERR ---\n" + stderr) if stderr else ""))
