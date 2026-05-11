"""
Local backend: builds every kernel binary on the dev box (whatever DEVICE_ARCH
the user has exported, default SM90 per the Makefile) and executes them via
plain subprocess. The build step is amortised across the session.
"""
from __future__ import annotations

import os
import subprocess
import time

from .base import KernelRunner, RunResult, REPO_ROOT


class LocalRunner(KernelRunner):
    def __init__(self) -> None:
        self._built = False

    def build_all(self) -> None:
        if self._built:
            return
        env = os.environ.copy()
        # Build both tiers in one make invocation so we don't re-run the
        # nvcc info banner twice. parity-kernels is non-fatal if llm.c isn't
        # checked out — caller will only get a useful error if a parity test
        # actually tries to run a missing probe.
        cmd = ["make", "-j", "test-kernels", "parity-kernels"]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"`make test-kernels parity-kernels` failed (exit {proc.returncode}). "
                f"DEVICE_ARCH={env.get('DEVICE_ARCH', 'SM90')}.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        self._built = True

    def run(self, name: str) -> RunResult:
        return self._exec(name, [])

    def parity_run(self, name: str, *args: str) -> RunResult:
        return self._exec(name, list(args))

    def _exec(self, name: str, extra_argv: list[str]) -> RunResult:
        binary = REPO_ROOT / name
        if not binary.exists():
            raise FileNotFoundError(f"kernel binary not built: {binary}")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(binary), *extra_argv],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        dt = time.perf_counter() - t0
        result = RunResult(
            name=name,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=dt,
        )
        self.write_log(name, result.stdout, result.stderr)
        return result
