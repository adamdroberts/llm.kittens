"""
Pytest fixtures for the llm.kittens per-kernel accuracy harness.

The session-scoped `kernel_runner` fixture chooses a backend based on the
USE_RUNPOD_FLASH environment variable:

    USE_RUNPOD_FLASH=1   → tests.runners.flash.FlashRunner (remote H100)
    anything else        → tests.runners.local.LocalRunner (build + run on the
                            currently attached GPU; honours DEVICE_ARCH)

Both backends present the same `KernelRunner` interface (see runners/base.py),
so the per-kernel `tests/kernels/test_*.py` files don't care which one they
get. Build is done once per session in the fixture's setup phase.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def kernel_runner():
    use_flash = os.environ.get("USE_RUNPOD_FLASH", "0") == "1"
    if use_flash:
        # Imported lazily so a developer without runpod-flash installed can
        # still run the local-only path.
        from tests.runners.flash import FlashRunner
        runner = FlashRunner()
    else:
        from tests.runners.local import LocalRunner
        runner = LocalRunner()

    runner.build_all()
    yield runner
    runner.close()
