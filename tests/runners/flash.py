"""
Flash backend: ships the source tree(s) to a single warm runpod-flash worker
on H100 (GpuGroup.ADA_80_PRO), runs `make test-kernels parity-kernels` once,
then re-executes individual binaries on subsequent calls. Activates only when
USE_RUNPOD_FLASH=1.

Bundling: this trio of sibling source trees is uploaded together so the
worker can build every test:
  - llm.kittens (this repo)
  - ../ThunderKittens (TK headers — required to build any TK-wrapped kernel)
  - ../llm.c        (reference kernels for the parity tier)

We mirror remote stdout into local <name>.log files so the existing bash
harness's SMOKE_VALIDATE_ONLY replay path keeps working. Parity-test
artifacts (.npy) are roundtripped via base64 inside the call payload — keeps
the protocol the same as the smoke tests, no extra storage backend needed.

Live progress: every long-running operation (cold start, tarball upload,
remote build, per-test dispatch) prints a timestamped line to stderr with
flush=True so it shows up under `pytest -s` even when the harness is mid-call.
The flash SDK can stall for minutes on cold start + dependency install — the
user needs to see what's happening rather than stare at a blank pytest line.
"""
from __future__ import annotations

import asyncio
import base64
import io
import pathlib
import sys
import tarfile
import time
from typing import Optional

from runpod_flash import Endpoint, GpuGroup

from .base import KernelRunner, RunResult, REPO_ROOT


def _log(msg: str) -> None:
    """Timestamped progress line to stderr; pytest -s surfaces it live."""
    t = time.strftime("%H:%M:%S")
    print(f"[flash {t}] {msg}", file=sys.stderr, flush=True)


_PARENT = REPO_ROOT.parent  # /mnt/disk2/.../open-source

# Source trees to bundle. Each gets its own top-level directory in the tar.gz
# so the worker reproduces the dev-box layout (`../ThunderKittens`,
# `../llm.c`) and the Makefile's `TK_ROOT`/`LLMC_REF_ROOT` defaults resolve.
_SOURCE_TREES = (
    ("llm.kittens", REPO_ROOT),
    ("ThunderKittens", _PARENT / "ThunderKittens"),
    ("llm.c", _PARENT / "llm.c"),
)

_EXCLUDE_DIRS = {".git", ".flash", "__pycache__", "build", "data",
                 "edu_fineweb100B", "node_modules", ".pytest_cache",
                 "examples", "doc", "figs", ".cache", ".vscode",
                 ".idea", "logs", ".deps", "log124M", "log",
                 "site-packages", "venv", ".venv", "dist"}
# Path.suffix only returns the last extension, so versioned .so.13 / .so.9
# slip through a `.so` filter. We catch them via _EXCLUDE_DIR (.deps where
# CUDA libs land) and via name predicates below.
_EXCLUDE_SUFFIXES = (".bin", ".pyc", ".so", ".o", ".log", ".png", ".jpg",
                     ".pdf", ".weights", ".tar", ".gz", ".pt", ".onnx",
                     ".ckpt", ".npz", ".safetensors", ".json")
_EXCLUDE_NAME_CONTAINS = (".so.",)  # libfoo.so.13, libbar.so.9.4 etc.


def _build_source_tarball() -> bytes:
    """tar.gz the three source trees, excluding data/binaries.

    Returns base64-encoded bytes. Each tree is rooted at its own top-level
    directory, so on extract `/workspace/{llm.kittens,ThunderKittens,llm.c}/...`
    mirrors the dev box. Uses os.walk + dirs.remove() to prune large excluded
    subtrees (notably .git, which is 48 MB on ThunderKittens) at walk time
    rather than visit-then-skip.
    """
    import os

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for top_name, root in _SOURCE_TREES:
            if not root.exists():
                raise RuntimeError(f"flash bundle: source tree missing: {root}")
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune excluded dirs in place — os.walk respects mutation.
                dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
                for fn in filenames:
                    p = pathlib.Path(dirpath) / fn
                    if p.suffix in _EXCLUDE_SUFFIXES:
                        continue
                    if any(s in fn for s in _EXCLUDE_NAME_CONTAINS):
                        continue
                    # Skip locally-built binaries: extensionless executables
                    # at the repo root (test_*, probe_*, train_*cu, etc.).
                    if not p.suffix and os.access(p, os.X_OK):
                        try:
                            if p.parent == root:
                                continue
                        except Exception:
                            pass
                    rel = p.relative_to(root)
                    tar.add(p, arcname=f"{top_name}/{rel}")
    raw = buf.getvalue()
    encoded = base64.b64encode(raw)
    if len(encoded) > 9 * 1024 * 1024:
        raise RuntimeError(
            f"Combined source tarball is {len(encoded)/1024/1024:.1f} MB "
            "encoded, exceeds flash 9 MB headroom. Trim via _EXCLUDE_DIRS / "
            "_EXCLUDE_SUFFIXES in tests/runners/flash.py, or add a git-clone "
            "fallback path."
        )
    return encoded


# Endpoint is built dynamically inside FlashRunner.__init__ so the GPU group
# (and therefore the endpoint name) can come from the RUNPOD_GPU env var.
# Decorating at module-load time would lock us to a single GPU and would
# also reuse the cached endpoint name across runs, which produces "Endpoint
# not found" errors when runpod garbage-collects an entry the SDK still
# thinks exists.
async def _remote_impl(payload: dict) -> dict:
    import base64
    import io
    import os
    import pathlib
    import subprocess
    import tarfile

    workspace = pathlib.Path("/workspace")
    op = payload.get("op")

    if op == "bootstrap":
        # Wipe-and-extract so a re-bootstrap on a reused worker always reflects
        # the local source tree.
        for sub in ("llm.kittens", "ThunderKittens", "llm.c"):
            tgt = workspace / sub
            if tgt.exists():
                subprocess.run(["rm", "-rf", str(tgt)], check=False)
        workspace.mkdir(parents=True, exist_ok=True)
        tar_bytes = base64.b64decode(payload["tarball"])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(workspace)
        kittens = workspace / "llm.kittens"
        env = os.environ.copy()
        env["TK_ROOT"] = str(workspace / "ThunderKittens")
        env["LLMC_REF_ROOT"] = str(workspace / "llm.c")
        # DEVICE_ARCH is forwarded from the client based on the worker's GPU
        # (SM90 on ADA_80_PRO, SM120 on ADA_32_PRO).
        env["DEVICE_ARCH"] = payload.get("device_arch", "SM90")
        proc = subprocess.run(
            ["make", "-j", "test-kernels", "parity-kernels"],
            cwd=kittens, env=env, capture_output=True, text=True,
        )
        return {
            "build_rc": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }

    if op == "run" or op == "parity_run":
        kittens = workspace / "llm.kittens"
        name = payload["name"]
        argv = payload.get("argv", [])
        # If the test ships an iodir bundle, drop it on the worker first.
        iodir_bytes = payload.get("iodir_tarball")
        iodir_path = None
        if iodir_bytes:
            iodir_path = pathlib.Path("/tmp/parity_iodir")
            if iodir_path.exists():
                subprocess.run(["rm", "-rf", str(iodir_path)], check=False)
            iodir_path.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(iodir_bytes)),
                              mode="r:gz") as tar:
                tar.extractall(iodir_path)
            # Replace the iodir argv (assumed to be argv[0]) with the worker path.
            argv = [str(iodir_path)] + argv[1:]

        binary = kittens / name
        if not binary.exists():
            return {"exit_code": 127, "stdout": "",
                    "stderr": f"binary not found: {binary} (bootstrap failed?)"}
        proc = subprocess.run(
            [str(binary), *argv], cwd=kittens,
            capture_output=True, text=True, timeout=300,
        )
        result = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        # If a parity probe wrote outputs into the iodir, ship them back.
        if iodir_path and iodir_path.exists() and proc.returncode == 0:
            out_buf = io.BytesIO()
            with tarfile.open(fileobj=out_buf, mode="w:gz") as tar:
                for sub in ("ref", "tk"):
                    sub_path = iodir_path / sub
                    if sub_path.exists():
                        for p in sub_path.rglob("*"):
                            if p.is_file():
                                tar.add(p, arcname=str(p.relative_to(iodir_path)))
            result["iodir_outputs"] = base64.b64encode(out_buf.getvalue()).decode("ascii")
        return result

    return {"error": f"unknown op: {op}"}


# Default device-arch per GPU group. Extend if you add another GPU.
# Note: this repo only ships TK kernels for Hopper (SM90+). Ampere (SM80 /
# A100) is not supported — every wrapper under llmc/tk/ is *_h100.cuh and
# uses WGMMA + TMA, sm_90+ only.
_GPU_TO_DEVICE_ARCH = {
    "ADA_80_PRO": "SM90",     # H100 (80GB) — primary target
    "HOPPER_141": "SM90",     # H200 — same Hopper ISA
    "ADA_32_PRO": "SM120",    # RTX 5090 (32GB) — Blackwell consumer (CUDA fallback only)
}

# Runpod datacenter IDs across US + EU. Used as the default `locations` so
# the worker can spin up wherever a free GPU is available, instead of being
# pinned to one region (the SDK's default is EU-RO-1, which often hits
# capacity limits for popular GPU types). Override with the
# `RUNPOD_LOCATIONS` env var.
_DEFAULT_LOCATIONS_US_EU = (
    "US-CA-2,US-KS-2,US-NC-1,US-OR-1,US-TX-3,US-IL-1,US-WA-1,"
    "EU-CZ-1,EU-NL-1,EU-RO-1,EU-SE-1,EU-FR-1"
)

# Minimum CUDA version we ask runpod to allocate the worker on. Runpod's
# API enumerates 12.8 as the latest available — passing 13.x is rejected
# with `Value error, '13.0' is not a valid CudaVersion`. Override with
# `RUNPOD_CUDA_VERSIONS` env var (comma-separated string) once 13.x lands.
_DEFAULT_CUDA_VERSIONS = "12.8"


def _apply_locations_and_cuda(endpoint, locations: str, cuda_versions: str) -> None:
    """Inject `locations` and `allowedCudaVersions` into the endpoint's
    underlying serverless config.

    The Endpoint() constructor doesn't expose either as a kwarg in this SDK
    version. The SDK reads `RUNPOD_DEFAULT_LOCATIONS` from the env at
    resource-build time, so for `locations` we set the env var first. CUDA
    has no env var, so we monkey-patch `_build_resource_config` to mutate
    the freshly-built model.
    """
    import os as _os
    _os.environ["RUNPOD_DEFAULT_LOCATIONS"] = locations
    original = endpoint._build_resource_config

    def patched():
        cfg = original()
        # Free-string field; runpod backend validates the version list. If
        # CUDA 13 isn't actually available the deploy will reject — surface
        # that instead of silently falling back.
        if hasattr(cfg, "allowedCudaVersions"):
            cfg.allowedCudaVersions = cuda_versions
        return cfg

    endpoint._build_resource_config = patched


class FlashRunner(KernelRunner):
    def __init__(self) -> None:
        # Auth is the SDK's problem — `flash login` writes to ~/.config/runpod
        # (XDG), not a path we can reliably probe. Let the SDK error out itself
        # if it can't authenticate. Run dev/flash_smoke.py first if you want a
        # 1-minute isolation check before invoking the full parity harness.
        import os
        gpu_name = os.environ.get("RUNPOD_GPU", "ADA_80_PRO")
        if not hasattr(GpuGroup, gpu_name):
            raise RuntimeError(
                f"RUNPOD_GPU='{gpu_name}' is not a valid GpuGroup. "
                "See `runpod_flash.GpuGroup` for valid names "
                "(e.g. ADA_80_PRO for H100, ADA_32_PRO for RTX 5090)."
            )
        self._gpu_name = gpu_name
        self._gpu = getattr(GpuGroup, gpu_name)
        # Forward device arch to the worker's `make`. Override with
        # DEVICE_ARCH=SM... if you want to compile for a different SM than
        # the GPU's default.
        self._device_arch = os.environ.get(
            "DEVICE_ARCH", _GPU_TO_DEVICE_ARCH.get(gpu_name, "SM90")
        )
        locations = os.environ.get("RUNPOD_LOCATIONS", _DEFAULT_LOCATIONS_US_EU)
        cuda_versions = os.environ.get("RUNPOD_CUDA_VERSIONS", _DEFAULT_CUDA_VERSIONS)

        # Stable endpoint name per GPU type — reuses the warm worker across
        # runs (no cold-start, no rebuild). Override with `RUNPOD_ENDPOINT_NAME`
        # if you need a fresh deploy (e.g. to dodge an orphan-template
        # collision). When that happens, clean up via the runpod console or
        # `flash undeploy <name>` then retry.
        gpu_suffix = gpu_name.lower().replace("_", "-")
        default_name = f"llm-kittens-parity-{gpu_suffix}"
        endpoint_name = os.environ.get("RUNPOD_ENDPOINT_NAME", default_name)
        # Set the env var BEFORE constructing the Endpoint so the SDK picks
        # it up when it builds the underlying resource model.
        os.environ["RUNPOD_DEFAULT_LOCATIONS"] = locations
        endpoint = Endpoint(
            name=endpoint_name,
            gpu=self._gpu,
            workers=1,
            idle_timeout=900,
            flashboot=True,
            dependencies=[],
            system_dependencies=["build-essential"],
            execution_timeout_ms=900_000,
        )
        _apply_locations_and_cuda(endpoint, locations, cuda_versions)
        # Keep the endpoint object accessible for debugging/inspection.
        self._endpoint = endpoint
        self._endpoint_name = endpoint_name
        self._remote = endpoint(_remote_impl)

        self._tarball: Optional[bytes] = None
        self._built = False
        self._loop = asyncio.new_event_loop()
        _log(f"FlashRunner: GPU={gpu_name}, DEVICE_ARCH={self._device_arch}")
        _log(f"  endpoint={endpoint_name}")
        _log(f"  locations={locations}")
        _log(f"  cuda_versions={cuda_versions}")

    def _await(self, coro, op_name: str = "op", interval: int = 10):
        """Run a coroutine to completion while emitting a heartbeat to stderr.

        The flash SDK's await is a single opaque blocking call — the worker
        could be cold-starting, apt-installing, or running `make`, and we
        have zero observability inside it. The heartbeat at least proves the
        Python process is alive and tells the user how long they've been
        waiting, so a slow-but-progressing call doesn't look like a hang.
        """
        async def _run():
            task = asyncio.create_task(coro)
            t0 = time.perf_counter()
            while not task.done():
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                except asyncio.TimeoutError:
                    elapsed = int(time.perf_counter() - t0)
                    _log(f"  ... {op_name} still waiting ({elapsed}s elapsed)")
            return task.result()

        return self._loop.run_until_complete(_run())

    def build_all(self) -> None:
        if self._built:
            return
        if self._tarball is None:
            _log("packaging source tree (llm.kittens + ThunderKittens + llm.c)...")
            t0 = time.perf_counter()
            self._tarball = _build_source_tarball()
            _log(f"tarball ready: {len(self._tarball)/1024/1024:.2f} MB encoded "
                 f"in {time.perf_counter()-t0:.1f}s")
        _log(f"dispatching bootstrap to {self._gpu_name} worker (DEVICE_ARCH={self._device_arch}).")
        _log("Stages on first call:")
        _log(f"  1. cold-start {self._gpu_name} (~30-90s)")
        _log("  2. apt install build-essential (~10-30s, cached on warm worker)")
        _log("  3. extract tarball (~1s)")
        _log("  4. make -j test-kernels parity-kernels (~3-8 min, 30+ nvcc binaries)")
        _log("Subsequent tests in this session reuse the warm worker (idle_timeout=900s).")
        _log("If it's been > 15 min with no return, RUN dev/flash_smoke.py to verify "
             "flash itself is alive — that takes ~1 min and isolates flash from our build.")
        t0 = time.perf_counter()
        result = self._await(
            self._remote({
                "op": "bootstrap",
                "tarball": self._tarball.decode("ascii"),
                "device_arch": self._device_arch,
            }),
            op_name="bootstrap",
            interval=15,
        )
        _log(f"bootstrap returned in {time.perf_counter()-t0:.1f}s, "
             f"build_rc={result.get('build_rc', '?')}")
        if result.get("build_rc", 1) != 0:
            raise RuntimeError(
                "remote `make test-kernels parity-kernels` failed on the H100 worker.\n"
                f"--- stdout ---\n{result.get('stdout', '')}\n"
                f"--- stderr ---\n{result.get('stderr', '')}"
            )
        self._built = True

    def run(self, name: str) -> RunResult:
        return self._dispatch("run", name, [], iodir=None)

    def parity_run(self, name: str, *args: str) -> RunResult:
        # Convention: argv[0] is the iodir path. Tarball it, send, untar
        # outputs back into the same dir.
        iodir = pathlib.Path(args[0]) if args else None
        return self._dispatch("parity_run", name, list(args), iodir=iodir)

    def _dispatch(self, op: str, name: str, argv: list, iodir):
        payload = {"op": op, "name": name, "argv": argv}
        if iodir is not None and iodir.exists():
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for p in iodir.rglob("*"):
                    if p.is_file():
                        tar.add(p, arcname=str(p.relative_to(iodir)))
            payload["iodir_tarball"] = base64.b64encode(buf.getvalue()).decode("ascii")

        _log(f"dispatching {name}{' ' + ' '.join(argv) if argv else ''}...")
        t0 = time.perf_counter()
        result = self._await(self._remote(payload), op_name=name, interval=10)
        dt = time.perf_counter() - t0
        _log(f"  {name} returned in {dt:.1f}s "
             f"(exit={result.get('exit_code', '?')})")

        # Round-trip parity outputs back into the local iodir.
        if iodir is not None and "iodir_outputs" in result:
            data = base64.b64decode(result["iodir_outputs"])
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(iodir)

        run_result = RunResult(
            name=name,
            exit_code=int(result.get("exit_code", 1)),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            duration_s=dt,
        )
        self.write_log(name, run_result.stdout, run_result.stderr)
        return run_result

    def close(self) -> None:
        try:
            self._loop.close()
        except Exception:
            pass
