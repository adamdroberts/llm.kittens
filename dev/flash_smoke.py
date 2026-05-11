"""Minimal smoke test: dispatch a CUDA `1 + 1` add to a runpod-flash worker
and print the result. Use to confirm flash itself is working before blaming
the bigger parity harness.

    pip install runpod-flash
    flash login                                  # or export RUNPOD_API_KEY=...
    python dev/flash_smoke.py                    # default: ADA_80_PRO (H100)
    python dev/flash_smoke.py --gpu ADA_32_PRO   # RTX 5090 (SM120)

Cold start is typically 30-90s; the script prints a heartbeat every 10s so
you can see it's not dead.

What it actually does on the worker:
  1. nvidia-smi → confirms a GPU is attached and reports its name + sm.
  2. nvcc-compiles a tiny `add<<<1,1>>>` kernel that writes a+b into device
     memory and copies it back.
  3. Runs the binary and returns its stdout.

If you see {'sum': 2, ...} on stdout, flash is verified end-to-end: auth,
endpoint creation, cold-start, code dispatch, GPU access, nvcc toolchain,
kernel launch, result return. If you see anything else, the fields tell
you which step failed.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from runpod_flash import Endpoint, GpuGroup


def _log(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


async def _cuda_add_impl(a: int, b: int) -> dict:
    """Body of the smoke test. Wrapped by Endpoint at runtime so we can pick
    the GPU group from a CLI flag instead of hard-coding it on the decorator."""
    import subprocess

    src = r"""
        #include <cstdio>
        #include <cuda_runtime.h>
        __global__ void add(int* c, int a, int b) { *c = a + b; }
        int main(int argc, char** argv) {
            int a = atoi(argv[1]); int b = atoi(argv[2]);
            int* d_c; cudaMalloc(&d_c, sizeof(int));
            add<<<1,1>>>(d_c, a, b);
            cudaDeviceSynchronize();
            int h_c = 0;
            cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
            printf("%d\n", h_c);
            cudaFree(d_c);
            return 0;
        }
    """
    with open("/tmp/add.cu", "w") as f:
        f.write(src)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    cc = subprocess.run(
        ["nvcc", "/tmp/add.cu", "-o", "/tmp/add"],
        capture_output=True, text=True,
    )
    if cc.returncode != 0:
        return {"step": "compile", "stderr": cc.stderr, "gpu": smi.stdout.strip()}
    rn = subprocess.run(["/tmp/add", str(a), str(b)], capture_output=True, text=True)
    return {
        "step": "ok" if rn.returncode == 0 else "run",
        "sum": int(rn.stdout.strip()) if rn.returncode == 0 else None,
        "stdout": rn.stdout,
        "stderr": rn.stderr,
        "gpu": smi.stdout.strip(),
    }


_DEFAULT_LOCATIONS_US_EU = (
    "US-CA-2,US-KS-2,US-NC-1,US-OR-1,US-TX-3,US-IL-1,US-WA-1,"
    "EU-CZ-1,EU-NL-1,EU-RO-1,EU-SE-1,EU-FR-1"
)
_DEFAULT_CUDA_VERSIONS = "12.8"  # runpod's latest enumerated CUDA


def _build_endpoint(gpu_name: str):
    """Apply Endpoint decorator with the chosen GPU group at call time.

    Also pins `locations` to any US+EU datacenter (override with
    `RUNPOD_LOCATIONS`) and `allowedCudaVersions` to CUDA 13.0+ (override
    with `RUNPOD_CUDA_VERSIONS`). The SDK's typed CudaVersion enum tops at
    12.8 — we set the free-string field directly via a monkey-patch.
    """
    import os
    if not hasattr(GpuGroup, gpu_name):
        raise SystemExit(f"unknown GpuGroup '{gpu_name}'. See `runpod_flash.GpuGroup`.")
    gpu = getattr(GpuGroup, gpu_name)
    locations = os.environ.get("RUNPOD_LOCATIONS", _DEFAULT_LOCATIONS_US_EU)
    cuda_versions = os.environ.get("RUNPOD_CUDA_VERSIONS", _DEFAULT_CUDA_VERSIONS)
    # SDK reads RUNPOD_DEFAULT_LOCATIONS during resource-model build.
    os.environ["RUNPOD_DEFAULT_LOCATIONS"] = locations
    # Stable name per GPU type — reuses warm worker across calls.
    # Override with RUNPOD_ENDPOINT_NAME for a fresh deploy.
    gpu_suffix = gpu_name.lower().replace("_", "-")
    default_name = f"llm-kittens-smoke-1plus1-{gpu_suffix}"
    endpoint_name = os.environ.get("RUNPOD_ENDPOINT_NAME", default_name)
    endpoint = Endpoint(
        name=endpoint_name,
        gpu=gpu,
        workers=1,
        idle_timeout=60,
        flashboot=True,
        dependencies=[],
        system_dependencies=["build-essential"],
        execution_timeout_ms=180_000,
    )
    # CUDA has no env hook; monkey-patch the resource builder.
    original_build = endpoint._build_resource_config
    def patched_build():
        cfg = original_build()
        if hasattr(cfg, "allowedCudaVersions"):
            cfg.allowedCudaVersions = cuda_versions
        return cfg
    endpoint._build_resource_config = patched_build
    _log(f"locations={locations}")
    _log(f"cuda_versions={cuda_versions}")
    return endpoint(_cuda_add_impl)


async def main(gpu_name: str) -> int:
    cuda_add = _build_endpoint(gpu_name)
    _log(f"dispatching cuda_add(1, 1) to runpod-flash (GpuGroup.{gpu_name})...")
    t0 = time.time()
    task = asyncio.create_task(cuda_add(1, 1))

    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
        except asyncio.TimeoutError:
            _log(f"  ... still waiting ({int(time.time() - t0)}s elapsed)")

    result = task.result()
    elapsed = time.time() - t0
    _log(f"done in {elapsed:.1f}s")
    print(f"Result: {result}")
    if result.get("step") == "ok" and result.get("sum") == 1 + 1:
        _log(f"PASS — flash is working end-to-end on {gpu_name}")
        return 0
    _log("FAIL — see Result above for which step broke")
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--gpu", default="ADA_80_PRO",
        help="GpuGroup name (e.g. ADA_80_PRO for H100, ADA_32_PRO for RTX 5090). "
             "Default: ADA_80_PRO",
    )
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.gpu)))
