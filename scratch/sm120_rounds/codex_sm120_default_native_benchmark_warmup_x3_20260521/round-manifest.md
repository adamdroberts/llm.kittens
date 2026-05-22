# SM120 Round Manifest

- run label: `codex_sm120_default_native_benchmark_warmup_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_default_native_benchmark_warmup_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_default_native_benchmark_warmup_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `573`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `16e9a7302f323b91c23d9be7fe191b070461fceedef12894b51b015d25f75c35` |
| `test_attention` | `True` | `1760032` | `8d3f7a71926299563ae84d0f7007862d9408c91fcb74f847f6e36c6d3974d155` |
| `test_layernorm` | `True` | `1237784` | `ef515b6c35d79051b2bdf2b62b1d9efcb3928390a0e7f71648423a4f91027724` |
| `test_bias` | `True` | `2048616` | `2d30b7f2d3d0c48c76eb35fc78b3feb71eff3e61c97a86b2ca73720cb102fb12` |
| `test_gelu` | `True` | `1139336` | `97f06528cdef88aab6e1293437f5395900d7d0a52018708d01c9faa0676d3199` |
| `test_fused_classifier` | `True` | `1164032` | `df08fc99cf2ba4d05eee1212856c91b359e3222e36765d63b2de673ee935f8bd` |
| `test_encoder` | `True` | `1165512` | `2c3ff9f8b2cdbb394517bf1c3ac6f2871e3da238e8b6abf92ed336a3b7572489` |
| `test_adamw` | `True` | `1143192` | `5e1f70f4bfd7e8f4436d2dd6b72a680ac484207b1f4ba3436c10e597f6846709` |
| `test_global_norm` | `True` | `1138880` | `8b8f42c825b856e67d8f8178bf66d582257d0f8e12a3b7bcc2013cc8f71cf19e` |
| `bench_sm120_matmul` | `True` | `2373912` | `c502aebc76ce87f7e33fd91f942d7a7fa227cb1ff406e14745231ae3d596b402` |
| `bench_sm120_attention` | `True` | `1728312` | `567b8cf7c1f621b3a09ad85679fc7e54e21afb512db9d3cdf40781631e09e5a9` |
| `bench_sm120_layernorm` | `True` | `1233728` | `81ae7a7af1d51ebaf6cae96a452e7810e8bd594ecb0914c768bc609a2f0fe5ab` |
| `bench_sm120_runtime` | `True` | `2226576` | `8f408ebebb2706ef9422697c70e0ac58a9ed8be091f3c570fbcf46b726946b24` |
| `train_gpt2cu` | `True` | `3064992` | `9edf24a2c21d78a60c1c8f9b3c6b81053e6c0526cac85bd5e17f8aa032e2af0e` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
