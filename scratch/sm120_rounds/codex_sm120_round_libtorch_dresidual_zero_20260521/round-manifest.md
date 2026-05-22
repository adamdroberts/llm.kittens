# SM120 Round Manifest

- run label: `codex_sm120_round_libtorch_dresidual_zero_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_libtorch_dresidual_zero_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_libtorch_dresidual_zero_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `503`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `b215dd774d113de5fc0d2255a906a1e998e1ae62036fa8a7c935ef0e714bf6b5` |
| `test_attention` | `True` | `1800528` | `bdb7a1ee7210151182acbf03c287e30910bcc3811619e9d79f95e60cef6550c0` |
| `test_layernorm` | `True` | `1278296` | `f37794744724be356af79d0f97ca5cb16c4358638572d76499a2639a27ea989b` |
| `test_bias` | `True` | `2089120` | `a5a7b155ca27e0b281ad2c2a825bc89d9c8ed87e37f0b13e60e2585490639216` |
| `test_gelu` | `True` | `1179912` | `a37e24ff8b32341fc23968d6eea3d07aaf0277c36ad9b3b898fa8795edc84120` |
| `test_fused_classifier` | `True` | `1208704` | `ed9c2d6bd9165bef0f697466f4736a076bec7a5a457ba8f478822ce87723d0d0` |
| `test_encoder` | `True` | `1210168` | `f32c85525f7e3b237576858f4aa38fc50f8c2b5d5ec21ed84f6906a0dfde58a5` |
| `test_adamw` | `True` | `1178984` | `603ee39ef0aabf95334d9e234682e26b0edad93a765e15059cd1afe18e5e9b15` |
| `test_global_norm` | `True` | `1179464` | `8af8ae5b3bf6aac6283e02c170fcf267711f31953cb20a317a620260f8f08477` |
| `bench_sm120_matmul` | `True` | `2410304` | `b32ba4455c94a56db7f09773bc287d40f07a10a2a2963458fa4bd5285620559d` |
| `bench_sm120_attention` | `True` | `1768800` | `2319c871b7c8778ec0304992f9702e4a394733df4fcccecfce6a87fdb3c144a5` |
| `bench_sm120_layernorm` | `True` | `1274232` | `70b3ca9160c281d12cf1bc78f5b86dea47a05ba6c18577a2afcd956332da3c69` |
| `bench_sm120_runtime` | `True` | `2270544` | `54d6a7bca2ebe529a36010b1e3135fe6a8733f007550a889c0134f3a69731e6b` |
| `train_gpt2cu` | `True` | `3100592` | `a744b24a0c7416eaf03ce9a37b878fece047f7fed3488d3a797d1385f7f252b5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
