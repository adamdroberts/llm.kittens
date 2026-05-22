# SM120 Round Manifest

- run label: `codex_sm120_precompute_disable_profiler_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_precompute_disable_profiler_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_precompute_disable_profiler_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `676`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `6cdd9d12680641145b203180435951561b6881d4c5f70ede646e65ddacee8455` |
| `test_attention` | `True` | `1800528` | `370be5612f6ab1968d919c03607fc753218d1009d266e6695828fb4576505b73` |
| `test_layernorm` | `True` | `1278296` | `47bcc15951400c16ff277f22b9fce3f04274eebba0d061b51c09c6cd1fe87021` |
| `test_bias` | `True` | `2089120` | `2ca84c98c67e53a8951487473e953293373f935a13c84411852dbad7bdaae89b` |
| `test_gelu` | `True` | `1179912` | `dd97260b86bb287c29323a9c042b133cff787c4ae2dc452953273e7e08a2ce8b` |
| `test_fused_classifier` | `True` | `1208704` | `d8f44148812d9964d66d09e4775bc97f07b342279fea8774ae0d29c6de5e00ca` |
| `test_encoder` | `True` | `1210168` | `9ee2947720d8829594c62668b81a51746e271013a852b35e32475b5533f4a38c` |
| `test_adamw` | `True` | `1183768` | `0d7b29069717e06cd4273116c61b0ae2dc1136fbe0806835579f15805c2e9e6d` |
| `test_global_norm` | `True` | `1179464` | `b58a0d9f714bb4f775c49686939d46a02eb073fcbb956bec5c6b2258a2734050` |
| `bench_sm120_matmul` | `True` | `2410304` | `85d214ff17d330064d28f545d392e4f360eeba533ac70c5a8157c04bb5b74473` |
| `bench_sm120_attention` | `True` | `1768800` | `aacde146243df099fca8819c1c420396d4acb781cb68f9f85b3dd3ff66fbdfcc` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c5e368888ac75549a6f430fa9a68f37d2105661483d6c0f492329829280ccd9b` |
| `bench_sm120_runtime` | `True` | `2271168` | `0b87900782383a0cd33541a148929a264a28eb7f494310315729d0f45d6d6711` |
| `train_gpt2cu` | `True` | `3119216` | `157c164e05d831782b7de5c5b6793814b1c1e069ef4c8a93312950ac95bec943` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
