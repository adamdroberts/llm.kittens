# SM120 Round Manifest

- run label: `codex_sm120_cublas_dinp_fcproj_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_cublas_dinp_fcproj_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_cublas_dinp_fcproj_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `549`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `851f11ea1b8d69646e98b9430403c62e66b1547cd11f318157c9ea2124b6820d` |
| `test_attention` | `True` | `1760032` | `d2939e5439ed1ed308202db7f76c8f1e875142d7a2f42eaf4e0e151154af4355` |
| `test_layernorm` | `True` | `1237784` | `59534c1146085d120b08dc2da5d32db2383d4e8d788f898f91776d05444bead2` |
| `test_bias` | `True` | `2048616` | `326f3a1d272e633e5bddae84d0577be54cc119fc80badb2f729c955ce5cb3935` |
| `test_gelu` | `True` | `1139336` | `e4ec6f7b5c68f002bd5a94e1b52b0919b819c6f11316dccaad8111ddfdaabbd8` |
| `test_fused_classifier` | `True` | `1164032` | `6d849d17cf4b1008a4a135220a0ce6b879618868c0a411ce65542e0c3855bc6f` |
| `test_encoder` | `True` | `1165512` | `533cd6a6733cd9fdfd9343c5f606c6b2eb5ac3ae64bc6c690adaa50df70b4020` |
| `test_adamw` | `True` | `1143192` | `539376dd0cca6a113d77f15b943211e5c3bdbf6da77f7a6c1d6d2d9004d50366` |
| `test_global_norm` | `True` | `1138880` | `bbef7c8e86d173e5f0fedadfd95081a3854e7bea28e6f3737afe041be0a3067d` |
| `bench_sm120_matmul` | `True` | `2373912` | `5578003080dde8dbe896240a5069398bc7a07190e7ae5a1b79633f9efaa34bfb` |
| `bench_sm120_attention` | `True` | `1728312` | `4011ae9f4d60768b645ec507e36975193f46e316c70465446498523cf71a89e1` |
| `bench_sm120_layernorm` | `True` | `1233728` | `1c5579e4cb1fff3cb8ad2a94f459f6b18a27e6c978b0600e89bdb7d6755f0a48` |
| `bench_sm120_runtime` | `True` | `2226576` | `a86deaa040952317a75446b48fddf6f3969ef81422e016b9f15e50920f0a8687` |
| `train_gpt2cu` | `True` | `3064992` | `78594080a720b05da18fd6bb0b6ea7e7ead61ae43faf2649db7f209dbf066535` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
