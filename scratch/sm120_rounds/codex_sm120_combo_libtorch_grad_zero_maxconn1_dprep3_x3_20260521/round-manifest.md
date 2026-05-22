# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_maxconn1_dprep3_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `584`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `fd153f7f99ee6421c7abdb565bf67ea36384bf72b5c4e21fca6479b27346fea6` |
| `test_attention` | `True` | `1800528` | `d50f87c08d558d163a2f41768948d30b64daff8c2c9d516f90c9baf3e35d8970` |
| `test_layernorm` | `True` | `1278296` | `9f011b7bd20460a7fac8f5fe2c316ef5fc7f470a565b0eb3a552b023a08d1e69` |
| `test_bias` | `True` | `2089120` | `66c3fe2c6b180bfbcc2ee500d8380e782df1af3172853b7cf3832265eb6e7693` |
| `test_gelu` | `True` | `1179912` | `c6ea47d0b0c4f4ac610f68bfabbd2fd13057efa1bc63be767b8d7e4248f492a4` |
| `test_fused_classifier` | `True` | `1208704` | `51a7ffe248714c9d9189af62174b51e721fb5a4d2b232a9a370fe3cde6442c7b` |
| `test_encoder` | `True` | `1210168` | `d15b768f2a1ef005a1c79fcb405c9d66d94d58d3db67fed0d2346f17b6ddab40` |
| `test_adamw` | `True` | `1183768` | `c6c965331173a0cbb5ee0a76d2a7ea7dd793bc4a000c24dee78ad905668bd1f7` |
| `test_global_norm` | `True` | `1179464` | `5f791a0a1bac74039e3e86e5d1b959d8bb0554c1b96fad9d34d213257eccd29f` |
| `bench_sm120_matmul` | `True` | `2410304` | `d2864096ec4a782981eac542cef105478e38efa0fc9a38b66a26768b48fe3255` |
| `bench_sm120_attention` | `True` | `1768800` | `fcd343e8ecbb06a3169e8061df62a1ae05f8c8d9b72d4351a0cdaf7e9b4788db` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9d17991e1e667be7d9650281ee2a195a494ebf5bc228fc505dfa96f2cdcc75aa` |
| `bench_sm120_runtime` | `True` | `2271168` | `9d05f48db8ada920ed4e618e1d48d46827a0bd14d6bba999bd39ae0aad02cc09` |
| `train_gpt2cu` | `True` | `3105552` | `a30913635da6c856a3a697d35a461df80dc9940d285efd2b827d9ddc190630d7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
