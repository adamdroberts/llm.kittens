# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `554`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `05cf8519fd80f71a2252ca9eded13810f3828df8444c3256d5e77db82d4fa5ae` |
| `test_attention` | `True` | `1800528` | `918dd6a1445892d1e157e487c196fc8974e63c8505480fd55ba25e0c3762ce9c` |
| `test_layernorm` | `True` | `1278296` | `841119eb7f79a1ab45f8417ecc402589774b0bf82b16295a785f63b2c287925b` |
| `test_bias` | `True` | `2089120` | `037e0cc2c1501fecafc00a6e0c37cd4ba56b2e45c31fb637a648668298403c71` |
| `test_gelu` | `True` | `1179912` | `4fda7f80298dc48659b5d4d183274c05800ca9431967c0305c1b4584f1080188` |
| `test_fused_classifier` | `True` | `1208704` | `dd682297dbdfc0a705c22d23cee0ce0b93cc7e0dcf65e7ca68f6cd18e0787f01` |
| `test_encoder` | `True` | `1210168` | `631cde24b6db4ae4de820feb85ea3b1a493a13dc2c754abe1cc4d433867964d3` |
| `test_adamw` | `True` | `1183768` | `9c1e048b1717a7a2e1a9268be11488912288dd386eb1aba2cff47c8dea0ee92e` |
| `test_global_norm` | `True` | `1179464` | `c815246d009f696e70ae13bcc6c5e43240a4846aadd88132256326a626f876ec` |
| `bench_sm120_matmul` | `True` | `2410304` | `1e7500490eb4ee92d06ddb12f2100b59dfe926bfb80b0280d4142c72d782aa80` |
| `bench_sm120_attention` | `True` | `1768800` | `8b9577ae1fb058c218c6842de3095ef83ffb88d55b7ea829b83832fccae5e52f` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a491f44d5f7fc26b0f2f38730483f85affa2aaa6828cc5cee33a6d1422a134cb` |
| `bench_sm120_runtime` | `True` | `2271168` | `a70e3533f7e37308b272f8172d20e318fb7bb09f92f8c201cb806515e6e85470` |
| `train_gpt2cu` | `True` | `3105552` | `edc539be1d0e7bc0580ac343bb20a331a285e74d6e0e4da5e9d2308237b57a72` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
