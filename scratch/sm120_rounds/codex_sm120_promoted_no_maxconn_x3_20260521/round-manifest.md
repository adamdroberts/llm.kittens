# SM120 Round Manifest

- run label: `codex_sm120_promoted_no_maxconn_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_no_maxconn_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_no_maxconn_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `607`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `6c8373ebda3b5c582b8369e8aefbf81795da041dd8bc789462afdd4d0131a00c` |
| `test_attention` | `True` | `1800528` | `b0fc7179b31dbe7cdf4f7e255bc4a4203ba220b9ba457c3c9869f8f3da8b2bc1` |
| `test_layernorm` | `True` | `1278296` | `68709a0fa8e9f4a890dec9a88200a582bfc87baa409cb84f75c77cef276f18d6` |
| `test_bias` | `True` | `2089120` | `f3c02c7c8f20aa7f14bc6d889e8b5aa44d3d83c262729525c0e01ee2733651c5` |
| `test_gelu` | `True` | `1179912` | `90ba47587eb0e685c67d135a3a7607dd21c553b5b0dc01cf6db4f5dcf149c839` |
| `test_fused_classifier` | `True` | `1208704` | `a21051822c7345e5ca65776e9284f07145a959ad8c4664721040b91ea895e4a3` |
| `test_encoder` | `True` | `1210168` | `e0c0ddfe09cd15670f002edd12803e05fb3f15a82dffac2f03ea36e9f6880f1a` |
| `test_adamw` | `True` | `1183768` | `c57acb8c0fcf930ed6c478eb93af18bbe92f6445afd8528b808b45af7b928200` |
| `test_global_norm` | `True` | `1179464` | `77d77009d9f6c30e95011350d0dc43ad4728e35cf0c35158c1f65691318532f2` |
| `bench_sm120_matmul` | `True` | `2410304` | `a8fe0d0e7bc4bbcdfdcfcc3b597a05f526626f6f7afb1cd2f1b134f5698dd073` |
| `bench_sm120_attention` | `True` | `1768800` | `5873c954ed3d60bb770f794164dc094737f3098e47f07156f5e7eb24b20f8cd3` |
| `bench_sm120_layernorm` | `True` | `1274232` | `5e16e40bf09b9ed6b4612ada0cb0b44c4a65c0cdda8ccbb2e74f6984c81907e7` |
| `bench_sm120_runtime` | `True` | `2271168` | `3c95d87f699630464770903475e8b648ee359e7287aa2d761c77e0509541a597` |
| `train_gpt2cu` | `True` | `3105552` | `2b9712f285a626ecab9ff0485e62a28232ff66609c2c0318b82d3ae3cc050188` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
