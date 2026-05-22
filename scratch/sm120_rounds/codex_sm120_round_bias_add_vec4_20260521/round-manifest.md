# SM120 Round Manifest

- run label: `codex_sm120_round_bias_add_vec4_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_bias_add_vec4_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_bias_add_vec4_20260521`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `498`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2146144` | `8b64eec5799b33bb50d3dd61c25ee425701ce25480ee2fa9c01307a63bbdfd4d` |
| `test_attention` | `True` | `1760032` | `0d5facfd966598eee3995ea698166fc7055ab29714f59d8493441b1bce6c3731` |
| `test_layernorm` | `True` | `1237784` | `8426a45462cddf9fc4c042ce50fc505704c549cf3e95c134ecb8c53bc3d6c48a` |
| `test_bias` | `True` | `2057544` | `ecdd910e9398f477fdd949dff61efb439c63e5279fb20fa8df19698eefa5a62c` |
| `test_gelu` | `True` | `1139336` | `f1298e2f2e0566f13356513873e07d68e256b03931228b843594d287858bcb76` |
| `test_fused_classifier` | `True` | `1164032` | `04c7f66d6e7bbe44398fabc8b88344dfe66fb04431790d80e01ee7b2eb217a4e` |
| `test_encoder` | `True` | `1165512` | `8c73840586817b263f479333b737ce2f03cde6c2f79cc7d60aab77b56c89880e` |
| `test_adamw` | `True` | `1138408` | `628d9a9a1f8572232a38efbcc155616b6dbe453adbf39f94adb23f44bf09c83b` |
| `test_global_norm` | `True` | `1138880` | `3a18d8e7480f819a197445d76de90b1b71bc7f30c4836dd65b6df582132915e2` |
| `bench_sm120_matmul` | `True` | `2382480` | `d23372627b03717517f0ba4b059bf364b48b44d193d7dbf3cbaf739db5f011f7` |
| `bench_sm120_attention` | `True` | `1728312` | `87f07ee0e637fd2954820f6e8de99d0d0c2031566de8fefb9d311222a7e7b95a` |
| `bench_sm120_layernorm` | `True` | `1233728` | `c847f52caae64aac4709679a08c13a3098fd085f2855f7c6cd79a6a1178112ed` |
| `bench_sm120_runtime` | `True` | `2234808` | `06f15b5d3124ab61411d6602c194770ae334bb9032ae4568061a62e9fc7f1e9d` |
| `train_gpt2cu` | `True` | `3054872` | `d458f3ff6ce968b59cf56897f62c0cff97941a03699b7adf5bb1663ad378bfa1` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
