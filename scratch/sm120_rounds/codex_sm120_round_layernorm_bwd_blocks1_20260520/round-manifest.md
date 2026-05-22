# SM120 Round Manifest

- run label: `codex_sm120_round_layernorm_bwd_blocks1_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_layernorm_bwd_blocks1_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_layernorm_bwd_blocks1_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `468`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2137216` | `ac56f2d30c4621d64491f29cb7cec686d8f0c628f3aca12d992c55d5977ee26a` |
| `test_attention` | `True` | `1760032` | `88111ed5083000bc14770da0918e5f1882278a83d17da6ed973949b2e0b35b6c` |
| `test_layernorm` | `True` | `1237784` | `d42cd842ebb510d0dda8b1e0a4f2875040906929a38a0ca233fc60c924e1f8e1` |
| `test_bias` | `True` | `2048616` | `9c54a67d3af81fb67ff067df570446a7c8a8cfdcf5ee3e714855e02c9bda68a3` |
| `test_gelu` | `True` | `1139336` | `43d19c5f5073aa198991c550bee103d627608ad6e25c08d8bb61c549eb7d53e6` |
| `test_fused_classifier` | `True` | `1164032` | `934866082165356e1f579b5b477cf25d0c292e277b5b165ab638cf60c0750116` |
| `test_encoder` | `True` | `1165512` | `f80eba8a0ad5c572cb86488f5f8a5825fe7d038663e04fcdb3fba46a4f15561d` |
| `test_adamw` | `True` | `1138408` | `7f1d570791f035612c857ffcb59995a7abef1d84c0f66ff3c8c2a45aac7b1ee2` |
| `test_global_norm` | `True` | `1138880` | `500d5e9d1c59c5ff7e023f427445590556e4b588bfa38c95a32d8fe9627ab07d` |
| `bench_sm120_matmul` | `True` | `2373912` | `c323a2825e1e55ef8eb88ea2c2216facc255817186e16753a13341dcb011e0ba` |
| `bench_sm120_attention` | `True` | `1731352` | `f7f2d7caed5504e1efd11dbabd317320bf300e119386e774d677900035bbbbee` |
| `bench_sm120_layernorm` | `True` | `1233728` | `fd20bc4185dd78c84e48fa320325be592dfb2e07b0fe8294d489ce49dd22307b` |
| `bench_sm120_runtime` | `True` | `2199032` | `7417313ba4607394603bcaeaaa94d57e9951d9db089b11914f84ddfef4afc5fb` |
| `train_gpt2cu` | `True` | `3045944` | `1791f1d7f510f698757d579201b11b45badf64d346ff102ecb5343604bcf3a6f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
