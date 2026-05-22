# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `599`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `87977721e2b0dc163290f126a79b3fcbea4d513fd23dba3511b0dfec8f6b35ee` |
| `test_attention` | `True` | `1800528` | `fe86545ff4cd9ac9938ee0093c471525fc03b34fe6917aaacb0c0b08c94c387c` |
| `test_layernorm` | `True` | `1278296` | `5d003cb88b3d32e534d911f57cdc3d6cc64ec0d502326f55b446986e9f59c18b` |
| `test_bias` | `True` | `2089120` | `68c52c1147deb6125e95cf0d92a71785f39b1776ff419bafa76372cb9a46e5f3` |
| `test_gelu` | `True` | `1179912` | `34726c038b90e397bc6acf15a2a4c6dc2ebedf79cdbe6e515d6e27bffa46f447` |
| `test_fused_classifier` | `True` | `1208704` | `0f4d665b9e7da508d6cbd53e6f4957aac28912e7519e317aeab141cdf15a386f` |
| `test_encoder` | `True` | `1210168` | `78531b0d86547a832bdff06f121f6372ac85bcb244a351dde2d6d4ed9d5de053` |
| `test_adamw` | `True` | `1183768` | `8b19b142172178b567b03f76ca0b2d621a90b0b38273194196f91ed857199e00` |
| `test_global_norm` | `True` | `1179464` | `0df63f9950c47296d5cdd343a259629618f8b5b155e4f4a0ff3290573ad27fec` |
| `bench_sm120_matmul` | `True` | `2410304` | `d5caa202d5b05ca820d81bc49b1ec50f377a931f850468095b4aa0bcc0e96535` |
| `bench_sm120_attention` | `True` | `1768800` | `4627878584543a610a4777c8d216cff3dbb73fc81b00c98f97d241bb7cf5924d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `a46c0a4e02079d93800f3964143c23e9ada448049de64c7b88b8a79405a29abe` |
| `bench_sm120_runtime` | `True` | `2271168` | `0a149f83e211cb5233b908a4a4533fcb1305fbe800d84dc81043576435e328c5` |
| `train_gpt2cu` | `True` | `3105552` | `64414fc62eae5148f4845c81e7974db5af1f01b77b68912f3852ee91dff24559` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
