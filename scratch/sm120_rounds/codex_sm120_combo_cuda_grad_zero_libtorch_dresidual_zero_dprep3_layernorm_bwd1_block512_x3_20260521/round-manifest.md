# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block512_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `598`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `a2caa31c73206a3f38046e4ebf0741ff882d00f33b7dc441bad9181b7d9da904` |
| `test_attention` | `True` | `1800528` | `6b73182f335514fb51d875c358ba3681bfe4d47bc2b4addb29dcfc8af55cca43` |
| `test_layernorm` | `True` | `1278296` | `eaa46a9cfbdae56c4d09dac971d6da39252af57ac2483ca456b8dbc117a8f667` |
| `test_bias` | `True` | `2089120` | `2940cd1a661fc884d88ab039ce3e3c5fc3383e37792dde857ac750c7ea5fb9ad` |
| `test_gelu` | `True` | `1179912` | `f1c6ecb8ffad4f7a7da867732cdc664de8f23699a942246fc8374921de4a3326` |
| `test_fused_classifier` | `True` | `1208704` | `8a41cc8cde58c8445b474da7ee75501753e63e2c2232b6a26695bc1e4b10033f` |
| `test_encoder` | `True` | `1210168` | `6de0fb896f7cbe55f76dbd25b2d35d995e86e7d05e06f4e2c7b0edb474732e46` |
| `test_adamw` | `True` | `1183768` | `6ff3ad587385372191a5543da1d533384f807cbc387e13f5306b5a116e848059` |
| `test_global_norm` | `True` | `1179464` | `01491d729b3f026a6da11b0ed32e3ba6363efea878b9c8ca5c31d9149eb0d4c0` |
| `bench_sm120_matmul` | `True` | `2410304` | `d63206743c340ec3ee709747938c0f2068bd7eb42c7c974dd67e4612c5bd5366` |
| `bench_sm120_attention` | `True` | `1768800` | `c38ff1a78c7f80ccc3a450cb7b679db6f3147804732d79517c7c0823a5ade8f3` |
| `bench_sm120_layernorm` | `True` | `1274232` | `05591d3f3617d775da7d772be59e82587bb28e371e25083edb848fed091c1724` |
| `bench_sm120_runtime` | `True` | `2271168` | `ca317198258d8929dd621b12e1983df4282c28c66bac3b74a2f19e817b90c5e5` |
| `train_gpt2cu` | `True` | `3105552` | `12a73a4996cd17c9230aa1e5941cfd49ca138fc099729315ba2cbae37b793db6` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
