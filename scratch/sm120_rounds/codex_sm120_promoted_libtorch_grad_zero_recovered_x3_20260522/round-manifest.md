# SM120 Round Manifest

- run label: `codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `653`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e6c777c47d7f527604a764b8574f9a842245a491353577bbdc0584698970eb01` |
| `test_attention` | `True` | `1800528` | `acfefeff833c46ea5391c5e84a9765f932d745e7f8064357a1f36dcec03de585` |
| `test_layernorm` | `True` | `1278296` | `12ba2fefcf218e916ec103c6333895694c8c9b9d507d88569b6a6cf95a604de5` |
| `test_bias` | `True` | `2089120` | `04c7dab453823f6c2732cad28acef4cf388c5205bf0781e93a49cb110e0fd309` |
| `test_gelu` | `True` | `1179912` | `4a781177c3a0e58f4ab5fdfa414c4b29f118db110d93658a0259e80cd97dcd5d` |
| `test_fused_classifier` | `True` | `1208704` | `f3d98a741eb517e2a7bb5ba8850d2e58c45558f3bca183d865db44a6ebbfb8fa` |
| `test_encoder` | `True` | `1210168` | `2da4d4e8b1774feee4e66278dc3213832e1adc16eb60f32757d9fe4e5cb93e67` |
| `test_adamw` | `True` | `1183768` | `37ec8d2c9e25357b7ea54f13603b14f619c179bf60d2a677c65d89ad045af99d` |
| `test_global_norm` | `True` | `1179464` | `7bccf1fc7f0f1089d3fb9587ee0eaa44ab05e880904eec759a3877b1231b2e3a` |
| `bench_sm120_matmul` | `True` | `2410304` | `0b9126e22147c305f0029e0be22ab63ddc6e8d7d6c1e46947041954f74224292` |
| `bench_sm120_attention` | `True` | `1768800` | `f28bef8d6bcd1a327d8b1d20b38b96318d30d9c9fede23f57fac530f37d66aec` |
| `bench_sm120_layernorm` | `True` | `1274232` | `469b9e26567a04ef36ca01768ea77151c6eef36b69ac6697267265fe9b7774eb` |
| `bench_sm120_runtime` | `True` | `2271168` | `e16f8a9c52b4d6f54124e5e3f44bcc03fd7caf02c72408f53e298decc6aaa7c1` |
| `train_gpt2cu` | `True` | `3105632` | `c1ec88ad1677bf9eed9b6578b4d35255e2463fb96aad191ee1d7b4e75f9434e1` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
