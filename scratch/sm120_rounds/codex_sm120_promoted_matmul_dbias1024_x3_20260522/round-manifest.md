# SM120 Round Manifest

- run label: `codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_matmul_dbias1024_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `663`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `3df01ffc55a483d86de1808e6de9787160302fef13b5e85137a446963bda8207` |
| `test_attention` | `True` | `1800528` | `a01061727322ba2626c427322776e12a36c47c9d083bf35376e8a64f5b4a0a97` |
| `test_layernorm` | `True` | `1278296` | `662cc87253cc96274cc1b2dfbdc0c7eabf035a688bb01b5f3b55aef916f1c37e` |
| `test_bias` | `True` | `2089120` | `c6a30b010241ff5c228bace1bd71f3062af69979c6da62d1731c01d6c1808d74` |
| `test_gelu` | `True` | `1179912` | `14684713173f894088054e2d1a4a0f967c9edc80b50c59bf9398721f3d90c94b` |
| `test_fused_classifier` | `True` | `1208704` | `05a3b2a2b1e4ebc6f445571a42e22cc9fbaf906870ab6ae7b1b7c2f07917255b` |
| `test_encoder` | `True` | `1210168` | `1d86554c0028f1dc1929af3764f4de4dcaa1e0247bd9d2b2a1d7e77a0a475b51` |
| `test_adamw` | `True` | `1183768` | `ff1dfaddf464f819e3be499306491837e746218852026238e697e238ba20ef5b` |
| `test_global_norm` | `True` | `1179464` | `1ff9a241a991b0073b4e776326bea4f24b896d2b8f87bde578fdb6ec12c429ac` |
| `bench_sm120_matmul` | `True` | `2410304` | `0d0bd4f9ac40f92dca193445feb19014c4d23d87dabbe7a523a79d62e03866c8` |
| `bench_sm120_attention` | `True` | `1768800` | `0b525ce2aceb8dc0f93f1e73bd140b6b06b81a80f188e24f22d8c07e200f87f8` |
| `bench_sm120_layernorm` | `True` | `1274232` | `aaf00c21064c037f022b6c8d2f64d1ce60c3c243ca74aa59f539b9078b3ea890` |
| `bench_sm120_runtime` | `True` | `2271168` | `257c7af3b1cfa4fb256edb6108d98ec3b3dd4f315002057c37a43ba101b3614a` |
| `train_gpt2cu` | `True` | `3105552` | `d33444c91ece267faf43e36530d8c468bf8ff30ab07724f0e73b5e94a59c7298` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
