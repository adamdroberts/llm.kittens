# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_block1024_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `591`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `d321c44cebaf787dc2903cb67f3445d5860cb895098ba679001889d7a6e5a63f` |
| `test_attention` | `True` | `1800528` | `e43d9929b6d43737ba9ea1878c7b153cb918b35eeecaec7f3c396d3a5f24f1e6` |
| `test_layernorm` | `True` | `1278296` | `362d6dab9092ff68b091a9515a55afe29c1ed1b24653d7a24d0c71f6a76b0bc5` |
| `test_bias` | `True` | `2089120` | `3f0a0b65926186e2574ac7b422e350390f1fdaea97d5a0350405f20a3e4ce0f9` |
| `test_gelu` | `True` | `1179912` | `4d3adb7cf5efe2d486f6bfcdee9017384f5eee8adb369e0d78d0ccb78f2a0ca8` |
| `test_fused_classifier` | `True` | `1208704` | `798ef9dbaad228707c8b4046f1a30265cc759edfb9b832b83be99e754aa2ae67` |
| `test_encoder` | `True` | `1210168` | `22bcd5b99b71d23733265a44e0d5c1d4817e5db5ac0fbdec801cc49d57c7ff25` |
| `test_adamw` | `True` | `1183768` | `db95adcbdee94d6ff2fee82279ee85696e71d2cb10aa6c7342bf788675f4666b` |
| `test_global_norm` | `True` | `1179464` | `bd8e1966e3aa0cb4d04ae9cfa546f39d48f48862e8acee660304a761cafb823e` |
| `bench_sm120_matmul` | `True` | `2410304` | `231b152717ede2dce7d8bb5ce433dd1059a3444a29ee9461acda89b565848bb1` |
| `bench_sm120_attention` | `True` | `1768800` | `2c93d28eb1ab522735fe7d513cdab1137f9f23ad6d3acbd506fce4a9be6cedbe` |
| `bench_sm120_layernorm` | `True` | `1274232` | `110a451ed7466e76a4ba5cdeb45aa06c2796f5f54c2bd7715df47c634da5df71` |
| `bench_sm120_runtime` | `True` | `2271168` | `47a8a9110e90d8534f890c30d149c90dcace31b1df074f465c1b49b0e3bd9954` |
| `train_gpt2cu` | `True` | `3105552` | `2252c7d16e72813d52a3354c5537977a074ff22fb2acfec330f905164c187dbf` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
