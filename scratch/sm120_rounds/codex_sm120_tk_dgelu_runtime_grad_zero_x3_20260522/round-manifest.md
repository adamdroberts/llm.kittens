# SM120 Round Manifest

- run label: `codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_runtime_grad_zero_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `667`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `30f72724753de0d15ca3bb2dafd9616b91cf8cc09c6394d91780a5193d22fe67` |
| `test_attention` | `True` | `1800528` | `fbc4b49b789b36a32485cb4501f5a4b2b759961f2854f7871695555e1a49bb17` |
| `test_layernorm` | `True` | `1278296` | `6d874fbbe6a8b6af527f8c826f2184dd8ede2332fc0680983ed5ec4f6e51ead9` |
| `test_bias` | `True` | `2052256` | `dcbd5453f76cc7fcc899fb0e9801484520d1682e822e83d3b2b34ece212f7313` |
| `test_gelu` | `True` | `1179912` | `7fb3399cb8f957395bfd0aeee34fc99521db267839968307dfb2d5045bc3c716` |
| `test_fused_classifier` | `True` | `1208704` | `e3a673bba7f26d95db18e075b70deae1ae572b8c9f147a20dd98ff05c14ac7d0` |
| `test_encoder` | `True` | `1210168` | `35a549b94ab380dda7153bc9cd5d450d68cbc28706441f5663a73783932fcb71` |
| `test_adamw` | `True` | `1183768` | `1d26e2ddc7005d07c1a49b375fb012bdda2c67a115d11f09ee7f17442a9fcf83` |
| `test_global_norm` | `True` | `1179464` | `9c0ed6691f95a593fc8195cbb266d0e0488e83bdb134fc296cf5dda1ac752a3d` |
| `bench_sm120_matmul` | `True` | `2369344` | `fd66fb87fe578bd26323829b6d61a37696d69a15b1e8eddfc81d0dbfd47c6dbe` |
| `bench_sm120_attention` | `True` | `1768800` | `fe03f3ebf199a0cc2504c6c86629f800aa49a12495c27498ed30c2e17bf33606` |
| `bench_sm120_layernorm` | `True` | `1274232` | `07633802f3d41060ae52c16204d3d82027969c5b9ed0318b6bcdd0dfb240e0fd` |
| `bench_sm120_runtime` | `True` | `2234304` | `14f24510ba4427a1365dd9a6a589de33f315f1deebd2b70c61ba86f471525d7e` |
| `train_gpt2cu` | `True` | `3104616` | `ba9d126b2f4f9c5cd680b72dcf8437b637777fab2f76b99bbdd6f3ac4f5e63ce` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
