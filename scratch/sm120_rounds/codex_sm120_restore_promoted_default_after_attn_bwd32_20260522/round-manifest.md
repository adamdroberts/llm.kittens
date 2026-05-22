# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_attn_bwd32_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_attn_bwd32_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_attn_bwd32_20260522`
- device arch: `SM120`
- max steps: `0`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `640`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `e406549ae3d37baae56724ec0b256a184f1984211bb1a97a38f5c7f94ce0737f` |
| `test_attention` | `True` | `1800528` | `e814f3841133c5d4db415a7d1d9821dd25d9234d247f6463e2d0371de18259f5` |
| `test_layernorm` | `True` | `1278296` | `dc949e0722906f25a4eef5dff1cdf6d545e47855b74e375266bea2128b055024` |
| `test_bias` | `True` | `2089120` | `554ac10f6f1d55d6186d56bfd9f382c514c79978092d0f324048c3e07f748b12` |
| `test_gelu` | `True` | `1179912` | `37d61ccd05199d1a16c55a919d68419433d46c91cacdab37ea564b9783d42acc` |
| `test_fused_classifier` | `True` | `1208704` | `dd49435e0089e9a857d468a8aebc69fd40cfa83f497ecf74ab8caf559e6f8108` |
| `test_encoder` | `True` | `1210168` | `1d1b29762bccd17b0537cde6a2f54df5c88c8046637a87b7f938759d56dc43a1` |
| `test_adamw` | `True` | `1183768` | `dc4f2482bd95b1614312ba91f009684aae32657fbf45db88a833d52c92f02588` |
| `test_global_norm` | `True` | `1179464` | `ea239599c66f18457420af2aae2b0c79f3ae2cdfadb2f4b1e743cee76780ad4a` |
| `bench_sm120_matmul` | `True` | `2410304` | `4eba64e311e686624300badbde55f16dc36a5b67989f95b660d8328219b8c9a4` |
| `bench_sm120_attention` | `True` | `1768800` | `0f61629fd232fd67d3b717d96dfa101b53f16df9fd1538556faaa9eb644ab431` |
| `bench_sm120_layernorm` | `True` | `1274232` | `bd57342107bdc7599cbad4d571216b6216aba11196eeab27bd6341b5278776b5` |
| `bench_sm120_runtime` | `True` | `2271168` | `3ff62250306380f19aa6972823b857355f840aed43b0552fe7b1fe9aa5c21e33` |
| `train_gpt2cu` | `True` | `3105552` | `9940b7593e8ecf0e787da28ba17c3fd3c6eae5a431124312daf94d2383a323fe` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
