# SM120 Round Manifest

- run label: `codex_sm120_attn_bwd64_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_attn_bwd64_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_attn_bwd64_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `661`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `61bf9de485a82811198329acb1499e773fb72b0f0a2282ce56497da50b7ece60` |
| `test_attention` | `True` | `2804048` | `4da923fdbd22311db5e645e1b682af36fdd0228db740fc137ad78c59b48fd5dc` |
| `test_layernorm` | `True` | `1278296` | `f7844b37f0983eb6f38e85989b85fd6edd13c813957bf9b8e7fbaad079a88732` |
| `test_bias` | `True` | `2089120` | `20858874501ad957790b3004dc60ff15485ea059950055c7ca0d1e4bb9f348fe` |
| `test_gelu` | `True` | `1179912` | `a8ca8d064870e061acf62eadcafc23e68ead84a1a51b4c08f0a2e72f8e333ca7` |
| `test_fused_classifier` | `True` | `1208704` | `d3a969dc9fa155567646ce70899bafbfdf177b752a6c28b5afcb8e9f6af2504b` |
| `test_encoder` | `True` | `1210168` | `a06522cb8dfb8ad4a2f7ac40c55f2b3590e2dc2b4e6123271c90e6fddfa38875` |
| `test_adamw` | `True` | `1183768` | `300e3263b4a51c62f115c81c68abb9855cac49b0c15484e533528449bf948e1d` |
| `test_global_norm` | `True` | `1179464` | `9243d3e15b7ec3bfd7f175a0cd873fa72b09407f0eff7197b166e02e9835f3b6` |
| `bench_sm120_matmul` | `True` | `2410304` | `4745eb42ce7dc554fa80ccd2fd0ddcb6b22d8ee50a6c5e4175286f3ad25169da` |
| `bench_sm120_attention` | `True` | `2768224` | `37104819255aeb627542f27af11f760a4953b44f32cc7f3c825af3848ec9bff1` |
| `bench_sm120_layernorm` | `True` | `1274232` | `9e07959575c0f3dd0b6303bb86f28b8e1f4478e247238588fe6e971506973628` |
| `bench_sm120_runtime` | `True` | `2271168` | `da9d691d21b7e36c64b96bfac68f4a0cf1f51bad5c77659bcbd86cf68563a495` |
| `train_gpt2cu` | `True` | `4104976` | `72c545ca1f29ce0ff19e6a5ee15b8317fd25317120de8a18791901d708993cfe` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
