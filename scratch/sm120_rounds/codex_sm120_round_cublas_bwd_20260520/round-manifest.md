# SM120 Round Manifest

- run label: `codex_sm120_round_cublas_bwd_20260520`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_cublas_bwd_20260520`
- train output dir: `log124M/5090_S_codex_sm120_round_cublas_bwd_20260520`
- device arch: `SM120`
- max steps: `3`
- git commit: `0f21747`
- changed paths: `447`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2128288` | `edcf3e32ba03b7a1d51e52d533d7accc8ab490cc8144206934c213fe3e2b29ae` |
| `test_attention` | `True` | `1760032` | `974654c5c69d5dfd6a5ef3c5b0876248cc3f11da1a11e40ecdab223dac49dcaa` |
| `test_layernorm` | `True` | `1237784` | `e07077c46d3a7acc022eb85079e0581e607d8b0c8df820d001f7fe63cd2e6697` |
| `test_bias` | `True` | `2039504` | `6002cec3d183c200a57cb5cfb28a792f524e4a9111285c92697db5f4430f52bf` |
| `test_gelu` | `True` | `1139336` | `c88bcc2f04cc76ed21f5f72f055f44073ae8bdf3107a5198555051519e8435fb` |
| `test_fused_classifier` | `True` | `1146768` | `af9a312f6a0fb3ced5713a350812f2818c340d4c05790ecc1e98b3d4648cc82a` |
| `test_encoder` | `True` | `1165512` | `f852afe2b0d12b22dc49b0b7c808c3edd5310f21a621d303f6feed396b2b059e` |
| `test_adamw` | `True` | `1138408` | `97bfdef70fb4c4b5d2192555124c0c108245730fdb92e1dc9d61131f70d8ac6f` |
| `test_global_norm` | `True` | `1138816` | `9e7866edbf5b8530c3d2a7f0dabd903ee89400af1f184543df0a5a3994637808` |
| `bench_sm120_matmul` | `True` | `2344440` | `356ecc3b40081627edff385f23c46236b43fbd51ab071cc1cb2db19312934976` |
| `bench_sm120_attention` | `True` | `1731352` | `764f1d8a3037f408a832a1de621d6d2c888f431fe4b5bcf5d8e5c1839a3ae666` |
| `bench_sm120_layernorm` | `True` | `1229088` | `1efbb9f659036577a5df0b0c34cceb761706113a7e3bfe852ce0b17f06952ddd` |
| `bench_sm120_runtime` | `True` | `2168552` | `6e3e2f2e7c4fc346a8e6368420b7dd55c4d9c61e56f9da78453980095810a1de` |
| `train_gpt2cu` | `True` | `3037016` | `06acff5ca5b150eaa1638694452df2d4e9fbf59360e68db0364d8367b126cfc6` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
