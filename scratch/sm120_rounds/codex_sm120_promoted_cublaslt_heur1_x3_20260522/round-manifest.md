# SM120 Round Manifest

- run label: `codex_sm120_promoted_cublaslt_heur1_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_cublaslt_heur1_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_cublaslt_heur1_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `641`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `2ab481b979afe211fe193dbc5f0bef657d6db3820394650ca1c2081ec4831ad1` |
| `test_attention` | `True` | `1800528` | `39d92a83799680947e95db84c2f770fdded1f4376b9a8c6c4e21219a5e37f672` |
| `test_layernorm` | `True` | `1278296` | `8e39a7e8c93259ae467bc09dc7b84e65ad9b4132551034ae7482fda26c463ef0` |
| `test_bias` | `True` | `2089120` | `bf796f438be803f1770ed88f9fcd885de34a4aed9271b9e0b7444c0679d03505` |
| `test_gelu` | `True` | `1179912` | `6270989126a83aab60a8536d236abed6103c06b891cd3818fbb31f24c0fdcbf6` |
| `test_fused_classifier` | `True` | `1208704` | `ef60a813af9244b0ed62bf5540040583124abfa9c9e2c36f62da066022dcdd17` |
| `test_encoder` | `True` | `1210168` | `1c98526a203da79bd44088ebfa542432104bc4aa8bc6ad85e076c386e28b3ea7` |
| `test_adamw` | `True` | `1183768` | `6904a2b38f42cc7feccd2e1e207551068b17e5d962ef7de37bd3bdd37e62d88f` |
| `test_global_norm` | `True` | `1179464` | `4b41004e445bf1c54d8fded62f7758d9ccaddc5fc6c88227120f5a48e50faa45` |
| `bench_sm120_matmul` | `True` | `2410304` | `d8ab3c11715e1b49f2b4d8bdadfdc3753a9dce50d3f5e02008bbd646ed58ab2e` |
| `bench_sm120_attention` | `True` | `1768800` | `04e2e39ee200472918e9a36b07b18cb5fd1912b3885776d018185042d4046100` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c334e4d4fec07bab7f55fc11f12bd13a26a8895fd537f93f30b579be8d75295f` |
| `bench_sm120_runtime` | `True` | `2271168` | `39d010cdf29040f382284977f456c86d05480fcb50c74af0ec0c7520da6479cd` |
| `train_gpt2cu` | `True` | `3105552` | `7fabceccbbbd0ee79e3cb4db71f78d85f25adf62c299b1a5e7ee54d1888b0c1f` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
