# SM120 Round Manifest

- run label: `codex_sm120_matmul_dbias768_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_matmul_dbias768_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_matmul_dbias768_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `678`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `b0d3fe9751485bf293d04c9959e6beba031eba34517ed52f3e75f2ad20df26d4` |
| `test_attention` | `True` | `1800528` | `0a2b0ebf304ab99b62c130edac2b5e91e42ca9981d3e934d3676b51986e0112c` |
| `test_layernorm` | `True` | `1278296` | `6a3f24cfb2d4c3a1bd30744056070ddbadaa61719f342d7f704a80be2e5ca4c5` |
| `test_bias` | `True` | `2089120` | `c65a9b61abef2b48b601f62faa6603563de2a18fb173656580a7cd0b52e34af8` |
| `test_gelu` | `True` | `1179912` | `456af44f464660d650028df08e0ce31f507a10494a23b8699efbe7a63214b522` |
| `test_fused_classifier` | `True` | `1208704` | `e82430b5265abf3ca627907542a93684c195b0c40cecbb64843fde0c24eb7403` |
| `test_encoder` | `True` | `1210168` | `3569b380cdad489614904047bb5b8c9df4830d9e8ca0051b25d48206b2595155` |
| `test_adamw` | `True` | `1183768` | `092c066f9f654f5bf5e04acee25e56db02cfa80892ca217bd603e591815d41fe` |
| `test_global_norm` | `True` | `1179464` | `a3ea66c8e8dc2bb47ce2f13e0fffc272e05fffad106af652685989465415fc52` |
| `bench_sm120_matmul` | `True` | `2410304` | `2bf1a6d810b91e91f1f57384090d64b338b0a32c1fd348e4b4a39848eb8b2e7f` |
| `bench_sm120_attention` | `True` | `1768800` | `0ad6c15ab84b4786a03b93eaca82f8911fdc4dde30a38dc6acacd13048ca732d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `ee537da5e50ea64ac5f8b8bd92e1997aff49437ec6871cbc5da4f4c6a9579902` |
| `bench_sm120_runtime` | `True` | `2271168` | `ea816e0c8570b131d592c3dd4349101501ebf47ba5aafc5f4314763e5fb571e6` |
| `train_gpt2cu` | `True` | `3105552` | `8fa6406b868b634e484bef219064b1578bb9727dd27c430ac0adc1838ad83c11` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
