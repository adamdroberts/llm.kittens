# SM120 Round Manifest

- run label: `codex_sm120_promoted_precompute_grad_scale_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_precompute_grad_scale_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_precompute_grad_scale_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `615`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `07d73bc443fdceff37fca4921c3271652abf03ae34400c931eb46001b2bb7ddb` |
| `test_attention` | `True` | `1800528` | `ae34418fbddc58a763176b13fb8e45ca1948f3cbf505ef36b261f838e43bcdf5` |
| `test_layernorm` | `True` | `1278296` | `37f6719dec271f45d7e01952ca31f96e4a940e69cb0bde25772397ed6f353012` |
| `test_bias` | `True` | `2089120` | `fb5d7980fa756e1afb5f6c6c99e5dccba06e3d3ac7c976f6cdda3c41b738f596` |
| `test_gelu` | `True` | `1179912` | `3d0a0313102b1d898b9cc47fa403870c218ca03843a66b6565561bd9a0e69911` |
| `test_fused_classifier` | `True` | `1208704` | `178d878607304a95924eb83a7c3d602bcc36e8e49148c557acb139fe8feff9e8` |
| `test_encoder` | `True` | `1210168` | `665178852d56948481b1d4c2230e7bdf3636f72dfce3770736d27796121c3bc4` |
| `test_adamw` | `True` | `1183768` | `b5964ed7a4b84e101313ed2b285021c48b4a8c53e99326d21716f1fc9c6ccc74` |
| `test_global_norm` | `True` | `1179464` | `1590203da7cb258d28a068b6c1cc96d37321c01bb653706f684d86174ac3f92e` |
| `bench_sm120_matmul` | `True` | `2410304` | `192e1bf29374d543a3ce137ad3950f7a4a1903e1edd9ff8d73e160b4713c4fc5` |
| `bench_sm120_attention` | `True` | `1768800` | `9065160c2cafc82b0db8667a34beede76ab48efc72a9024df25a08a20d989b5a` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c043efe2dcbd20ce3b0511463b260b06c6ffd6d372e7050256b93bc68bc26532` |
| `bench_sm120_runtime` | `True` | `2271168` | `8734d177149dcde7ac1712cfcf05f150d8a44998f0e2b032887d78387da850ed` |
| `train_gpt2cu` | `True` | `3119216` | `758ce307d68ef4b5cc7bdb89323fa2265121630110cdc3e1fd481ba5877d9ec7` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
