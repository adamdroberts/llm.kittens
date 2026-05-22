# SM120 Round Manifest

- run label: `codex_sm120_dprep2_recovered_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_dprep2_recovered_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_dprep2_recovered_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `658`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `ad45a1c9b938d406276f61f0e38b4c4d14a727165f434e8768f80ed970837c2a` |
| `test_attention` | `True` | `1800528` | `f4893ac2376a88cc6202bf0b30076be29a383fbc264fe0a005410ce911e420e6` |
| `test_layernorm` | `True` | `1278296` | `5883e2db250ef7148f3b54effaa374f7d76e252524d5ccf271f3d96d9c0d0d75` |
| `test_bias` | `True` | `2089120` | `afc20f3bfeefa8a8d23e54b2985c911f48f5bce281fb56af617c28cc7736f2b9` |
| `test_gelu` | `True` | `1179912` | `2f1d44caf2f0f4a8b8c018e7744aa4f9e00ecc4918c0e18878c66c18b7036d9f` |
| `test_fused_classifier` | `True` | `1208704` | `b1a0e79004e09ad1a8d3036aad01e5d129d8e272b748537b603c0a485f58e0e5` |
| `test_encoder` | `True` | `1210168` | `dab680b59a33a2549afb65b540f0c5c2ae7ef27ca44bb008a0a147661abd8249` |
| `test_adamw` | `True` | `1183768` | `ef17b68671d256b2cd223c3ba6065529ae6f7326fc5158cbed6b3fb140d8eef7` |
| `test_global_norm` | `True` | `1179464` | `cdb75d2e3c39c256098e4a074cada9442d137cbf30977d2f8b426e717e6e5676` |
| `bench_sm120_matmul` | `True` | `2410304` | `62c1d8113512b790cd039578058f37a7c1f50259d54a21745b9011eb3e35796a` |
| `bench_sm120_attention` | `True` | `1768800` | `23c19be7765943a091dd4caae200b382b1c241946a669f47f85c85e9024fbc48` |
| `bench_sm120_layernorm` | `True` | `1274232` | `046cbb70df665142d14cd93eff885ba9d40e0653657925ca2a88ea27fdc29cbc` |
| `bench_sm120_runtime` | `True` | `2271168` | `3ad7cfd4f8f3c41bbe36b099f7d8aa6e584ccec621d713ac806e82d3bbcacecb` |
| `train_gpt2cu` | `True` | `3105552` | `30e70cabfa26dff60c30d15b6d41519ccbc4fed6f62f5e4b2a7a58c13015d5a5` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
