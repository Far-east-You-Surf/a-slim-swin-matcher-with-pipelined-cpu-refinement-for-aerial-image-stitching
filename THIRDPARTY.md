# Third-party matchers

The ten published matchers are used exactly as released: their repositories at the commits below and their public outdoor weights. None of them is included in this repository. `setup_thirdparty.sh` clones the repositories at the pinned commits; the weights must be downloaded from the links in each repository's README and placed at the paths below. Licences: see each repository (MatchFormer, LGFCTR, and EDM are Apache-2.0; Efficient LoFTR uses its own project licence). `revisi_d/results/e12/provenance.json` records where the weights were obtained. The weights of MatchFormer and LGFCTR retrained on our aerial data are derived from their Apache-2.0 weights and are released with this repository (see `WEIGHTS.md`).

| Label in the scripts | Code | Commit (date) | Weights used | Size | SHA-256 |
|---|---|---|---|---|---|
| `loftr` | LoFTR through kornia 0.8.2, `kornia.feature.LoFTR(pretrained="outdoor")` | – | `outdoor` weights downloaded by kornia | – | – |
| `efficientloftr` | https://github.com/zju3dv/EfficientLoFTR → `revisi_d/thirdparty/EfficientLoFTR` | `07e9c14` (2026-09-15) | `revisi_d/thirdparty/weights/eloftr/weights/eloftr_outdoor.ckpt` | 184 MB | `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878` |
| `topicfm_fast` | https://github.com/TruongKhang/TopicFM → `third_party/TopicFM` | `b1b1a56` (2026-01-04) | `third_party/TopicFM/pretrained/topicfm_fast.ckpt` | 45 MB | `e69df628def7449fe86334e37469637bf9b20db426c1a7d05c78d1ce3ea27d56` |
| `jamma` | https://github.com/leoluxxx/JamMa → `third_party/JamMa` | `24acce2` (2025-05-29) | `jamma.ckpt`, fetched by the repository's own loader (`demo/utlis.py`) from its GitHub release v0.1 into the torch hub cache | – | – |
| (JamMa dependency) | https://github.com/state-spaces/mamba → `third_party/mamba_src`, `mamba_ssm` built from source | `10b5d63` (2025-10-10) | – | – | – |
| `aspanformer` | https://github.com/apple/ml-aspanformer → `revisi_d/thirdparty/ml-aspanformer` | `ada1119` (2026-09-11) | `revisi_d/thirdparty/weights/aspanformer/weights/outdoor.ckpt` | 61 MB | `b1041cad1f484b85b86df8aadc3e7b0ea3060ba2a412550b2f1fc58f501689d1` |
| `matchformer` | https://github.com/jamycheung/MatchFormer → `revisi_d/thirdparty/MatchFormer` | `1b2da5c` (2022-10-07) | `revisi_d/thirdparty/weights/matchformer/outdoor-large-LA.ckpt` | 78 MB | `634002dc7163e45343ede54f0fd918d559f37f52667955ed7bc126375ee07074` |
| `lgfctr` | https://github.com/zwh0527/LGFCTR → `revisi_d/thirdparty/LGFCTR` | `29ae212` (2024-06-11) | `revisi_d/thirdparty/weights/lgfctr/LGFCTR_outdoor.ckpt` | 189 MB | `f88593ed38cecc917104d7e630ce3f25bc65e81e2e5fc5055dd393044e4213d1` |
| `swinmatcher_tgrs` | https://github.com/LotrL/SwinMatcher → `revisi_d/thirdparty/SwinMatcher` | `9dc1de0` (2026-05-18) | `revisi_d/thirdparty/weights/swinmatcher/swinmatcher_512.ckpt` | 340 MB | `c27a41fb3905fe1a2f032a95869590b979b4e54a7b54569a322415cb5f329b92` |
| `edm` | https://github.com/chicleee/EDM → `revisi_d/thirdparty/EDM` | `8a31d81` (2025-11-18) | `revisi_d/thirdparty/EDM/weights/edm_outdoor.ckpt` | 40 MB | `f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97` |
| `disk_lightglue` | DISK and LightGlue through kornia 0.8.2, `KF.DISK.from_pretrained("depth")`, `KF.LightGlue(features="disk")` | – | weights downloaded by kornia | – | – |

Notes

* Each matcher runs in its own process: EfficientLoFTR, LGFCTR, ASpanFormer, TopicFM, JamMa, and EDM all use a top-level package named `src`, and MatchFormer a package named `model` (see the docstring of `revisi_d/scripts/tp_matchers.py`); EDM is registered by `revisi_d/scripts/arev_tp_extra.py` (outdoor configuration, 1680 coarse candidates at 640×480).
* All matches are used: the evaluation sets `STITCH_CAP` to 10^9, so no matcher is limited to a fixed number of matches.
* The wrappers that load the models and export the coarse path and the complete matcher to ONNX are `revisi_d/scripts/p2c_<matcher>.py` with `p2c_common.py`; TensorRT engines are built from those ONNX files by `revisi_d/cloud_e14/scripts/build_engines.py`.

## Note on the EDM engine

The first FP16 export of EDM used its optimised dual-softmax path (`ds_opt`), whose exponential overflows in half
precision and turns every confidence into infinity. The export was repaired by switching to the native dual softmax,
which is mathematically identical (`revisi_d/scripts/arev_edm_onnx.py`; agreement with the FP32 model on 100 UseGeo
pairs in `revisi_d/results/arev/g2/edm_fidelity.json`). The EDM entries in the rented-GPU sessions
`cloud_e14/results/A_l4_r2` and `A_a100sxm_r1` come from the engine before the repair and are not used;
`revisi_d/scripts/arev_g2_summary.py` drops them and takes EDM from the repaired engine only. The accuracy of EDM was
evaluated in PyTorch at FP32 and is not affected.
