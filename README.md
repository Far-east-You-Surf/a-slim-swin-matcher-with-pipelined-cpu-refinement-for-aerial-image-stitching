# A Slim Swin Matcher with Pipelined CPU Refinement for Aerial Image Stitching

Code, trained weights, and recorded results that accompany the manuscript of the same title (F. Y. Baktiar,
T. K. Priyambodo, M. Riasetiawan, Universitas Gadjah Mada; under review). The manuscript itself is not part of this
repository.

The matcher is designed for sequential nadir aerial surveys. A slim Swin-B backbone (stage 3 pruned from 18 to 8
blocks) encodes each image, a single attention body of exact softmax self- and cross-attention exchanges context between
the two images once on a 1/16 grid, and dual-softmax with mutual nearest neighbours gives coarse matches. All of this is
one static TensorRT FP16 engine. Subpixel precision comes from a refinement without learned parameters that runs on CPU
worker threads beside the engine: a global homography from the coarse matches (MAGSAC++), a prewarp of the second
image, pyramidal Lucas–Kanade tracking from zero displacement, and a map back.

## What is here

| Path | Content |
|---|---|
| `model_p16.py`, `dataset_grid_p16.py`, `dataset_homography.py`, `train_fp16_p16.py` | model, synthetic and pseudo-labelled training data, training loop |
| `train_fp16.py`, `model.py` | recipe of the earlier /32 version used for initialisation |
| `regen_pseudo_gt_p16.py`, `revisi_d/scripts/arev_pseudo_*.py` | pseudo-labels of real sequential pairs (outdoor LoFTR teacher, RANSAC filters) |
| `revisi_d/scripts/arev_train.py` | training entry point for every variant: seed, backbone depth, body depth, grid stride, internal resolution, initialisation, data roots |
| `revisi_d/scripts/arev_tp_train.py`, `arev_tp_eval.py` | retraining of MatchFormer and LGFCTR on the same aerial sample stream, and their evaluation |
| `revisi_d/scripts/e13_refine.py` | the CPU refinement (`prewarp_refine`) and the UseGeo scorer |
| `revisi_d/scripts/arev_eval.py`, `arev_eval_units.py`, `evaltool.py`, `tp_matchers.py`, `p2c_*.py` | evaluation of the proposal and of the published matchers (no limit on the number of matches) |
| `revisi_d/scripts/arev_x*.py`, `arev_*_stats.py` | analyses: estimands and cluster bootstrap, registration checked against ground truth, controlled scenario, seeded comparisons, sequence test |
| `revisi_d/scripts/arev_export_variants.py`, `arev_g2_bench.py`, `revisi_d/cloud_e14/` | ONNX export, TensorRT engines, latency, throughput, and GPU and CPU energy |
| `revisi_d/results/` | recorded summaries (JSON) of every analysis |
| `docs/preregistration.md` | decision rules fixed before the new training runs (original in Indonesian, English summary on top) |
| `DATA.md`, `THIRDPARTY.md`, `WEIGHTS.md` | datasets and their layout, pinned third-party matchers, trained weights |

Per-pair results (JSON lines, 117 MB) and all trained weights are attached to the release
`v1.0`; `WEIGHTS.md` lists every file with its SHA-256 digest.

## Environment

Python 3.14, PyTorch with CUDA, OpenCV, TensorRT 11.1 (CUDA 13); the exact versions are in `environment/`. Paths come
from environment variables: `AERIAL_ROOT` (repository root, default: derived from the file location), `AERIAL_DATA`
(dataset root, see `DATA.md`), and optionally `AERIAL_GPU_TOOLS`.

## Reproducing

1. Prepare the data as described in `DATA.md` and decode the training images once into memory-mapped arrays:
   `python revisi_d/scripts/arev_cache.py all --workers 4`.
2. Train a variant, for example the proposal with seed 2:
   `python revisi_d/scripts/arev_train.py --tag d8_s2 --seed 2`; the 18-block backbone adds `--blocks 18`, the
   variant without attention body `--layers 0`, the /32 grid `--stride 32`, the 480×640 internal input `--res 480x640`,
   initialisation from ImageNet alone `--init clean`, and the reproduction on public data
   `--init clean --train-root ... --val-root ... --pseudo ...` (see `docs/preregistration.md`, amendments 1 and 2).
   A given seed fixes the order and content of the training samples, so variants with the same seed see the same stream.
3. Evaluate: `python revisi_d/scripts/arev_eval.py d8_s2 all` (checkpoint selection on the 15 selection pairs, UseGeo
   accuracy with and without refinement, registration on UseGeo and the OpenDroneMap surveys).
4. Analyse: `arev_seed_stats.py` (seeded comparisons and the pre-registered rules), `arev_ctrl_stats.py` (controlled
   scenario), `arev_env_stats.py` (range of validity and gate), `arev_x12_stats.py` (deployment scenario),
   `arev_x6_sequence.py` (sequence test).
5. Cost: `arev_export_variants.py` and `arev_g2_bench.py` on a local GPU; `revisi_d/cloud_e14/pod4/` for rented GPUs.

## Licence

The code and documentation in this repository, including release v1.0, are released under the MIT License (LICENSE). The trained weights in release v1.0 are released under the same licence, except the MatchFormer and LGFCTR weights retrained on the aerial stream (`tp_*`), which are derived from Apache-2.0 weights and remain under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The reproduction on public data (`pub_s1_*`) was trained on OpenAerialMap orthomosaics, part of which are licensed CC BY-NC 4.0 (DATA.md). Third-party code and weights keep their own licences (THIRDPARTY.md).

## Citation

If you use this code, please cite the article once it is published; until then, cite this repository.
