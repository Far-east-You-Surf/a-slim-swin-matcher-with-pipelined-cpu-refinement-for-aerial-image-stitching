"""
Pengambilan data kualitas untuk kandidat Pilar #3 yang LOLOS gerbang kelayakan
(gate_pilar3.py). Metodologi eval identik dgn quant_sweep.py supaya angkanya
langsung comparable dgn tabel §18/§19/§23.

PENTING -- semua angka kualitas di sini adalah ZERO-SHOT (tanpa finetune):
  * b1_sparse24  : magnitude pruning 2:4 tanpa recovery finetune
  * b2_depth*    : blok dibuang begitu saja tanpa finetune
  * b3/b4        : resolusi/token diubah, checkpoint dilatih di konfigurasi lama
Jadi ini BATAS BAWAH kualitas, bukan kualitas akhir. Gunanya: mengukur seberapa
besar lubang yang harus ditutup finetune, dan memilih kandidat mana yang layak
dibiayai training.

Catatan grid: b4_* mengubah ukuran grid output (bukan 24x32 lagi), jadi
  - cos-sim vs referensi FP16 TIDAK didefinisikan (bentuk matriks beda) -> null
  - inlier ratio TETAP valid krn dihitung di ruang piksel gambar asli

Pakai: python collect_pilar3.py [results_dir]
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

from quant_sweep import (PAIRS, load_pair, evaluate_matches, compare,
                         run_pytorch_conf, run_engine_conf, RES_HW, DEVICE)
from gate_pilar3 import load_baseline, VARIANTS, DEFAULT_OUT

# grid feature (H_feat, W_feat) per varian -- target_size / 32
GRID = {
    "baseline":       (24, 32),
    "b1_sparse24":    (24, 32),
    "b2_depth12":     (24, 32),
    "b2_depth8":      (24, 32),
    "b3_stage3_half": (24, 32),
    "b4_512x640":     (16, 20),
    "b4_384x512":     (12, 16),
}


def extract_matches_grid(conf_matrix, orig_w, orig_h, grid, threshold=0.1):
    """Sama dgn quant_sweep.extract_matches, tapi grid-nya parameter (bukan hardcode 24x32)."""
    conf = conf_matrix[0]
    H_feat, W_feat = grid
    mask = conf > threshold
    indices = torch.nonzero(mask)
    scale_x, scale_y = orig_w / W_feat, orig_h / H_feat
    pts1, pts2 = [], []
    for idx in indices:
        idx1, idx2 = idx[0].item(), idx[1].item()
        y1, x1 = divmod(idx1, W_feat)
        y2, x2 = divmod(idx2, W_feat)
        pts1.append([x1 * scale_x, y1 * scale_y])
        pts2.append([x2 * scale_x, y2 * scale_y])
    return np.array(pts1), np.array(pts2)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    pairs_loaded = {name: load_pair(p1, p2) for name, (p1, p2) in PAIRS.items()}

    # referensi: model FP16 asli di PyTorch (sama seperti quant_sweep)
    print("[COLLECT] hitung referensi FP16 (PyTorch)...")
    ref_model = load_baseline()
    conf_ref = {name: run_pytorch_conf(ref_model, img1, img2)
                for name, (img1, img2, ow, oh) in pairs_loaded.items()}
    del ref_model
    torch.cuda.empty_cache()

    all_results = {}
    for variant in VARIANTS:
        gate_json = f"{out_dir}/{variant}.json"
        engine_path = f"{out_dir}/{variant}.engine"
        if not (os.path.exists(gate_json) and os.path.exists(engine_path)):
            print(f"[COLLECT] {variant}: dilewati (belum lolos gerbang / engine tidak ada)")
            continue

        gate = json.load(open(gate_json))
        if "error" in gate:
            print(f"[COLLECT] {variant}: dilewati (gerbang gagal: {gate['error']})")
            continue

        print(f"[COLLECT] {variant}: evaluasi kualitas...")
        grid = GRID[variant]
        quality = {}
        for name, (img1, img2, ow, oh) in pairs_loaded.items():
            conf_test = run_engine_conf(engine_path, img1, img2)
            if conf_test.shape == conf_ref[name].shape:
                mse, cos_sim = compare(conf_ref[name], conf_test)
            else:
                mse, cos_sim = None, None  # grid beda -> tidak terdefinisi
            pts1, pts2 = extract_matches_grid(conf_test, ow, oh, grid)
            matches, inliers, ratio = evaluate_matches(pts1, pts2)
            quality[name] = dict(mse=mse, cos_sim=cos_sim, matches=matches,
                                 inliers=inliers, inlier_ratio=ratio,
                                 conf_shape=list(conf_test.shape))

        entry = dict(
            variant=variant,
            gflops=gate.get("gflops"),
            params=gate.get("params"),
            latency_ms=gate.get("latency_ms"),
            fps=gate.get("fps"),
            engine_device_memory_mb=gate.get("engine_device_memory_mb"),
            onnx_total_nodes=gate.get("onnx", {}).get("total_nodes"),
            variant_info=gate.get("variant_info"),
            quality=quality,
            note="kualitas ZERO-SHOT, tanpa finetune",
        )
        all_results[variant] = entry
        print(f"[COLLECT] {variant}: "
              + ", ".join(f"{n}={q['inlier_ratio']:.1f}%" for n, q in quality.items()))

    out_path = f"{out_dir}/collect_summary.json"
    json.dump(dict(gpu=torch.cuda.get_device_name(0), res_hw=list(RES_HW),
                   results=all_results), open(out_path, "w"), indent=2)
    print(f"=== SELESAI, ditulis ke {out_path} ===")


if __name__ == "__main__":
    main()
