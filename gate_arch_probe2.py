"""
GERBANG #1b: ulangi probe grid dgn SELEKTIVITAS DISAMAKAN.

Kenapa perlu: `extract_matches` memakai ambang tetap 0.1 pada keluaran dual-softmax,
padahal dual-softmax dinormalisasi atas N token. Memperbesar grid 768 -> 3072
MENGENCERKAN nilai kepercayaan (terukur di probe_grid.json: pada 3072 token tidak
ada satu pun sel menembus 0.5, sementara pada 768 token rata-rata 90.8 sel menembus).
Jadi perbandingan lintas ukuran grid pada ambang tetap BIAS terhadap grid halus, dan
angka di probe_grid.json tidak boleh dibaca sebagai jawaban akhir.

Di sini seleksi dibuat setara dgn dua cara:
  topk      : ambil K korespondensi berkepercayaan tertinggi (anggaran match disamakan)
  mnn_topk  : mutual nearest neighbour dulu (cara LoFTR), baru ambil top-K

Pertanyaan yang dijawab: pada anggaran match yang SAMA, apakah grid halus menghasilkan
korespondensi BENAR lebih banyak? Dan berapa langit-langit inlier absolutnya?
Pembanding yang relevan: LoFTR 464.3 inlier out-of-domain vs kita 198.5 (log §24.10).

Pakai: python gate_arch_probe2.py [results_dir]
"""

import json
import os
import sys

import numpy as np
import torch

from quant_sweep import evaluate_matches, DEVICE
from collect_pilar3_multipair import PAIR_SETS, agg
from gate_pilar3 import load_baseline
from gate_arch_probe import CONFIGS, load_pair_at, patch_feature_upsample

DEFAULT_OUT = "gate_arch_results"
K_VALUES = [200, 400, 800, 1600, 3200]


def topk_points(conf, orig_w, orig_h, grid, k, mutual):
    """K korespondensi teratas -> titik piksel di gambar asli."""
    c = conf[0].clone()
    h_feat, w_feat = grid
    n = c.shape[0]
    if mutual:
        best_j = c.argmax(dim=1)
        best_i = c.argmax(dim=0)
        rows = torch.arange(n)
        keep = best_i[best_j[rows]] == rows          # mutual NN
        vals = c[rows[keep], best_j[rows[keep]]]
        i1 = rows[keep]
        i2 = best_j[rows[keep]]
        if vals.numel() > k:
            sel = torch.topk(vals, k).indices
            i1, i2 = i1[sel], i2[sel]
    else:
        k_eff = min(k, c.numel())
        flat = torch.topk(c.flatten(), k_eff).indices
        i1 = torch.div(flat, n, rounding_mode="floor")
        i2 = flat % n
    sx, sy = orig_w / w_feat, orig_h / h_feat
    pts1 = torch.stack([(i1 % w_feat).float() * sx,
                        torch.div(i1, w_feat, rounding_mode="floor").float() * sy], 1).numpy()
    pts2 = torch.stack([(i2 % w_feat).float() * sx,
                        torch.div(i2, w_feat, rounding_mode="floor").float() * sy], 1).numpy()
    return pts1, pts2


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted({cfg[1] for cfg in CONFIGS})
    loaded = {}
    for size in sizes:
        loaded[size] = {}
        for domain, pairs in PAIR_SETS.items():
            items = []
            for p1, p2 in pairs:
                if os.path.exists(p1) and os.path.exists(p2):
                    items.append((os.path.basename(p1), os.path.basename(p2)) + load_pair_at(p1, p2, size))
            loaded[size][domain] = items
        print(f"[MUAT] input {size[0]}x{size[1]}: {sum(len(v) for v in loaded[size].values())} pasangan")

    results = {}
    for name, in_hw, target, up, grid in CONFIGS:
        print(f"\n[PROBE2] {name} (grid {grid[0]}x{grid[1]} = {grid[0]*grid[1]} token)")
        model = load_baseline()
        model.target_size = target
        model = patch_feature_upsample(model, up)

        entry = {}
        for domain, items in loaded[in_hw].items():
            acc = {(m, k): {"inliers": [], "ratio": [], "matches": []}
                   for m in ("topk", "mnn_topk") for k in K_VALUES}
            for n1, n2, img1, img2, ow, oh in items:
                with torch.no_grad():
                    conf = model(img1.to(DEVICE).half(), img2.to(DEVICE).half()).float().cpu()
                for method in ("topk", "mnn_topk"):
                    for k in K_VALUES:
                        p1, p2 = topk_points(conf, ow, oh, grid, k, mutual=(method == "mnn_topk"))
                        m, i, r = evaluate_matches(p1, p2)
                        acc[(method, k)]["matches"].append(m)
                        acc[(method, k)]["inliers"].append(i)
                        acc[(method, k)]["ratio"].append(r)

            dom_out = {}
            for (method, k), v in acc.items():
                dom_out.setdefault(method, {})[str(k)] = dict(
                    matches=agg(v["matches"]), inliers=agg(v["inliers"]), inlier_ratio=agg(v["ratio"]))
            entry[domain] = dict(n=len(items), **dom_out)

            tag = "in " if "aerial" in domain else "out"
            for method in ("topk", "mnn_topk"):
                line = "  ".join(
                    f"K={k}: {dom_out[method][str(k)]['inliers']['mean']:6.1f}i/"
                    f"{dom_out[method][str(k)]['inlier_ratio']['mean']:4.1f}%" for k in K_VALUES)
                print(f"  {tag} {method:9s} {line}")

        results[name] = dict(config=dict(input_hw=list(in_hw), target_size=list(target),
                                         feature_upsample=up, grid=list(grid),
                                         tokens=grid[0] * grid[1]),
                             by_domain=entry)
        del model
        torch.cuda.empty_cache()

    payload = dict(
        gpu=torch.cuda.get_device_name(0),
        checkpoint="aerial_stitching_fp16_best.pth",
        k_values=K_VALUES,
        note=("ZERO-SHOT, nol parameter baru. Seleksi disamakan (top-K / mutual-NN + top-K) "
              "supaya perbandingan lintas ukuran grid tidak bias oleh pengenceran dual-softmax. "
              "Pembanding target: LoFTR 464.3 inlier out-of-domain (log §24.10)."),
        results=results,
    )
    path = f"{out_dir}/probe_grid_matched_selectivity.json"
    json.dump(payload, open(path, "w"), indent=2)
    print(f"\n[SELESAI] -> {path}")


if __name__ == "__main__":
    main()
