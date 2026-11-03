"""
Evaluasi kualitas multi-pair untuk varian Pilar #3, memakai SET PASANGAN YANG SAMA
PERSIS dengan `evaluate_multi_pairs.py` (§16): 5 pasang in-domain (Aerial234) +
5 pasang out-of-domain (G Uji). Menggantikan evaluasi 1-pasang-per-domain di
`collect_pilar3.py` yang sampelnya terlalu kecil untuk menyimpulkan apa pun.

Metodologi identik §16/§18: RANSAC inlier ratio di ruang piksel gambar asli, plus
cos-sim matriks confidence vs referensi FP16 PyTorch.

Semua angka tetap ZERO-SHOT (tidak ada finetune) -- ini batas bawah, bukan hasil akhir.

Pakai: python collect_pilar3_multipair.py [results_dir]
"""

import json
import os
import sys

import numpy as np
import torch

from quant_sweep import load_pair, evaluate_matches, compare, run_pytorch_conf, DEVICE
from gate_pilar3 import load_baseline, VARIANTS, DEFAULT_OUT
from collect_pilar3 import GRID, extract_matches_grid
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

AERIAL234_DIR = AERIAL_DATA_UJICOBA + "/Dataset Aerial234"
GUJI_DIR = AERIAL_DATA_UJICOBA + "/G Uji"

# Out-of-domain dinaikkan ke n=15 (2026-08-04): simpangan baku di n=5 mencapai 31.9,
# terlalu besar untuk klaim yang kuat. G Uji punya 39 foto berurutan (DSC00003-00041),
# jadi 15 pasangan berurutan disebar merata sepanjang jalur terbang.
# Kelima pasangan lama (3,10,20,30,40) SENGAJA dipertahankan sebagai subset supaya
# angka n=15 bisa dibandingkan langsung dgn angka n=5 di §24.8/§24.9.
_GUJI_STARTS = [3, 5, 8, 10, 13, 16, 18, 20, 23, 26, 28, 30, 33, 36, 40]

PAIR_SETS = {
    "aerial234_in_domain": [
        (f"{AERIAL234_DIR}/001.JPG", f"{AERIAL234_DIR}/002.JPG"),
        (f"{AERIAL234_DIR}/050.JPG", f"{AERIAL234_DIR}/051.JPG"),
        (f"{AERIAL234_DIR}/100.JPG", f"{AERIAL234_DIR}/101.JPG"),
        (f"{AERIAL234_DIR}/150.JPG", f"{AERIAL234_DIR}/151.JPG"),
        (f"{AERIAL234_DIR}/230.JPG", f"{AERIAL234_DIR}/231.JPG"),
    ],
    "g_uji_out_of_domain": [
        (f"{GUJI_DIR}/DSC{i:05d}.JPG", f"{GUJI_DIR}/DSC{i + 1:05d}.JPG")
        for i in _GUJI_STARTS
    ],
}

# 5 pasangan out-of-domain asli (§24.8/§24.9) -- dipakai utk melaporkan subset
# yang sebanding saat membandingkan n=5 vs n=15.
GUJI_ORIGINAL_5 = {f"DSC{i:05d}.JPG-DSC{i + 1:05d}.JPG" for i in (3, 10, 20, 30, 40)}


class EngineRunner:
    """Deserialize engine SEKALI, lalu pakai ulang utk semua pasangan."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.out_shape = tuple(self.engine.get_tensor_shape("conf_matrix"))
        self.stream = torch.cuda.Stream()

    def __call__(self, img1, img2):
        d1 = img1.to(DEVICE).half().contiguous()
        d2 = img2.to(DEVICE).half().contiguous()
        out = torch.empty(self.out_shape, dtype=torch.float16, device=DEVICE)
        self.context.set_tensor_address("img1", d1.data_ptr())
        self.context.set_tensor_address("img2", d2.data_ptr())
        self.context.set_tensor_address("conf_matrix", out.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out.float().cpu()


def agg(values):
    if not values:
        return None
    a = np.array(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    # muat semua pasangan sekali
    loaded = {}
    for domain, pairs in PAIR_SETS.items():
        loaded[domain] = []
        for p1, p2 in pairs:
            if not (os.path.exists(p1) and os.path.exists(p2)):
                print(f"[LEWAT] file tidak ada: {p1} / {p2}")
                continue
            loaded[domain].append((os.path.basename(p1), os.path.basename(p2)) + load_pair(p1, p2))
        print(f"[MUAT] {domain}: {len(loaded[domain])} pasangan")

    # referensi FP16 PyTorch per pasangan
    print("[REF] hitung referensi FP16 (PyTorch) utk semua pasangan...")
    ref_model = load_baseline()
    conf_ref = {}
    for domain, items in loaded.items():
        for n1, n2, img1, img2, ow, oh in items:
            conf_ref[(domain, n1, n2)] = run_pytorch_conf(ref_model, img1, img2)
    del ref_model
    torch.cuda.empty_cache()

    all_results = {}
    for variant in VARIANTS:
        engine_path = f"{out_dir}/{variant}.engine"
        gate_json = f"{out_dir}/{variant}.json"
        if not (os.path.exists(engine_path) and os.path.exists(gate_json)):
            continue
        gate = json.load(open(gate_json))
        if "error" in gate:
            continue

        print(f"[EVAL] {variant} ...")
        runner = EngineRunner(engine_path)
        grid = GRID[variant]
        per_domain = {}
        for domain, items in loaded.items():
            rows, matches, inliers, ratios, cos_sims = [], [], [], [], []
            for n1, n2, img1, img2, ow, oh in items:
                conf = runner(img1, img2)
                ref = conf_ref[(domain, n1, n2)]
                cs = compare(ref, conf)[1] if conf.shape == ref.shape else None
                pts1, pts2 = extract_matches_grid(conf, ow, oh, grid)
                m, i, r = evaluate_matches(pts1, pts2)
                rows.append(dict(pair=f"{n1}-{n2}", matches=m, inliers=i,
                                 inlier_ratio=r, cos_sim=cs))
                matches.append(m); inliers.append(i); ratios.append(r)
                if cs is not None:
                    cos_sims.append(cs)
            per_domain[domain] = dict(
                n_pairs=len(rows), per_pair=rows,
                matches=agg(matches), inliers=agg(inliers),
                inlier_ratio=agg(ratios), cos_sim=agg(cos_sims),
            )
            a = per_domain[domain]
            cos_str = f"{a['cos_sim']['mean']:.4f}" if a["cos_sim"] else "n/a"
            print(f"    {domain:22} n={a['n_pairs']} "
                  f"match={a['matches']['mean']:.1f} inlier={a['inliers']['mean']:.1f} "
                  f"ratio={a['inlier_ratio']['mean']:.1f}% "
                  f"(rentang {a['inlier_ratio']['min']:.1f}-{a['inlier_ratio']['max']:.1f}) "
                  f"cos={cos_str}")

        all_results[variant] = dict(
            variant=variant,
            gflops=gate.get("gflops"), latency_ms=gate.get("latency_ms"),
            engine_device_memory_mb=gate.get("engine_device_memory_mb"),
            grid=list(grid), domains=per_domain,
            note="ZERO-SHOT, tanpa finetune; set pasangan identik evaluate_multi_pairs.py (§16)",
        )
        del runner
        torch.cuda.empty_cache()

    n_out = len(loaded["g_uji_out_of_domain"])
    out_path = f"{out_dir}/collect_multipair_n{n_out}.json"
    json.dump(dict(gpu=torch.cuda.get_device_name(0), results=all_results),
              open(out_path, "w"), indent=2)
    print(f"=== SELESAI, ditulis ke {out_path} ===")


if __name__ == "__main__":
    main()
