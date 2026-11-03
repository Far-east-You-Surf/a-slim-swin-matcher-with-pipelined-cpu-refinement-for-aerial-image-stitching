"""
GERBANG #1 (tanpa training): apakah batas 768 korespondensi itu STRUKTURAL?

Latar: model memakai HANYA keluaran /32 Swin (grid 24x32 = 768 token), sementara
72.4% FLOPs dibelanjakan di stage 3 pada /16 (48x64 = 3072 token) yang lalu dibuang
oleh downsample. Satu-satunya sumbu tempat kita kalah dari LoFTR adalah jumlah
inlier ABSOLUT (198.5 vs 464.3, log §24.10) -- diduga murni karena batas grid.

Probe ini memisahkan TIGA hal yang selama ini tercampur, semuanya ZERO-SHOT
(bobot `aerial_stitching_fp16_best.pth` apa adanya, nol parameter baru):

  p0_ref          resize internal 768x1024, input 480x640   -> grid 24x32  (protokol §18/§24)
  p0_hires_in     resize internal 768x1024, input 960x1280  -> grid 24x32  (kontrol: detail input saja)
  p1_up_grid      resize internal 1536x2048, input 480x640  -> grid 48x64  (grid halus, piksel sama)
  p2_up_both      resize internal 1536x2048, input 960x1280 -> grid 48x64  (grid halus + detail nyata)
  p3_featup       resize internal 768x1024, fitur /32 di-upsample 2x       -> grid 48x64 (nol param baru)

Set pasangan IDENTIK dgn collect_pilar3_multipair.py (5 in-domain + 15 out-of-domain).
Ekstraksi match & RANSAC identik quant_sweep.evaluate_matches (thr 0.1, ruang piksel asli).

CATATAN PENTING soal ambang: keluaran dual-softmax dinormalisasi atas N token, jadi
memperbesar grid MENGENCERKAN nilai kepercayaan. Ambang tetap 0.1 karena itu protokol
yang dipakai seluruh log, tapi jumlah match juga dicatat pada 0.01/0.05/0.2/0.5 supaya
efek pengenceran itu terukur, bukan tersembunyi.

Pakai: python gate_arch_probe.py [results_dir]
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from quant_sweep import evaluate_matches, DEVICE
from collect_pilar3 import extract_matches_grid
from collect_pilar3_multipair import PAIR_SETS, agg
from gate_pilar3 import load_baseline

DEFAULT_OUT = "gate_arch_results"
THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]
PROTOCOL_THR = 0.1
MAX_PTS_RANSAC = 100_000  # di atas ini disubsampel dgn seed tetap, dan DICATAT


# --------------------------------------------------------------------------
# Konfigurasi probe
# --------------------------------------------------------------------------
# (nama, input_hw, target_size internal, feature_upsample, grid keluaran)
CONFIGS = [
    ("p0_ref",      (480, 640),  (768, 1024),  1, (24, 32)),
    ("p0_hires_in", (960, 1280), (768, 1024),  1, (24, 32)),
    ("p1_up_grid",  (480, 640),  (1536, 2048), 1, (48, 64)),
    ("p2_up_both",  (960, 1280), (1536, 2048), 1, (48, 64)),
    ("p3_featup",   (480, 640),  (768, 1024),  2, (48, 64)),
]


def load_pair_at(img1_path, img2_path, size_hw):
    """Sama dgn quant_sweep.load_pair tapi ukuran input jadi parameter."""
    img1_cv = cv2.imread(img1_path)
    img2_cv = cv2.imread(img2_path)
    orig_h, orig_w = img1_cv.shape[:2]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize(size_hw)])
    img1 = tf(cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    img2 = tf(cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    return img1, img2, orig_w, orig_h


def extract_matches_fast(conf_matrix, orig_w, orig_h, grid, threshold):
    """Vektorisasi dari collect_pilar3.extract_matches_grid (loop Python-nya tidak
    sanggup pada grid 3072x3072). Kesetaraannya diuji di main()."""
    conf = conf_matrix[0]
    h_feat, w_feat = grid
    idx = torch.nonzero(conf > threshold)
    if idx.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    i1, i2 = idx[:, 0], idx[:, 1]
    sx, sy = orig_w / w_feat, orig_h / h_feat
    pts1 = torch.stack([(i1 % w_feat).float() * sx, torch.div(i1, w_feat, rounding_mode="floor").float() * sy], 1)
    pts2 = torch.stack([(i2 % w_feat).float() * sx, torch.div(i2, w_feat, rounding_mode="floor").float() * sy], 1)
    return pts1.numpy(), pts2.numpy()


def patch_feature_upsample(model, factor):
    """p3: upsample bilinear peta fitur /32 sebelum `proj`. Nol parameter baru --
    hanya menaikkan kerapatan grid korespondensi dari fitur yang sudah ada."""
    original = model.extract_features

    def extract_features_up(img):
        img_resized = F.interpolate(img, size=model.target_size, mode="bilinear", align_corners=False)
        feats = model.backbone(img_resized)          # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2)            # [B, C, H, W]
        feats = F.interpolate(feats, scale_factor=factor, mode="bilinear", align_corners=False)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return model.proj(feats)

    model.extract_features = extract_features_up if factor != 1 else original
    return model


def eager_latency_ms(model, img1, img2, iters=20, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            model(img1.to(DEVICE).half(), img2.to(DEVICE).half())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(img1.to(DEVICE).half(), img2.to(DEVICE).half())
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    # muat pasangan sekali per ukuran input yang dipakai
    sizes = sorted({cfg[1] for cfg in CONFIGS})
    loaded = {}
    for size in sizes:
        loaded[size] = {}
        for domain, pairs in PAIR_SETS.items():
            items = []
            for p1, p2 in pairs:
                if not (os.path.exists(p1) and os.path.exists(p2)):
                    print(f"[LEWAT] tidak ada: {p1}")
                    continue
                items.append((os.path.basename(p1), os.path.basename(p2)) + load_pair_at(p1, p2, size))
            loaded[size][domain] = items
        n = sum(len(v) for v in loaded[size].values())
        print(f"[MUAT] input {size[0]}x{size[1]}: {n} pasangan")

    results = {}
    equiv_checked = False

    for name, in_hw, target, up, grid in CONFIGS:
        print(f"\n[PROBE] {name}: input {in_hw[0]}x{in_hw[1]}, internal {target[0]}x{target[1]}, "
              f"featup {up}x -> grid {grid[0]}x{grid[1]}")
        model = load_baseline()
        model.target_size = target
        model = patch_feature_upsample(model, up)

        per_domain = {}
        per_pair = {}
        lat = None
        for domain, items in loaded[in_hw].items():
            m_list, i_list, r_list = [], [], []
            for n1, n2, img1, img2, ow, oh in items:
                with torch.no_grad():
                    conf = model(img1.to(DEVICE).half(), img2.to(DEVICE).half()).float().cpu()
                assert tuple(conf.shape[1:]) == (grid[0] * grid[1],) * 2, \
                    f"bentuk conf {conf.shape} tidak cocok grid {grid}"

                # kesetaraan ekstraksi vektor vs loop asli, sekali saja pada grid baseline
                nonlocal_check = (not equiv_checked and grid == (24, 32))
                if nonlocal_check:
                    a1, a2 = extract_matches_grid(conf, ow, oh, grid, PROTOCOL_THR)
                    b1, b2 = extract_matches_fast(conf, ow, oh, grid, PROTOCOL_THR)
                    assert a1.shape == b1.shape and np.allclose(a1, b1) and np.allclose(a2, b2), \
                        "ekstraksi vektor TIDAK setara dgn versi loop"
                    equiv_checked = True
                    print("  [OK] ekstraksi tervektorisasi setara dgn collect_pilar3.extract_matches_grid")

                counts = {}
                for thr in THRESHOLDS:
                    counts[str(thr)] = int((conf[0] > thr).sum().item())

                pts1, pts2 = extract_matches_fast(conf, ow, oh, grid, PROTOCOL_THR)
                subsampled = None
                if len(pts1) > MAX_PTS_RANSAC:
                    rng = np.random.default_rng(0)
                    sel = rng.choice(len(pts1), MAX_PTS_RANSAC, replace=False)
                    subsampled = dict(n_raw=int(len(pts1)), n_used=MAX_PTS_RANSAC)
                    pts1, pts2 = pts1[sel], pts2[sel]
                matches, inliers, ratio = evaluate_matches(pts1, pts2)

                per_pair[f"{domain}/{n1}-{n2}"] = dict(
                    matches=matches, inliers=inliers, inlier_ratio=ratio,
                    counts_by_threshold=counts, ransac_subsampled=subsampled)
                m_list.append(matches); i_list.append(inliers); r_list.append(ratio)

            per_domain[domain] = dict(n=len(m_list), matches=agg(m_list),
                                      inliers=agg(i_list), inlier_ratio=agg(r_list))
            print(f"  {domain:24s} n={len(m_list):2d}  match={agg(m_list)['mean']:8.1f}  "
                  f"inlier={agg(i_list)['mean']:8.1f}  ratio={agg(r_list)['mean']:5.1f}%"
                  f" +/- {agg(r_list)['std']:.1f}")

            if lat is None and items:
                lat = eager_latency_ms(model, items[0][2], items[0][3])

        results[name] = dict(config=dict(input_hw=list(in_hw), target_size=list(target),
                                         feature_upsample=up, grid=list(grid),
                                         tokens=grid[0] * grid[1]),
                             eager_latency_ms=lat, per_domain=per_domain, per_pair=per_pair)
        print(f"  latensi eager (PyTorch, bukan engine): {lat:.2f} ms")

        del model
        torch.cuda.empty_cache()

    payload = dict(
        gpu=torch.cuda.get_device_name(0),
        checkpoint="aerial_stitching_fp16_best.pth",
        note=("ZERO-SHOT, nol parameter baru, tanpa training. Latensi = PyTorch eager, "
              "BUKAN engine TensorRT -- jangan disandingkan dgn tabel §23/§24."),
        protocol_threshold=PROTOCOL_THR,
        results=results,
    )
    path = f"{out_dir}/probe_grid.json"
    json.dump(payload, open(path, "w"), indent=2)
    print(f"\n[SELESAI] -> {path}")


if __name__ == "__main__":
    main()
