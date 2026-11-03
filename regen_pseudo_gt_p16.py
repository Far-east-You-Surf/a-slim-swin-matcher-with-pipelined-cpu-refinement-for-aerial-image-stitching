"""
Regenerasi pseudo-GT pada grid /16 (48x64) untuk arsitektur Artikel D.

Cache lama (`pseudo_gt_cache/`, 687 pasangan) memanggang indeks sel pada grid 24x32,
jadi tidak bisa dipakai ulang untuk grid 48x64 -- sel tujuannya berbeda.

Yang dikerjakan ulang HANYA 687 pasangan yang SUDAH LOLOS penyaringan di index lama
(bukan seluruh 2194 pasangan sekuensial): guru LoFTR dijalankan lagi, RANSAC dijalankan
lagi, lalu indeks sel dihitung pada grid baru. Ini menjaga SET PASANGAN tetap identik
dgn yang dipakai Model A/B, sehingga perbandingan data tidak berubah, dan menghemat
~2/3 waktu dibanding menyapu semua folder dari nol.

Disimpan JARANG (`gt_idx2` saja, -1 = tidak valid): matriks GT padat pada grid ini
berukuran 37.7 MB per pasangan -- 26 GB untuk 687 pasangan.

Pakai: python regen_pseudo_gt_p16.py [out_cache_dir]
"""

import json
import os
import sys

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import yaml
from tqdm import tqdm

from generate_pseudo_gt_v2 import compute_gt_tensors

SRC_INDEX = "pseudo_gt_cache/pseudo_gt_index.json"
DEFAULT_OUT = "pseudo_gt_cache_p16"
W_FEAT, H_FEAT = 64, 48


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    with open("config.yaml") as f:
        pcfg = yaml.safe_load(f)["pseudo_gt"]
    entries = json.load(open(SRC_INDEX))
    print(f"[regen] {len(entries)} pasangan dari index lama, target grid {H_FEAT}x{W_FEAT}")

    device = torch.device("cuda")
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    new_index, n_drop = [], 0
    with torch.no_grad():
        for e in tqdm(entries, desc="pseudo-GT /16"):
            f1, f2 = e["img1_path"], e["img2_path"]
            im1, im2 = cv2.imread(f1), cv2.imread(f2)
            if im1 is None or im2 is None:
                n_drop += 1
                continue
            h1, w1 = im1.shape[:2]
            h2, w2 = im2.shape[:2]
            g1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (800, 600))
            g2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (800, 600))
            t1 = K.image_to_tensor(g1, keepdim=False).float().to(device) / 255.0
            t2 = K.image_to_tensor(g2, keepdim=False).float().to(device) / 255.0

            corr = matcher({"image0": t1, "image1": t2})
            p1 = corr["keypoints0"].cpu().numpy() * np.array([w1 / 800.0, h1 / 600.0])
            p2 = corr["keypoints1"].cpu().numpy() * np.array([w2 / 800.0, h2 / 600.0])
            if len(p1) < pcfg["min_raw_matches"]:
                n_drop += 1
                continue

            Hm, mask = cv2.findHomography(p1, p2, cv2.RANSAC, pcfg["ransac_reproj_threshold"])
            inl = int(mask.sum()) if mask is not None else 0
            ratio = inl / len(p1)
            if ratio < pcfg["min_inlier_ratio"]:
                n_drop += 1
                continue
            keep = mask.ravel().astype(bool)

            _, gt_idx2, _, valid_mask = compute_gt_tensors(
                p1[keep], p2[keep], w1, h1, w2, h2, W_feat=W_FEAT, H_feat=H_FEAT)
            sparse_idx = torch.where(valid_mask > 0, gt_idx2, torch.full_like(gt_idx2, -1))

            name = f"{os.path.basename(os.path.dirname(f1))}__" \
                   f"{os.path.splitext(os.path.basename(f1))[0]}__" \
                   f"{os.path.splitext(os.path.basename(f2))[0]}_gt16.pt"
            path = os.path.join(out_dir, name)
            torch.save({"gt_idx2": sparse_idx}, path)
            new_index.append(dict(img1_path=f1, img2_path=f2, gt_path=path,
                                  flight=e.get("flight"), inlier_ratio=float(ratio),
                                  num_valid=int((sparse_idx >= 0).sum())))

    idx_path = os.path.join(out_dir, "pseudo_gt_index.json")
    json.dump(new_index, open(idx_path, "w"), indent=2)
    valid = [x["num_valid"] for x in new_index]
    print(f"[regen] disimpan {len(new_index)} pasangan (dibuang {n_drop}) -> {idx_path}")
    print(f"[regen] sel valid per pasangan: rata-rata {np.mean(valid):.1f} "
          f"dari {W_FEAT*H_FEAT} (min {min(valid)}, maks {max(valid)})")


if __name__ == "__main__":
    main()
