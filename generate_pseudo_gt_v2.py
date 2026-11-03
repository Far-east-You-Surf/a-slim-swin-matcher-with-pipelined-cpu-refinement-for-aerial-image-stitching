"""
Versi 2 dari generate_pseudo_gt_gpu.py -- generator pseudo-GT dari data mentah
(belum di-stitch) memakai LoFTR (kornia, pretrained 'outdoor') sbg "guru",
utk pasangan foto SEKUENSIAL nyata (bukan crop+homografi sintetis).

Perbaikan dibanding versi lama (generate_pseudo_gt_gpu.py, TIDAK diubah --
dibiarkan apa adanya sesuai konvensi proyek ini):
  1. Multi-folder: tiap flight (Flight 12A, Zieg_09, dst.) dipasangkan HANYA
     dgn tetangganya SENDIRI (sekuensial per folder) -- versi lama menggabung
     semua file dari satu folder train_path scr alfabetis, salah kalau ada
     banyak folder flight terpisah.
  2. Skala grid dihitung dari RESOLUSI ASLI tiap gambar (scale_x = orig_w/32),
     bukan hardcode 4000x3000 -- data kita campuran 4800x3200 (Flight/G6) dan
     4000x3000 (Zieg).
  3. Filter kualitas SEBELUM disimpan (§14.5 poin 1): buang pasangan dgn match
     mentah < min_raw_matches, atau inlier ratio RANSAC < min_inlier_ratio.
  4. Selain gt_matches coarse (24x32), juga menghitung gt_idx2/fine_gt_offset/
     valid_mask -- skema IDENTIK dgn dataset_homography_finematch.py, supaya
     Model B (fine-matching) juga bisa disupervisi dari data real ini, bukan
     cuma Model A.
  5. Menulis index.json yg dikonsumsi dataset_pseudo_gt.py (PseudoGTPairDataset).

Parameter difilter/dikonfigurasi lewat config.yaml -> `pseudo_gt`.
"""
import os
import json
import glob
import cv2
import torch
import numpy as np
import kornia as K
import kornia.feature as KF
import yaml
from tqdm import tqdm


def list_flight_dirs(pre_stitch_path):
    dirs = []
    for name in sorted(os.listdir(pre_stitch_path)):
        p = os.path.join(pre_stitch_path, name)
        if os.path.isdir(p):
            dirs.append(p)
    return dirs


def list_images(flight_dir):
    files = glob.glob(os.path.join(flight_dir, "*.JPG")) + \
        glob.glob(os.path.join(flight_dir, "*.jpg")) + \
        glob.glob(os.path.join(flight_dir, "*.jpeg"))
    return sorted(files)


def compute_gt_tensors(pts1, pts2, orig_w1, orig_h1, orig_w2, orig_h2,
                        W_feat=32, H_feat=24, ratio=4):
    """
    pts1/pts2: np.ndarray [K,2] koordinat piksel di gambar1/gambar2 ASLI
               (sudah discale balik dari resolusi kerja LoFTR).
    Mengembalikan (gt_matches [N,N], gt_idx2 [N], fine_gt_offset [N,2], valid_mask [N])
    dgn skema koordinat identik dataset_homography_finematch.py: tiap sel grid
    koarse img1 dipetakan ke SATU target di img2 (kalau ada match yg jatuh di
    sel itu -- kalau ada >1 keypoint jatuh di sel koarse yg sama, dipakai yg
    PALING PERTAMA ditemukan (deterministik dari urutan LoFTR, cukup utk pseudo-GT)).
    """
    N = W_feat * H_feat
    scale_x1, scale_y1 = orig_w1 / W_feat, orig_h1 / H_feat
    scale_x2, scale_y2 = orig_w2 / W_feat, orig_h2 / H_feat
    fine_scale_x2, fine_scale_y2 = scale_x2 / ratio, scale_y2 / ratio

    gt_matches = torch.zeros((N, N))
    gt_idx2 = torch.zeros(N, dtype=torch.long)
    fine_gt_offset = torch.zeros((N, 2))
    valid_mask = torch.zeros(N)

    if len(pts1) == 0:
        return gt_matches, gt_idx2, fine_gt_offset, valid_mask

    grid1_x = np.floor(pts1[:, 0] / scale_x1).astype(np.int64)
    grid1_y = np.floor(pts1[:, 1] / scale_y1).astype(np.int64)
    grid2_x = np.floor(pts2[:, 0] / scale_x2).astype(np.int64)
    grid2_y = np.floor(pts2[:, 1] / scale_y2).astype(np.int64)

    valid = (grid1_x >= 0) & (grid1_x < W_feat) & (grid1_y >= 0) & (grid1_y < H_feat) & \
            (grid2_x >= 0) & (grid2_x < W_feat) & (grid2_y >= 0) & (grid2_y < H_feat)

    for k in np.nonzero(valid)[0]:
        idx1 = int(grid1_y[k] * W_feat + grid1_x[k])
        if valid_mask[idx1] > 0:
            continue  # sel koarse ini sudah diisi match sebelumnya, skip
        idx2 = int(grid2_y[k] * W_feat + grid2_x[k])

        gt_matches[idx1, idx2] = 1.0
        gt_idx2[idx1] = idx2
        valid_mask[idx1] = 1.0

        fine_center_x = grid2_x[k] * ratio + ratio // 2
        fine_center_y = grid2_y[k] * ratio + ratio // 2
        target_fine_x = pts2[k, 0] / fine_scale_x2
        target_fine_y = pts2[k, 1] / fine_scale_y2
        fine_gt_offset[idx1, 0] = float(target_fine_x - fine_center_x)
        fine_gt_offset[idx1, 1] = float(target_fine_y - fine_center_y)

    return gt_matches, gt_idx2, fine_gt_offset, valid_mask


def generate_pseudo_gt_v2(limit_pairs=None):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pcfg = config["pseudo_gt"]
    pre_stitch_path = config["dataset"]["pre_stitch_path"]
    cache_dir = pcfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pseudo-gt-v2] device={device}")

    matcher = KF.LoFTR(pretrained="outdoor").to(device)
    matcher.eval()

    flight_dirs = list_flight_dirs(pre_stitch_path)
    print(f"[pseudo-gt-v2] {len(flight_dirs)} folder flight ditemukan di {pre_stitch_path}")

    index_entries = []
    n_kept, n_dropped_matches, n_dropped_inlier = 0, 0, 0

    pairs = []
    for flight_dir in flight_dirs:
        files = list_images(flight_dir)
        for i in range(len(files) - 1):
            pairs.append((files[i], files[i + 1]))
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]
    print(f"[pseudo-gt-v2] total {len(pairs)} pasangan sekuensial akan diproses")

    with torch.no_grad():
        for img1_path, img2_path in tqdm(pairs, desc="Pseudo-GT v2"):
            img1_cv_full = cv2.imread(img1_path)
            img2_cv_full = cv2.imread(img2_path)
            if img1_cv_full is None or img2_cv_full is None:
                continue
            h1, w1 = img1_cv_full.shape[:2]
            h2, w2 = img2_cv_full.shape[:2]

            img1_gray = cv2.cvtColor(img1_cv_full, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2_cv_full, cv2.COLOR_BGR2GRAY)
            img1_small = cv2.resize(img1_gray, (800, 600))
            img2_small = cv2.resize(img2_gray, (800, 600))

            img1_t = K.image_to_tensor(img1_small, keepdim=False).float().to(device) / 255.0
            img2_t = K.image_to_tensor(img2_small, keepdim=False).float().to(device) / 255.0

            correspondences = matcher({"image0": img1_t, "image1": img2_t})
            pts1 = correspondences["keypoints0"].cpu().numpy()
            pts2 = correspondences["keypoints1"].cpu().numpy()

            if len(pts1) < pcfg["min_raw_matches"]:
                n_dropped_matches += 1
                continue

            # Skala balik ke resolusi ASLI (per-sumbu, per-gambar -- bukan konstanta 5.0)
            pts1_full = pts1 * np.array([w1 / 800.0, h1 / 600.0])
            pts2_full = pts2 * np.array([w2 / 800.0, h2 / 600.0])

            H_mat, mask = cv2.findHomography(pts1_full, pts2_full, cv2.RANSAC,
                                              pcfg["ransac_reproj_threshold"])
            inliers = int(mask.sum()) if mask is not None else 0
            inlier_ratio = inliers / len(pts1_full)
            if inlier_ratio < pcfg["min_inlier_ratio"]:
                n_dropped_inlier += 1
                continue

            inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts1_full), dtype=bool)
            pts1_in = pts1_full[inlier_mask]
            pts2_in = pts2_full[inlier_mask]

            gt_matches, gt_idx2, fine_gt_offset, valid_mask = compute_gt_tensors(
                pts1_in, pts2_in, w1, h1, w2, h2)

            pair_id = f"{os.path.splitext(os.path.basename(img1_path))[0]}__{os.path.splitext(os.path.basename(img2_path))[0]}"
            flight_name = os.path.basename(os.path.dirname(img1_path))
            gt_filename = f"{flight_name}__{pair_id}_gt.pt"
            gt_path = os.path.join(cache_dir, gt_filename)

            torch.save({
                "gt_matches": gt_matches,
                "gt_idx2": gt_idx2,
                "fine_gt_offset": fine_gt_offset,
                "valid_mask": valid_mask,
            }, gt_path)

            index_entries.append({
                "img1_path": img1_path,
                "img2_path": img2_path,
                "gt_path": gt_path,
                "flight": flight_name,
                "num_raw_matches": int(len(pts1_full)),
                "inlier_ratio": float(inlier_ratio),
                "num_valid_coarse": int(valid_mask.sum().item()),
            })
            n_kept += 1

    index_path = os.path.join(cache_dir, "pseudo_gt_index.json")
    with open(index_path, "w") as f:
        json.dump(index_entries, f, indent=2)

    print(f"\n[pseudo-gt-v2] Selesai. Disimpan: {n_kept} pasangan.")
    print(f"[pseudo-gt-v2] Dibuang (match mentah < {pcfg['min_raw_matches']}): {n_dropped_matches}")
    print(f"[pseudo-gt-v2] Dibuang (inlier ratio < {pcfg['min_inlier_ratio']}): {n_dropped_inlier}")
    print(f"[pseudo-gt-v2] Index: {index_path}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_pseudo_gt_v2(limit_pairs=limit)
