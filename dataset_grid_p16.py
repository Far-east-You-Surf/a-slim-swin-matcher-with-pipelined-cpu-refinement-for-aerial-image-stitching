"""
Dataset untuk arsitektur grid /16 (48x64 = 3072 token).

Dua perubahan wajib dibanding `dataset_homography.py`:

  1. GRID DIPARAMETERKAN. Versi lama meng-hardcode `W_feat, H_feat = 32, 24`.

  2. GROUND TRUTH DIKEMBALIKAN SECARA JARANG (SPARSE), bukan matriks N x N.
     Pada 768 token matriks GT itu 768^2 x 4 byte = 2.4 MB/sampel -- masih wajar.
     Pada 3072 token menjadi 3072^2 x 4 byte = **37.7 MB/sampel**, dan dgn
     num_workers=12 x prefetch itu berarti belasan GB RAM hanya untuk label yang
     isinya <=3072 angka. Diganti `gt_idx2` bentuk (N,) int64, -1 = tidak valid.
     `compute_loss_sparse` di bawah SETARA SECARA MATEMATIS dgn loss lama:
     loss lama = sum_ij -log(conf_ij + eps) * gt_ij / sum(gt), dan karena tiap
     baris valid punya tepat satu 1, itu sama dgn rata-rata -log(conf[i, idx2_i])
     atas baris yang valid.
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from dataset_homography import HomographyOrthoDataset

GRID_HW = (48, 64)          # (H_feat, W_feat) pada resize internal 768x1024


def compute_loss_sparse(conf_matrix, gt_idx2, eps=1e-6):
    """Setara compute_loss lama, tapi tanpa memateralisasi matriks GT N x N."""
    valid = gt_idx2 >= 0
    idx = gt_idx2.clamp(min=0).unsqueeze(-1)
    p = conf_matrix.gather(2, idx).squeeze(-1)
    loss = -torch.log(p + eps) * valid
    return loss.sum() / (valid.sum() + eps)


def _grid_targets(H, crop_w, crop_h, grid_hw):
    """Indeks sel tujuan tiap sel sumber di bawah homografi H. -1 = keluar batas."""
    h_feat, w_feat = grid_hw
    scale_x, scale_y = crop_w / w_feat, crop_h / h_feat
    grid_y, grid_x = np.mgrid[0:h_feat, 0:w_feat]
    px = grid_x * scale_x + scale_x / 2.0
    py = grid_y * scale_y + scale_y / 2.0
    pts1 = np.stack([px.ravel(), py.ravel(), np.ones(px.size)], axis=0)
    pts2 = H.dot(pts1)
    pts2 = pts2 / (pts2[2, :] + 1e-8)
    gx = np.floor(pts2[0, :] / scale_x).astype(np.int64)
    gy = np.floor(pts2[1, :] / scale_y).astype(np.int64)
    valid = (gx >= 0) & (gx < w_feat) & (gy >= 0) & (gy < h_feat)
    return np.where(valid, gy * w_feat + gx, -1)


class HomographyOrthoDatasetP16(HomographyOrthoDataset):
    """Sama dgn induknya (sumber gambar, crop, homografi acak, cache LRU), hanya
    grid GT-nya diparameterkan dan labelnya jarang."""

    def __init__(self, *args, grid_hw=GRID_HW, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_hw = grid_hw

    def __getitem__(self, idx):
        path = self.image_files[np.random.randint(0, len(self.image_files))]
        ortho = self._load_ortho(path)
        h_img, w_img, _ = ortho.shape

        crop_w, crop_h = 1024, 768
        x = np.random.randint(0, w_img - crop_w)
        y = np.random.randint(0, h_img - crop_h)
        img1_cv = ortho[y:y + crop_h, x:x + crop_w].copy()

        H = self.get_random_homography()
        img2_cv = cv2.warpPerspective(img1_cv, H, (crop_w, crop_h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        gt_idx2 = _grid_targets(H, crop_w, crop_h, self.grid_hw)
        return (self.transform(img1_cv), self.transform(img2_cv),
                torch.from_numpy(gt_idx2))


class PseudoGTSparseDataset(Dataset):
    """Pasangan foto mentah nyata (parallax 3D asli) dgn GT jarang pada grid /16,
    dihasilkan `regen_pseudo_gt_p16.py`. Augmentasi ColorJitter disamakan dgn
    dataset sintetis (perbaikan 2026-07-10 yang tidak sempat dipakai run lama)."""

    def __init__(self, index_path, epoch_size=800):
        with open(index_path, "r") as f:
            self.entries = json.load(f)
        if not self.entries:
            raise RuntimeError(f"Index pseudo-GT kosong: {index_path}")
        self.epoch_size = epoch_size
        self.target_wh = (1024, 768)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        e = self.entries[np.random.randint(0, len(self.entries))]
        i1 = cv2.resize(cv2.cvtColor(cv2.imread(e["img1_path"]), cv2.COLOR_BGR2RGB), self.target_wh)
        i2 = cv2.resize(cv2.cvtColor(cv2.imread(e["img2_path"]), cv2.COLOR_BGR2RGB), self.target_wh)
        gt = torch.load(e["gt_path"], weights_only=True)
        return self.transform(i1), self.transform(i2), gt["gt_idx2"]


def build_mixed_sparse(synthetic_dataset, index_path, mix_ratio, enabled=True):
    """Campur data sintetis + pseudo-GT nyata, proporsi seperti `dataset_pseudo_gt`."""
    if not enabled or mix_ratio <= 0 or not os.path.exists(index_path):
        print(f"[data] pseudo-GT DILEWATI (enabled={enabled}, ada={os.path.exists(index_path)})")
        return synthetic_dataset
    n_syn = len(synthetic_dataset)
    n_pseudo = int(round(n_syn * mix_ratio / (1.0 - mix_ratio)))
    pseudo = PseudoGTSparseDataset(index_path, epoch_size=n_pseudo)
    print(f"[data] sintetis {n_syn} + pseudo-GT nyata {n_pseudo} "
          f"(proporsi {n_pseudo / (n_syn + n_pseudo):.2f})")
    return ConcatDataset([synthetic_dataset, pseudo])
