"""Revisi A IJIES -- kelas dataset ber-cache (memmap) yang identik dengan dataset pelatih asli.

Hanya SUMBER PIKSEL yang diganti: dekode JPEG ortho raksasa per sampel -> memmap .npy hasil dekode sekali
(`arev_cache.py`), dan foto pseudo-GT -> .npy 1024x768 yang sudah diubah ukurannya. Urutan pengambilan acak
(np.random), potongan, homografi, ColorJitter, dan label tidak berubah, sehingga untuk RNG yang sama keluaran
kedua loader identik bit demi bit (diuji `revisi_d/tests/test_arev_loader.py`).
"""
import os
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))

import numpy as np
import torch

import dataset_grid_p16 as DG

ROOT = AERIAL_ROOT + ""
CACHE = f"{ROOT}/cache_arev"
OrigOrtho = DG.HomographyOrthoDatasetP16
OrigPseudo = DG.PseudoGTSparseDataset


def cache_key(path):
    d = os.path.basename(os.path.dirname(path)).replace(" ", "_")
    return f"{d}__{os.path.basename(path)}.npy"


class MemmapOrtho(OrigOrtho):
    """Sama persis dengan induknya; hanya sumber piksel = memmap (cache_arev/orthos)."""

    def _load_ortho(self, path):
        arr = self._cache.get(path)
        if arr is None:
            src = f"{CACHE}/orthos/{cache_key(path)}"
            if not os.path.exists(src + ".done"):
                raise FileNotFoundError(f"cache ortho belum ada: {src} (jalankan arev_cache.py)")
            arr = np.load(src, mmap_mode="r")
            self._cache[path] = arr          # memmap tidak memakan RAM; semua berkas boleh tetap terbuka
        return arr


class CachedPseudo(OrigPseudo):
    """Sama dengan induknya (satu np.random.randint per sampel, transform yang sama); foto dari cache .npy."""

    def __getitem__(self, idx):
        e = self.entries[np.random.randint(0, len(self.entries))]
        i1 = np.load(f"{CACHE}/pseudo/{cache_key(e['img1_path'])}")
        i2 = np.load(f"{CACHE}/pseudo/{cache_key(e['img2_path'])}")
        gt = torch.load(e["gt_path"], weights_only=True)
        return self.transform(i1), self.transform(i2), gt["gt_idx2"]


def install():
    """Pasang kelas ber-cache ke modul dataset_grid_p16 (dipanggil sebelum pelatih diimpor)."""
    DG.HomographyOrthoDatasetP16 = MemmapOrtho
    DG.PseudoGTSparseDataset = CachedPseudo


# ---- untuk pelatihan pembanding (arev_tp_train.py): sampel yang SAMA + homografi tiap sampel ------------------------
ARHIDX = f"{ROOT}/pseudo_gt_cache_arev/index_48x64.json"


def _tmat(s):
    """Pemetaan pusat piksel cv2.resize: x' = (x + 0.5) s - 0.5."""
    return np.array([[s[0], 0, 0.5 * s[0] - 0.5], [0, s[1], 0.5 * s[1] - 0.5], [0, 0, 1.0]])


class OrthoH(MemmapOrtho):
    """MemmapOrtho + H acak sampel itu (tanpa tarikan RNG tambahan)."""

    def get_random_homography(self):
        H = super().get_random_homography()
        self._lastH = H
        return H

    def __getitem__(self, idx):
        i1, i2, _ = super().__getitem__(idx)
        return i1, i2, torch.from_numpy(np.asarray(self._lastH, np.float64)), torch.tensor(0)


class PseudoH(CachedPseudo):
    """CachedPseudo (satu np.random.randint per sampel, transform sama) + H guru RANSAC dalam koordinat 1024x768."""
    _hidx = None

    def __getitem__(self, idx):
        import json
        if PseudoH._hidx is None:
            PseudoH._hidx = {(e["img1_path"], e["img2_path"]): e for e in json.load(open(ARHIDX))}
        e = self.entries[np.random.randint(0, len(self.entries))]
        i1 = np.load(f"{CACHE}/pseudo/{cache_key(e['img1_path'])}")
        i2 = np.load(f"{CACHE}/pseudo/{cache_key(e['img2_path'])}")
        h = PseudoH._hidx.get((e["img1_path"], e["img2_path"]))
        if h is None:
            Hc = np.full((3, 3), np.nan)
        else:
            (w1, h1), (w2, h2) = h["size1"], h["size2"]
            Hc = _tmat((1024 / w2, 768 / h2)) @ np.array(h["H_teacher"]) @ np.linalg.inv(_tmat((1024 / w1, 768 / h1)))
        return self.transform(i1), self.transform(i2), torch.from_numpy(Hc), torch.tensor(1)
