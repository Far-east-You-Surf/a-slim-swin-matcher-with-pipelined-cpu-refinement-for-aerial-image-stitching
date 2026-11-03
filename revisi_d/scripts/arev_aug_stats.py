"""Revisi A IJIES -- augmentasi nyata pelatihan (cacat no. 8: naskah menyebut "rotation within ±20°").

Homografi acak = dataset_homography.HomographyOrthoDataset.get_random_homography (dipanggil apa adanya): tiap dari
empat sudut potongan 1024x768 digeser randint(-150, 150) piksel pada x dan y. Untuk 100.000 sampel (benih 0) dihitung,
dari Jacobian H di pusat potongan (dekomposisi polar), rotasi tersirat, faktor skala sqrt(det), dan pergeseran pusat.
Keluaran: revisi_d/results/arev/t0/aug_stats.json. Pakai: python revisi_d/scripts/arev_aug_stats.py
"""
import json
import os
import sys

import numpy as np
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
sys.path.insert(0, ROOT)


def main():
    from dataset_homography import HomographyOrthoDataset as HD
    np.random.seed(0)
    c = np.array([512.0, 384.0])
    rot, sc, tr = [], [], []
    for _ in range(100000):
        H = HD.get_random_homography(None)
        p = H @ np.array([c[0], c[1], 1.0])
        w = p[2]
        J = (H[:2, :2] - np.outer(p[:2] / w, H[2, :2])) / w          # Jacobian of the projective map at the centre
        U, S, Vt = np.linalg.svd(J)
        R = U @ Vt
        rot.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        sc.append(np.sqrt(abs(np.linalg.det(J))))
        tr.append(np.linalg.norm(p[:2] / w - c))
    rot, sc, tr = map(np.array, (rot, sc, tr))
    res = dict(n=len(rot), corner_shift_px=150, rot_sd_deg=float(rot.std()), rot_abs_p99_deg=float(np.percentile(abs(rot), 99)),
               rot_abs_max_deg=float(abs(rot).max()), scale_sd=float(sc.std()), scale_p1_p99=np.percentile(sc, [1, 99]).tolist(),
               centre_shift_sd_px=float(np.sqrt((tr ** 2).mean() / 2)), centre_shift_p99_px=float(np.percentile(tr, 99)))
    os.makedirs(f"{ROOT}/revisi_d/results/arev/t0", exist_ok=True)
    json.dump(res, open(f"{ROOT}/revisi_d/results/arev/t0/aug_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
