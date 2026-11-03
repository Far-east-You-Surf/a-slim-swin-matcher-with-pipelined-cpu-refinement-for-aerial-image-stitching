"""Hitung ULANG kewarasan homografi, panjang rantai, dan drift dari H yang tersimpan.

GALAT YANG DIPERBAIKI
---------------------
`h_sanity` semula menolak pasangan bila `cond(H) > 1e7`. Itu SALAH secara mendasar:
homografi terdefinisi hingga faktor skala, dan komponen translasinya bernilai
ribuan piksel, sehingga bilangan kondisinya besar terlepas dari sebagus apa
transformasinya. Ia bukan ukuran pengondisian yang sah dalam koordinat piksel.

Akibatnya nyata: pada G6 sebanyak 17-20 pasangan per model ditolak karena kriteria
ini, termasuk pasangan LoFTR dgn 512 inlier, rasio 30,8% dan area ratio 1,414 --
geometri yang sepenuhnya wajar (cond 1,05e7, tepat di atas ambang).

UJI PENGGANTI -- plausibilitas homografi baku:
  1. det(H) tidak nol (tidak degenerate),
  2. keempat sudut citra yang di-warp membentuk segiempat KONVEKS dgn orientasi
     konsisten (semua hasil kali silang bertanda sama) -- ini uji yang benar,
     karena homografi yang melipat atau membalik gambar pasti tidak sahih,
  3. rasio luas hasil warp terhadap luas asli dalam [0,2 , 5,0].
Bilangan kondisi DIBUANG sepenuhnya.

Tidak butuh GPU dan tidak mengulang inferensi: seluruh H sudah tersimpan di JSON.
Pakai: python recompute_sanity.py [pola_glob]
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

WIN = 10
AREA_LO, AREA_HI = 0.2, 5.0
MIN_INLIERS = 8


def corners_ok(H, w, h):
    """Segiempat hasil warp harus konveks, tidak terbalik, dan luasnya wajar."""
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    try:
        p = cv2.perspectiveTransform(c, np.asarray(H, np.float64)).reshape(-1, 2)
    except cv2.error:
        return False, None, None
    if not np.all(np.isfinite(p)):
        return False, None, None
    cross = []
    for i in range(4):
        a, b, cc = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append(np.cross(b - a, cc - b))
    signs = np.sign(cross)
    convex = bool(np.all(signs > 0) or np.all(signs < 0))
    area = float(abs(cv2.contourArea(p.astype(np.float32))))
    ratio = area / (w * h) if w * h else 0.0
    return convex, ratio, p


def sanity(H, w, h):
    if H is None:
        return dict(valid=False, reason="tanpa H")
    Hm = np.asarray(H, np.float64)
    det = float(np.linalg.det(Hm))
    if abs(det) < 1e-8:
        return dict(valid=False, reason="det~0", det=det)
    convex, ratio, _ = corners_ok(Hm, w, h)
    if ratio is None:
        return dict(valid=False, reason="warp gagal", det=det)
    ok = convex and (AREA_LO < ratio < AREA_HI)
    return dict(valid=bool(ok), det=det, convex=convex, warp_area_ratio=ratio,
                reason=None if ok else ("tidak konveks" if not convex else "luas di luar rentang"))


def umeyama_sim(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    c = float((D * np.diag(S)).sum() / var) if var > 1e-12 else 1.0
    return c, R, mu_d - c * R @ mu_s


def windowed_drift(Hs, gps, wh):
    if gps is None or len(Hs) < WIN:
        return None
    vals = []
    c0 = np.array([[[wh[0] / 2.0, wh[1] / 2.0]]], np.float64)
    for st in range(0, len(Hs) - WIN + 1):
        if any(Hs[j] is None for j in range(st, st + WIN)):
            continue
        cw, cm = [], np.eye(3)
        cw.append(cv2.perspectiveTransform(c0, cm).reshape(2))
        bad = False
        for j in range(st, st + WIN):
            cm = cm @ np.linalg.inv(np.asarray(Hs[j], np.float64))
            pt = cv2.perspectiveTransform(c0, cm).reshape(2)
            if not np.all(np.isfinite(pt)):
                bad = True
                break
            cw.append(pt)
        if bad:
            continue
        est, ref = np.array(cw), gps[st:st + WIN + 1]
        if len(ref) != len(est):
            continue
        try:
            c, R, t = umeyama_sim(est, ref)
        except np.linalg.LinAlgError:
            continue
        e = np.linalg.norm((c * (R @ est.T)).T + t - ref, axis=1)
        vals.append(dict(start=st, rmse_m=float(np.sqrt(np.mean(e ** 2)))))
    if not vals:
        return None
    rm = sorted(v["rmse_m"] for v in vals)
    return dict(window=WIN, n_windows=len(vals), median_rmse_m=rm[len(rm) // 2],
                per_window=vals)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "gate_arch_results/*/stitch__*.json"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stitch_demo import FLIGHTS, read_gps_xy

    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        if not any(p.get("H") for v in d["results"].values() if "pairs" in v
                   for p in v["pairs"]):
            continue
        folder = FLIGHTS.get(d["flight"], d["flight"])
        files = sorted(os.path.join(folder, x) for x in os.listdir(folder)
                       if x.lower().endswith((".jpg", ".jpeg")))
        gps = read_gps_xy(files)
        print("=" * 92)
        print(f"{os.path.basename(f)}  jalur={d['flight']} refine={d['refine']}")
        print("=" * 92)
        for model, v in d["results"].items():
            if "pairs" not in v:
                continue
            Hs, n_ok_old, n_ok_new, rescued = [], 0, 0, 0
            for p in v["pairs"]:
                old = bool(p.get("success"))
                n_ok_old += old
                s = sanity(p.get("H"), *(p.get("image_wh") or [1, 1]))
                new = bool(s["valid"] and (p.get("n_inliers") or 0) >= MIN_INLIERS)
                p["sanity_v2"], p["success_v2"] = s, new
                n_ok_new += new
                rescued += (new and not old)
                Hs.append(np.asarray(p["H"], np.float64) if (p.get("H") and new) else None)
            chain = 0
            for H in Hs:
                if H is None:
                    break
                chain += 1
            dw = windowed_drift(Hs, gps, v["pairs"][0].get("image_wh", [1, 1]))
            v["drift_windowed_v2"], v["chain_length_v2"] = dw, chain
            n = len(v["pairs"])
            print(f"  {model:14} sukses {100*n_ok_old/n:5.1f}% -> {100*n_ok_new/n:5.1f}% "
                  f"(+{rescued} diselamatkan) | rantai {v['chain_length']} -> {chain} | "
                  f"drift {(v.get('drift_windowed') or {}).get('median_rmse_m', float('nan')):6.2f} -> "
                  f"{dw['median_rmse_m'] if dw else float('nan'):6.2f} m "
                  f"(n={dw['n_windows'] if dw else 0})")
        out = f.replace(".json", "_resanity.json")
        json.dump(d, open(out, "w"), indent=2)
        print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
