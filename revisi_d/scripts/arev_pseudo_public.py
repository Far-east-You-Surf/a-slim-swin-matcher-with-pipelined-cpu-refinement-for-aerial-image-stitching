"""Revisi A IJIES (T2) -- pasangan pseudo-GT dari penerbangan PUBLIK berlisensi CC0 (ODM aukerman, lewis, zoo).

Guru dan saringan IDENTIK dengan regen_pseudo_gt_p16.py / arev_pseudo_kp.py: kornia LoFTR outdoor pada abu-abu 800x600,
skala balik ke ukuran foto, RANSAC 5 px (cv2.setRNGSeed(0) per pasangan), buang < min_raw_matches atau rasio inlier
< min_inlier_ratio (config.yaml pseudo_gt). Pasangan = bingkai berurutan (i, i+1) menurut nama berkas. Bingkai yang GPS
EXIF-nya < 100 km dari situs uji dilewati. Bila lolos > 687 pasangan (jumlah set privat), diambil 687 secara merata.
Keluaran: pseudo_gt_cache_arev_pub/{kp/*.npz, g48x64/*_gt.pt, index_48x64.json} (format indeks sama dengan set privat,
plus H_teacher dan ukuran foto).
Pakai: python revisi_d/scripts/arev_pseudo_public.py
"""
import glob
import json
import math
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import yaml
from PIL import Image

from generate_pseudo_gt_v2 import compute_gt_tensors

FL = AERIAL_DATA + "/data_publik_arev/flights"
OUT = f"{ROOT}/pseudo_gt_cache_arev_pub"
SITES = [(28.0391, -82.6975), (27.9584, -82.8322), (41.0347, -83.3057), (41.2257, -81.7025)]
N_TARGET = 687


def gps(p):
    try:
        g = (Image.open(p)._getexif() or {}).get(34853)
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600  # noqa: E731
        return (dms(g[2]) * (1 if g[1] == "N" else -1), dms(g[4]) * (1 if g[3] == "E" else -1))
    except Exception:  # noqa: BLE001
        return None


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def main():
    pcfg = yaml.safe_load(open("config.yaml"))["pseudo_gt"]
    os.makedirs(f"{OUT}/kp", exist_ok=True)
    os.makedirs(f"{OUT}/g48x64", exist_ok=True)
    pairs = []
    extra = [AERIAL_DATA + "/Data Uji Coba/DroneMapper_Golf9_May2016"]   # DroneMapper Back 9 (Colorado)
    for fl in sorted(d for d in glob.glob(f"{FL}/odm_data_*") if os.path.isdir(d)) + extra:
        imgs = sorted(p for p in glob.glob(f"{fl}/images/*") + glob.glob(f"{fl}/*")
                      if p.lower().endswith((".jpg", ".jpeg")))
        g0 = gps(imgs[0]) if imgs else None
        dmin = min(km(g0, s) for s in SITES) if g0 else None
        print(f"{os.path.basename(fl)}: {len(imgs)} foto, GPS {g0}, jarak min ke situs uji {dmin and round(dmin)} km", flush=True)
        if dmin is not None and dmin < 100:
            print("  dilewati (< 100 km dari situs uji)", flush=True)
            continue
        pairs += [(os.path.basename(fl), imgs[i], imgs[i + 1]) for i in range(len(imgs) - 1)]
    dev = torch.device("cuda")
    matcher = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    kept = []
    for k, (fl, f1, f2) in enumerate(pairs, 1):
        base = f"{fl}__{os.path.splitext(os.path.basename(f1))[0]}__{os.path.splitext(os.path.basename(f2))[0]}"
        kp = f"{OUT}/kp/{base}.npz"
        if not os.path.exists(kp):
            im1, im2 = cv2.imread(f1), cv2.imread(f2)
            (h1, w1), (h2, w2) = im1.shape[:2], im2.shape[:2]
            g1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (800, 600))
            g2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (800, 600))
            t1 = K.image_to_tensor(g1, keepdim=False).float().to(dev) / 255.0
            t2 = K.image_to_tensor(g2, keepdim=False).float().to(dev) / 255.0
            with torch.no_grad():
                corr = matcher({"image0": t1, "image1": t2})
            p1 = corr["keypoints0"].cpu().numpy() * np.array([w1 / 800.0, h1 / 600.0])
            p2 = corr["keypoints1"].cpu().numpy() * np.array([w2 / 800.0, h2 / 600.0])
            ok, Hm, keep = len(p1) >= pcfg["min_raw_matches"], np.zeros((3, 3)), np.zeros(len(p1), bool)
            if ok:
                cv2.setRNGSeed(0)
                H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, pcfg["ransac_reproj_threshold"])
                if H is not None:
                    Hm, keep = H, mask.ravel().astype(bool)
                ok = H is not None and keep.sum() / len(p1) >= pcfg["min_inlier_ratio"]
            np.savez(kp + ".tmp.npz", p1=p1, p2=p2, keep=keep, Hm=Hm, size1=np.array([w1, h1]),
                     size2=np.array([w2, h2]), ok=np.array(ok))
            os.replace(kp + ".tmp.npz", kp)
        z = np.load(kp)
        if bool(z["ok"]):
            kept.append((fl, f1, f2, base))
        if k % 20 == 0:
            print(f"PROGRESS {k}/{len(pairs)} lolos {len(kept)}", flush=True)
    if len(kept) > N_TARGET:
        idx = np.linspace(0, len(kept) - 1, N_TARGET).round().astype(int)
        kept = [kept[i] for i in idx]
    index = []
    for fl, f1, f2, base in kept:
        z = np.load(f"{OUT}/kp/{base}.npz")
        p1, p2 = z["p1"][z["keep"]], z["p2"][z["keep"]]
        (w1, h1), (w2, h2) = z["size1"], z["size2"]
        _, gt_idx2, _, valid = compute_gt_tensors(p1, p2, int(w1), int(h1), int(w2), int(h2), W_feat=64, H_feat=48)
        sp = torch.where(valid > 0, gt_idx2, torch.full_like(gt_idx2, -1))
        path = f"{OUT}/g48x64/{base}_gt.pt"
        torch.save({"gt_idx2": sp}, path + ".tmp")
        os.replace(path + ".tmp", path)
        index.append(dict(img1_path=f1, img2_path=f2, gt_path=path, flight=fl,
                          inlier_ratio=float(z["keep"].sum() / max(len(z["p1"]), 1)), num_valid=int((sp >= 0).sum()),
                          H_teacher=z["Hm"].tolist(), size1=[int(w1), int(h1)], size2=[int(w2), int(h2)]))
    json.dump(index, open(f"{OUT}/index_48x64.json.tmp", "w"), indent=1)
    os.replace(f"{OUT}/index_48x64.json.tmp", f"{OUT}/index_48x64.json")
    per = {}
    for e in index:
        per[e["flight"]] = per.get(e["flight"], 0) + 1
    print(json.dumps(dict(pairs_total=len(pairs), kept=len(index), per_flight=per)), flush=True)
    open(f"{OUT}/all.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
