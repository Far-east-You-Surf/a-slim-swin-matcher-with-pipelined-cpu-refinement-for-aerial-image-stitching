"""Revisi A IJIES -- jalankan ulang guru pseudo-label (regen_pseudo_gt_p16.py) dan simpan titik inlier + H.

Indeks pseudo-GT lama hanya menyimpan sel tujuan pada kisi 48x64, jadi varian kisi lain (30x40 untuk resolusi
internal 480x640, 24x32 untuk kisi /32) dan pelatihan pembanding (supervisi homografi) butuh titik guru aslinya.
Prosedur guru IDENTIK dengan regen_pseudo_gt_p16.py (kornia LoFTR outdoor pada abu-abu 800x600, skala balik ke
ukuran foto, RANSAC 5 px, buang < min_raw_matches atau rasio inlier < min_inlier_ratio), ditambah cv2.setRNGSeed(0)
per pasangan supaya hasilnya dapat diulang. Untuk tiap kisi dibuat indeks JSON + berkas .pt sel tujuan
(compute_gt_tensors yang sama); kesesuaian kisi 48x64 dengan berkas lama dilaporkan di agree_48x64.json.

Pakai: python revisi_d/scripts/arev_pseudo_kp.py
Keluaran: pseudo_gt_cache_arev/{kp/*.npz, g48x64/, g30x40/, g24x32/, index_<grid>.json}
"""
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import yaml

from generate_pseudo_gt_v2 import compute_gt_tensors

SRC = "pseudo_gt_cache_p16/pseudo_gt_index.json"
OUT = "pseudo_gt_cache_arev"
GRIDS = {"48x64": (48, 64), "30x40": (30, 40), "24x32": (24, 32)}


def atomic_save_pt(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    pcfg = yaml.safe_load(open("config.yaml"))["pseudo_gt"]
    entries = json.load(open(SRC))
    os.makedirs(f"{OUT}/kp", exist_ok=True)
    for g in GRIDS:
        os.makedirs(f"{OUT}/g{g}", exist_ok=True)
    dev = torch.device("cuda")
    matcher = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    idx = {g: [] for g in GRIDS}
    agree, n_drop = [], 0
    for k, e in enumerate(entries, 1):
        f1, f2 = e["img1_path"], e["img2_path"]
        base = os.path.basename(e["gt_path"]).replace("_gt16.pt", "")
        kp_path = f"{OUT}/kp/{base}.npz"
        if not os.path.exists(kp_path):
            im1, im2 = cv2.imread(f1), cv2.imread(f2)
            h1, w1 = im1.shape[:2]
            h2, w2 = im2.shape[:2]
            g1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (800, 600))
            g2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (800, 600))
            t1 = K.image_to_tensor(g1, keepdim=False).float().to(dev) / 255.0
            t2 = K.image_to_tensor(g2, keepdim=False).float().to(dev) / 255.0
            with torch.no_grad():
                corr = matcher({"image0": t1, "image1": t2})
            p1 = corr["keypoints0"].cpu().numpy() * np.array([w1 / 800.0, h1 / 600.0])
            p2 = corr["keypoints1"].cpu().numpy() * np.array([w2 / 800.0, h2 / 600.0])
            ok, Hm, keep = len(p1) >= pcfg["min_raw_matches"], None, np.zeros(len(p1), bool)
            if ok:
                cv2.setRNGSeed(0)
                Hm, mask = cv2.findHomography(p1, p2, cv2.RANSAC, pcfg["ransac_reproj_threshold"])
                keep = mask.ravel().astype(bool) if mask is not None else keep
                ok = keep.sum() / len(p1) >= pcfg["min_inlier_ratio"]
            np.savez(kp_path + ".tmp.npz", p1=p1, p2=p2, keep=keep, Hm=(Hm if Hm is not None else np.zeros((3, 3))),
                     size1=np.array([w1, h1]), size2=np.array([w2, h2]), ok=np.array(ok))
            os.replace(kp_path + ".tmp.npz", kp_path)
        z = np.load(kp_path)
        if not bool(z["ok"]):
            n_drop += 1
            print(f"PROGRESS {k}/{len(entries)} {base} DIBUANG", flush=True)
            continue
        p1, p2, keep = z["p1"][z["keep"]], z["p2"][z["keep"]], z["keep"]
        (w1, h1), (w2, h2) = z["size1"], z["size2"]
        for g, (hf, wf) in GRIDS.items():
            path = f"{OUT}/g{g}/{base}_gt.pt"
            _, gt_idx2, _, valid = compute_gt_tensors(p1, p2, int(w1), int(h1), int(w2), int(h2), W_feat=wf, H_feat=hf)
            sparse = torch.where(valid > 0, gt_idx2, torch.full_like(gt_idx2, -1))
            atomic_save_pt({"gt_idx2": sparse}, path)
            idx[g].append(dict(img1_path=f1, img2_path=f2, gt_path=os.path.abspath(path), flight=e.get("flight"),
                               inlier_ratio=float(keep.sum() / max(len(z["p1"]), 1)),
                               num_valid=int((sparse >= 0).sum()), H_teacher=z["Hm"].tolist(),
                               size1=[int(w1), int(h1)], size2=[int(w2), int(h2)]))
            if g == "48x64":
                old = torch.load(e["gt_path"], weights_only=True)["gt_idx2"]
                agree.append(float((old == sparse).float().mean()))
        print(f"PROGRESS {k}/{len(entries)} {base} kesesuaian48 {agree[-1] if agree else float('nan'):.4f}", flush=True)
    for g in GRIDS:
        tmp = f"{OUT}/index_{g}.json.tmp"
        json.dump(idx[g], open(tmp, "w"), indent=1)
        os.replace(tmp, f"{OUT}/index_{g}.json")
    a = np.array(agree)
    summ = dict(n=len(entries), kept=len(idx["48x64"]), dropped=n_drop, agree_mean=float(a.mean()),
                agree_exact_pairs=int((a == 1.0).sum()), agree_min=float(a.min()), t=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(summ, open(f"{OUT}/agree_48x64.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)
    open(f"{OUT}/all.done", "w").write(summ["t"])


if __name__ == "__main__":
    main()
