"""Revisi A IJIES -- G6: audit mutu pseudo-label (butir R2.4) pada UseGeo yang punya GT LiDAR.

Prosedur guru IDENTIK dengan pembuatan label latih (regen_pseudo_gt_p16.py): kornia LoFTR outdoor pada abu-abu 800x600,
skala balik, RANSAC 5 px (setRNGSeed 0), saringan min_raw_matches dan min_inlier_ratio, label sel kisi 48x64
(compute_gt_tensors). Diukur pada pasangan UseGeo sepanjang jalur d = 1 dan 2:
  lolos saringan (%), galat kisi-H homografi guru terhadap GT, dan untuk tiap sel berlabel: apakah sel tujuan label sama
  dengan sel tujuan GT LiDAR (pusat sel sumber diproyeksikan dengan kedalaman + pose GT), atau bertetangga (±1 sel).
Keluaran: revisi_d/results/arev/g6/audit.json (+ jsonl per pasangan)
"""
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g6"


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, RD)
    sys.path.insert(0, f"{RD}/scripts")
    import cv2
    import kornia as K
    import kornia.feature as KF
    import numpy as np
    import torch
    import yaml
    from crefine import data as D
    from crefine import geom as Gm
    from generate_pseudo_gt_v2 import compute_gt_tensors
    import e13_refine as E
    os.makedirs(OUT, exist_ok=True)
    pcfg = yaml.safe_load(open("config.yaml"))["pseudo_gt"]
    dev = torch.device("cuda")
    matcher = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    cache = D.DepthCache()
    res = {}
    for d in (1, 2):
        jl = f"{OUT}/usegeo_d{d}.jsonl"
        if os.path.exists(jl + ".done"):
            recs = [json.loads(l) for l in open(jl)]
        else:
            recs = []
            pairs = E.along_pairs(d, (1, 2, 3))
            with open(jl + ".tmp", "w") as fj:
                for k, (ds, A, B) in enumerate(pairs, 1):
                    f1, f2 = D.img_path(ds, A[0]), D.img_path(ds, B[0])
                    im1, im2 = cv2.imread(f1), cv2.imread(f2)
                    (h1, w1), (h2, w2) = im1.shape[:2], im2.shape[:2]
                    g1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (800, 600))
                    g2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (800, 600))
                    with torch.no_grad():
                        corr = matcher({"image0": K.image_to_tensor(g1, keepdim=False).float().to(dev) / 255.0,
                                        "image1": K.image_to_tensor(g2, keepdim=False).float().to(dev) / 255.0})
                    p1 = corr["keypoints0"].cpu().numpy() * np.array([w1 / 800.0, h1 / 600.0])
                    p2 = corr["keypoints1"].cpu().numpy() * np.array([w2 / 800.0, h2 / 600.0])
                    r = dict(dataset=ds, frame_a=A[0], d=d, n_raw=int(len(p1)))
                    ok = len(p1) >= pcfg["min_raw_matches"]
                    Hm, keep = None, np.zeros(len(p1), bool)
                    if ok:
                        cv2.setRNGSeed(0)
                        Hm, mask = cv2.findHomography(p1, p2, cv2.RANSAC, pcfg["ransac_reproj_threshold"])
                        if Hm is not None:
                            keep = mask.ravel().astype(bool)
                        ok = Hm is not None and keep.sum() / len(p1) >= pcfg["min_inlier_ratio"]
                    r["passed"] = bool(ok)
                    r["inlier_ratio"] = float(keep.sum() / max(len(p1), 1))
                    depthA = cache.get(ds, A[0])
                    hg = Gm.HGridGT(A, B, depthA)
                    r["hgrid_teacher"] = hg.err(Hm) if Hm is not None else None
                    if ok:
                        _, gt_idx2, _, valid = compute_gt_tensors(p1[keep], p2[keep], w1, h1, w2, h2, W_feat=64, H_feat=48)
                        lab = np.where(valid.numpy() > 0, gt_idx2.numpy(), -1)
                        src = np.flatnonzero(lab >= 0)
                        sx, sy = (src % 64 + 0.5) * w1 / 64, (src // 64 + 0.5) * h1 / 48
                        ub, vb, vv = D.gt_project(A, B, depthA, sx, sy)
                        inb = vv & (ub >= 0) & (ub < w2) & (vb >= 0) & (vb < h2)
                        gtc = np.where(inb, (np.floor(vb / (h2 / 48)) * 64 + np.floor(ub / (w2 / 64))).astype(int), -1)
                        lx, ly = lab[src] % 64, lab[src] // 64
                        gx, gy = gtc % 64, gtc // 64
                        same = inb & (lab[src] == gtc)
                        near = inb & (np.abs(lx - gx) <= 1) & (np.abs(ly - gy) <= 1)
                        r.update(n_labels=int(len(src)), n_gt=int(inb.sum()), label_exact=int(same.sum()),
                                 label_within1=int(near.sum()))
                    recs.append(r)
                    fj.write(json.dumps(r) + "\n")
                    if k % 50 == 0:
                        print(f"PROGRESS {k + (d - 1) * 729}/1418 audit pseudo-label d{d}", flush=True)
            os.replace(jl + ".tmp", jl)
            open(jl + ".done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        P = [r for r in recs if r["passed"]]
        hg = np.array([r["hgrid_teacher"] for r in P if r["hgrid_teacher"] is not None])
        nl = sum(r.get("n_gt", 0) for r in P)
        res[f"d{d}"] = dict(n=len(recs), passed=len(P), passed_pct=100 * len(P) / max(len(recs), 1),
                            hgrid_teacher_med=float(np.median(hg)) if len(hg) else None,
                            hgrid_teacher_p90=float(np.percentile(hg, 90)) if len(hg) else None,
                            label_exact_pct=100 * sum(r.get("label_exact", 0) for r in P) / max(nl, 1),
                            label_within1_pct=100 * sum(r.get("label_within1", 0) for r in P) / max(nl, 1),
                            labels_per_pair_med=float(np.median([r.get("n_labels", 0) for r in P])) if P else None)
        print(json.dumps({f"d{d}": res[f"d{d}"]}), flush=True)
    json.dump(res, open(f"{OUT}/audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
