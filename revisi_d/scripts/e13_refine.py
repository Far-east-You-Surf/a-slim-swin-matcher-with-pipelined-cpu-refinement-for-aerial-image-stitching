"""E1.3 -- penghalusan sub-sel TANPA pelatihan pada UseGeo (rencana revisi Artikel D).

Pertanyaan arsitektural: apakah lantai kuantisasi (korespondensi hanya terlokalisasi ke sel kisi)
adalah batas LANGKAH KELUARAN, atau dapat diatasi pascaproses.

Varian (diterapkan pada titik target b; titik sumber a = pusat sel, GT-nya eksak):
  none  pusat sel (keluaran apa adanya, = protokol registrasi)
  A     soft-argmax 3x3 pada baris matriks keyakinan di sekitar sel target terbaik
  B     pra-pelengkungan: H0 = MAGSAC(a,b) -> lengkungkan citra 2 ke bingkai citra 1 ->
        LK berinisialisasi NOL (a -> a') -> petakan balik b' = H0(a')
  C     seperti B, NCC + parabola menggantikan LK (opsional)

KONTROL (wajib sebelum uji pada pencocok): korespondensi GT dibulatkan ke pusat sel /16 dan /32;
setiap varian harus MENURUNKAN galat median terhadap versi bulat. Varian A diuji dgn matriks keyakinan
sintetis Gaussian di sekitar posisi GT sejati (menguji implementasinya). Varian yang gagal digugurkan.

Metrik per (model, varian, d), pasangan UseGeo sepanjang-jalur, dataset 1-3:
  med_err_inbounds (median galat GT, GT di dalam bingkai B), frac2 (benar < 2% lebar), gerbang v2,
  AUC pose @5/10/20 derajat (pose_err = max(rot, trans), gagal = 180).

Pakai: python revisi_d/scripts/e13_refine.py control | run [--d 1,2,...] [--models ...] [--limit N]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revisi_d", "scripts"))

import cv2
import numpy as np
import torch

import stitch_demo as SD
import usegeo_gt_eval as UG
from recompute_sanity import sanity as gate_v2_fn

OUT = "revisi_d/results/e13"
os.makedirs(OUT, exist_ok=True)
MODELS = {"baseline_p32": ("p32", "aerial_stitching_fp16_best.pth", None, (24, 32)),
          "p16_nofpn": ("p16", "ckpt_archive_p16_nofpn/ep008_val4.3737.pth", "p16_depth8", (48, 64)),
          "p16_depth6": ("p16", "gate_arch_results/p16_depth6_pruned.pth", "p16_depth6", (48, 64)),
          "p16_depth4": ("p16", "gate_arch_results/p16_depth4_pruned.pth", "p16_depth4", (48, 64))}
VARIANTS = ["none", "A", "B", "C"]
THR2 = 0.02 * UG.IMG_W


# ------------------------------------------------------------------ varian
def softargmax3(conf_rows, i2, grid, ow, oh):
    """conf_rows: [n, N] baris keyakinan utk tiap kecocokan; i2: indeks sel target terbaik."""
    h, w = grid
    cy, cx = i2 // w, i2 % w
    out = np.zeros((len(i2), 2))
    for k in range(len(i2)):
        ys = np.clip(np.arange(cy[k] - 1, cy[k] + 2), 0, h - 1)
        xs = np.clip(np.arange(cx[k] - 1, cx[k] + 2), 0, w - 1)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        wts = conf_rows[k][(Y * w + X).ravel()].astype(np.float64)
        wts = np.clip(wts, 0, None)
        s = wts.sum()
        if s <= 0:
            ex, ey = cx[k], cy[k]
        else:
            ex, ey = (X.ravel() * wts).sum() / s, (Y.ravel() * wts).sum() / s
        out[k] = [(ex + 0.5) * ow / w, (ey + 0.5) * oh / h]
    return out


def prewarp_refine(g1, g2, a, b, H0, method):
    """Lengkungkan citra 2 ke bingkai 1 dgn H0^-1, haluskan a->a' (init nol), petakan balik."""
    hh, ww = g1.shape
    try:
        g2w = cv2.warpPerspective(g2, np.linalg.inv(H0), (ww, hh), flags=cv2.INTER_LINEAR)
    except (cv2.error, np.linalg.LinAlgError):
        return b.copy(), 0
    if method == "lk":
        p0 = a.astype(np.float32).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2w, p0, p0.copy(), winSize=(21, 21), maxLevel=3,
                                             criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        st = st.reshape(-1).astype(bool)
        ap = a.copy()
        ap[st] = p1.reshape(-1, 2)[st]
        n_ok = int(st.sum())
    else:
        c1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
        c2 = cv2.cvtColor(g2w, cv2.COLOR_GRAY2BGR)
        _, ap, n_ok = SD.refine_ncc(c1, c2, a, a.copy())
    bp = cv2.perspectiveTransform(ap.reshape(-1, 1, 2).astype(np.float64), H0).reshape(-1, 2)
    # titik yang tidak berhasil dihaluskan kembali ke b semula (bukan ke H0(a))
    moved = np.any(np.abs(ap - a) > 1e-9, axis=1)
    out = b.copy()
    out[moved] = bp[moved]
    return out, n_ok


# ------------------------------------------------------------------ metrik
def gt_err(a, b, A, B, depthA):
    ub, vb, valid = UG.gt_project(A, B, depthA, a[:, 0], a[:, 1])
    inb = valid & (ub >= 0) & (ub < UG.IMG_W) & (vb >= 0) & (vb < UG.IMG_H)
    err = np.hypot(ub - b[:, 0], vb - b[:, 1])
    return err, valid, inb


def score(a, b, ow, oh, A, B, depthA):
    r = dict(n=int(len(a)))
    err, valid, inb = gt_err(a, b, A, B, depthA)
    if inb.sum():
        r["med_err_inb"] = float(np.median(err[inb]))
    if valid.sum():
        r["frac2"] = float((err[valid] < THR2).mean())
        r["n_correct2"] = int((err[valid] < THR2).sum())
    H, m = (cv2.findHomography(a, b, cv2.USAC_MAGSAC, SD.RANSAC_THR) if len(a) >= 8 else (None, None))
    inl = int(m.sum()) if m is not None else 0
    r["n_inliers"] = inl
    r["gate_v2"] = bool(H is not None and gate_v2_fn(H, ow, oh)["valid"] and inl >= 8)
    if len(a) >= 6:
        E, em = cv2.findEssentialMat(a, b, UG.K, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=1.0)
        if E is not None and E.shape == (3, 3):
            _, R_e, t_e, _ = cv2.recoverPose(E, a, b, UG.K, mask=em)
            re, te = UG.pose_error(R_e, t_e, *UG.gt_relpose(A, B))
            r["pose_err"] = max(re, te)
    return r


def pose_auc(errs, ths=(5, 10, 20)):
    """AUC kurva kumulatif galat pose (konvensi SuperGlue/LoFTR), gagal = 180 derajat."""
    e = np.sort(np.asarray(errs, float))
    rec = np.arange(1, len(e) + 1) / len(e)
    e = np.r_[0.0, e]
    rec = np.r_[0.0, rec]
    out = {}
    for t in ths:
        k = np.searchsorted(e, t)
        x = np.r_[e[:k], t]
        y = np.r_[rec[:k], rec[k - 1]]
        out[f"auc@{t}"] = float(np.trapezoid(y, x) / t * 100)
    return out


# ------------------------------------------------------------------ data
def along_pairs(d, datasets=(1, 2, 3)):
    out = []
    for ds in datasets:
        rows = UG.load_poses(ds)
        for i in range(len(rows) - d):
            ok = all(UG.dkappa(rows[k][6], rows[k + 1][6]) < UG.DKAPPA_MAX and
                     np.hypot(rows[k][1] - rows[k + 1][1], rows[k][2] - rows[k + 1][2]) < UG.STEP_MAX
                     for k in range(i, i + d))
            if ok:
                out.append((ds, rows[i], rows[i + d]))
    return out


def gray(p):
    im = SD._imread(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------ kontrol
def control(n_pairs=30):
    cache = UG.DepthCache()
    res = {}
    for gname, grid in (("/16", (48, 64)), ("/32", (24, 32))):
        acc = {v: [] for v in VARIANTS}
        for ds, A, B in along_pairs(1)[:: max(1, len(along_pairs(1)) // n_pairs)][:n_pairs]:
            depthA = cache.get(ds, A[0])
            pA, pB = UG.img_path(ds, A[0]), UG.img_path(ds, B[0])
            g1, g2 = gray(pA), gray(pB)
            oh, ow = g1.shape
            h, w = grid
            gy, gx = np.mgrid[0:h, 0:w]
            a = np.stack([(gx.ravel() + 0.5) * ow / w, (gy.ravel() + 0.5) * oh / h], 1)
            ub, vb, valid = UG.gt_project(A, B, depthA, a[:, 0], a[:, 1])
            inb = valid & (ub >= 0) & (ub < ow) & (vb >= 0) & (vb < oh)
            a, btrue = a[inb], np.stack([ub[inb], vb[inb]], 1)
            cx = np.clip(np.floor(btrue[:, 0] / ow * w), 0, w - 1).astype(int)
            cy = np.clip(np.floor(btrue[:, 1] / oh * h), 0, h - 1).astype(int)
            bround = np.stack([(cx + 0.5) * ow / w, (cy + 0.5) * oh / h], 1)
            i2 = cy * w + cx
            # keyakinan sintetis: Gaussian (sigma = setengah sel) di sekitar posisi GT sejati
            cc = np.stack([(gx.ravel() + 0.5) * ow / w, (gy.ravel() + 0.5) * oh / h], 1)
            sig = 0.5 * ow / w
            d2 = ((cc[None, :, :] - btrue[:, None, :]) ** 2).sum(-1)
            conf = np.exp(-d2 / (2 * sig ** 2))
            H0, _ = cv2.findHomography(a, bround, cv2.USAC_MAGSAC, SD.RANSAC_THR)
            outs = {"none": bround, "A": softargmax3(conf, i2, grid, ow, oh)}
            if H0 is not None:
                outs["B"], _ = prewarp_refine(g1, g2, a, bround, H0, "lk")
                outs["C"], _ = prewarp_refine(g1, g2, a, bround, H0, "ncc")
            for v, bb in outs.items():
                acc[v].append(float(np.median(np.hypot(*(bb - btrue).T))))
        res[gname] = {v: dict(median_of_medians=float(np.median(x)), n_pairs=len(x)) for v, x in acc.items() if x}
        base = res[gname]["none"]["median_of_medians"]
        for v in res[gname]:
            res[gname][v]["lolos"] = bool(v == "none" or res[gname][v]["median_of_medians"] < base)
    json.dump(res, open(f"{OUT}/control.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ uji
def run(ds_list, d_list, models, variants, limit=0):
    cache = UG.DepthCache()
    summary = {}
    for mname in models:
        kind, ckpt, variant, grid = MODELS[mname]
        m = SD.OurModel(mname, kind, ckpt, variant=variant, grid=grid)
        for d in d_list:
            done_f = f"{OUT}/{mname}__d{d}__{'+'.join(variants)}.done.json"
            if os.path.exists(done_f):                      # tahan-putus: unit selesai dilewati
                summary.update(json.load(open(done_f)))
                print("lewat (sudah selesai):", done_f, flush=True)
                continue
            pairs = along_pairs(d, ds_list)
            if limit:
                pairs = pairs[:limit]
            recs = {v: [] for v in variants}
            fj = open(f"{OUT}/{mname}__d{d}.jsonl", "w")
            for ds, A, B in pairs:
                pA, pB = UG.img_path(ds, A[0]), UG.img_path(ds, B[0])
                t1, t2, ow, oh = SD.load_pair_tensor(pA, pB)
                with torch.no_grad():
                    conf = m.model(t1.to(m.device).half(), t2.to(m.device).half()).float().cpu()
                c = conf[0]
                bj, bi = c.argmax(1), c.argmax(0)
                rows = torch.arange(c.shape[0])
                keep = bi[bj[rows]] == rows
                i1, i2 = rows[keep].numpy(), bj[rows[keep]].numpy()
                h, w = grid
                a = np.stack([(i1 % w + 0.5) * ow / w, (i1 // w + 0.5) * oh / h], 1).astype(np.float64)
                b = np.stack([(i2 % w + 0.5) * ow / w, (i2 // w + 0.5) * oh / h], 1).astype(np.float64)
                depthA = cache.get(ds, A[0])
                outs = {"none": b}
                if "A" in variants:
                    outs["A"] = softargmax3(c[i1].numpy(), i2, grid, ow, oh)
                if "B" in variants or "C" in variants:
                    H0, _ = (cv2.findHomography(a, b, cv2.USAC_MAGSAC, SD.RANSAC_THR)
                             if len(a) >= 8 else (None, None))
                    g1, g2 = gray(pA), gray(pB)
                    for v, meth in (("B", "lk"), ("C", "ncc")):
                        if v in variants:
                            outs[v] = prewarp_refine(g1, g2, a, b, H0, meth)[0] if H0 is not None else b
                for v in variants:
                    r = score(a, outs[v], ow, oh, A, B, depthA)
                    r.update(model=mname, variant=v, d=d, dataset=ds, frame_a=A[0], frame_b=B[0])
                    recs[v].append(r)
                    fj.write(json.dumps(r) + "\n")
            fj.close()
            for v in variants:
                R = recs[v]
                med = [r["med_err_inb"] for r in R if "med_err_inb" in r]
                summary[f"{mname}|{v}|{d}"] = dict(
                    n=len(R), median_err_inb=float(np.median(med)) if med else None,
                    frac2=float(np.mean([r.get("frac2", 0) for r in R])),
                    success_v2=float(np.mean([r["gate_v2"] for r in R]) * 100),
                    **pose_auc([r.get("pose_err", 180.0) for r in R]))
                print(mname, v, d, summary[f"{mname}|{v}|{d}"], flush=True)
            json.dump({k: summary[k] for k in summary if k.startswith(f"{mname}|") and k.endswith(f"|{d}")},
                      open(done_f, "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    old = json.load(open(f"{OUT}/summary.json")) if os.path.exists(f"{OUT}/summary.json") else {}
    old.update(summary)
    json.dump(old, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["control", "run"])
    ap.add_argument("--d", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--datasets", default="1,2,3")
    ap.add_argument("--models", default="baseline_p32,p16_nofpn")
    ap.add_argument("--variants", default="none,A,B,C")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "control":
        control()
    else:
        run([int(x) for x in a.datasets.split(",")], [int(x) for x in a.d.split(",")],
            a.models.split(","), a.variants.split(","), a.limit)
