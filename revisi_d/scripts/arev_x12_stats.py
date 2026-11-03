"""Revisi A IJIES -- X1 (estimand Tabel 2 eksplisit) + X2 (bootstrap klaster segmen jalur, per penerbangan).

Estimand (dinyatakan di naskah §4.3):
  galat lokalisasi per pasangan  = median atas titik cocokan yang ber-GT LiDAR sah dan proyeksinya jatuh di citra B
  ringkasan per metode           = median atas pasangan yang galatnya terdefinisi
  Δ_sub   = median_A(subset) − median_B(subset), subset = pasangan yang galatnya terdefinisi di KEDUA metode
            (estimand Tabel 2 versi terkirim; untuk SwinMatcher subset = 797 dari 1418 pasangan)
  Δ_pair  = median atas pasangan subset dari (galat_A − galat_B)   (estimand berpasangan pendamping)
  AUC@θ   = luas di bawah CDF galat pose sampai θ (pasangan gagal = 180°), atas SEMUA pasangan
Ketidakpastian: (i) bootstrap pasangan iid 5000x benih 0 (kontrol: mengulang p24A persis), (ii) bootstrap klaster:
segmen jalur `crefine/data.py:segments` diresampel di dalam tiap penerbangan (strata), 10.000x, benih 20261101.

Pakai: python revisi_d/scripts/arev_x12_stats.py [--reg-dir revisi_d/results/eval/registrasi] [--tag capped]
Keluaran: revisi_d/results/arev/x12/<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
sys.path.insert(0, f"{RD}/scripts")
sys.path.insert(0, RD)
TH = (1, 2, 5, 10, 20)
BASELINES = ("matchformer", "lgfctr", "swinmatcher_tgrs", "aspanformer", "loftr",
             "efficientloftr", "topicfm_fast", "jamma", "disk_lightglue", "edm")


def pose_auc(errs):
    from e13_refine import pose_auc as pa
    a = pa(list(errs), ths=TH)
    return np.array([a[f"auc@{t}"] for t in TH])


def load_e13(path_fmt, variant):
    out = {}
    for d in (1, 2):
        for line in open(path_fmt.format(d=d)):
            r = json.loads(line)
            if r["variant"] == variant:
                pe = r.get("pose_err")
                out[(r["dataset"], r["frame_a"], d)] = (180.0 if pe is None else pe, r.get("med_err_inb"))
    return out


def load_reg(reg_dir, m):
    out = {}
    for d in (1, 2):
        f = f"{reg_dir}/usegeo__{m}__d{d}.jsonl"
        if not os.path.exists(f):
            return None
        for line in open(f):
            r = json.loads(line)
            if not r.get("along"):
                continue
            pe = r.get("pose_err")
            out[(r["dataset"], r["pair"][0].replace("_res.jpg", ".jpg"), d)] = (180.0 if pe is None else pe,
                                                                                r.get("med_err_inbounds"))
    return out


def seg_map():
    from crefine import data as D
    seg = {}
    for ds in (1, 2, 3):
        s, _ = D.segments(ds)
        seg.update({(ds, k): v for k, v in s.items()})
    return seg


def compare(A, B, seg, n_iid=5000, n_cl=10000):
    keys = sorted(set(A) & set(B))
    ea = np.array([A[k][0] for k in keys]); eb = np.array([B[k][0] for k in keys])
    ma = np.array([np.nan if A[k][1] is None else A[k][1] for k in keys])
    mb = np.array([np.nan if B[k][1] is None else B[k][1] for k in keys])
    ok = ~np.isnan(ma) & ~np.isnan(mb)
    io = np.flatnonzero(ok)
    cl = np.array([seg[(k[0], k[1])] for k in keys])
    res = dict(n_pairs=len(keys), n_subset=int(ok.sum()),
               n_err_undefined_a=int(np.isnan(ma).sum()), n_err_undefined_b=int(np.isnan(mb).sum()),
               n_pose_fail_a=int((ea >= 180).sum()), n_pose_fail_b=int((eb >= 180).sum()),
               med_a_all=float(np.nanmedian(ma)), med_b_all=float(np.nanmedian(mb)),
               med_a_sub=float(np.median(ma[ok])), med_b_sub=float(np.median(mb[ok])),
               d_sub=float(np.median(ma[ok]) - np.median(mb[ok])), d_pair=float(np.median(ma[ok] - mb[ok])),
               th=list(TH), auc_a=pose_auc(ea).tolist(), auc_b=pose_auc(eb).tolist())
    res["d_auc"] = (np.array(res["auc_a"]) - np.array(res["auc_b"])).tolist()
    # (i) iid -- persis pola p24_A_stats.compare (kontrol)
    rng = np.random.default_rng(0)
    bs, bm = [], []
    for _ in range(n_iid):
        ii = rng.integers(0, len(keys), len(keys))
        bs.append(pose_auc(ea[ii]) - pose_auc(eb[ii]))
        jj = rng.choice(io, len(io))
        bm.append(np.median(ma[jj]) - np.median(mb[jj]))
    bs, bm = np.array(bs), np.array(bm)
    res["iid"] = dict(d_auc_ci95=[np.percentile(bs[:, j], [2.5, 97.5]).tolist() for j in range(len(TH))],
                      d_sub_ci95=np.percentile(bm, [2.5, 97.5]).tolist())
    # (ii) klaster segmen, berstrata penerbangan
    rng = np.random.default_rng(20261101)
    strata = cl // 1000
    groups = {s: [np.flatnonzero(cl == c) for c in np.unique(cl[strata == s])] for s in np.unique(strata)}
    da, ds_, dp = [], [], []
    for _ in range(n_cl):
        ix = np.concatenate([gs[j] for gs in groups.values() for j in rng.integers(0, len(gs), len(gs))])
        da.append(pose_auc(ea[ix]) - pose_auc(eb[ix]))
        sub = ix[ok[ix]]
        ds_.append(np.median(ma[sub]) - np.median(mb[sub]))
        dp.append(np.median(ma[sub] - mb[sub]))
    da = np.array(da)
    res["cluster"] = dict(n_clusters=int(sum(len(g) for g in groups.values())),
                          d_auc_ci95=[np.percentile(da[:, j], [2.5, 97.5]).tolist() for j in range(len(TH))],
                          d_sub_ci95=np.percentile(ds_, [2.5, 97.5]).tolist(),
                          d_pair_ci95=np.percentile(dp, [2.5, 97.5]).tolist())
    # per penerbangan
    fl = {}
    for f in (1, 2, 3):
        sel = np.array([k[0] == f for k in keys])
        okf = ok & sel
        fl[str(f)] = dict(n=int(sel.sum()), auc5_a=float(pose_auc(ea[sel])[2]), auc5_b=float(pose_auc(eb[sel])[2]),
                          med_a=float(np.nanmedian(ma[sel])), med_b=float(np.nanmedian(mb[sel])),
                          med_a_sub=float(np.median(ma[okf])) if okf.any() else None,
                          med_b_sub=float(np.median(mb[okf])) if okf.any() else None,
                          fail_a=int((ea[sel] >= 180).sum()), fail_b=int((eb[sel] >= 180).sum()))
    res["per_flight"] = fl
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-dir", default=f"{RD}/results/eval/registrasi")
    ap.add_argument("--tag", default="capped_oct")
    ap.add_argument("--n-cl", type=int, default=10000)
    a = ap.parse_args()
    os.chdir(ROOT)
    seg = seg_map()
    U = load_e13(f"{RD}/results/e13/p16_nofpn__d{{d}}.jsonl", "B")
    out = {}
    for m in BASELINES:
        Bm = load_reg(a.reg_dir, m)
        if Bm is None:
            print("lewat (belum ada):", m, flush=True)
            continue
        out[f"proposed|{m}"] = r = compare(U, Bm, seg, n_cl=a.n_cl)
        print(f"{m:18s} n={r['n_pairs']} sub={r['n_subset']} medA {r['med_a_all']:.3f}/{r['med_a_sub']:.3f} "
              f"medB {r['med_b_all']:.3f}/{r['med_b_sub']:.3f} d_sub {r['d_sub']:+.3f} iid {np.round(r['iid']['d_sub_ci95'], 3)} "
              f"klaster {np.round(r['cluster']['d_sub_ci95'], 3)} | dAUC5 {r['d_auc'][2]:+.2f} iid "
              f"{np.round(r['iid']['d_auc_ci95'][2], 2)} klaster {np.round(r['cluster']['d_auc_ci95'][2], 2)}", flush=True)
    os.makedirs(f"{RD}/results/arev/x12", exist_ok=True)
    tmp = f"{RD}/results/arev/x12/{a.tag}.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{RD}/results/arev/x12/{a.tag}.json")


if __name__ == "__main__":
    main()
