"""Revisi A IJIES -- statistik model berbenih dan aturan keputusan prareg D1-D3, D5 (prereg klarifikasi 1).

Data per run (keluaran arev_eval.py dan arev_x3_gate.py, tanpa perubahan parameter):
  akurasi    results/arev/eval/<run>/acc_d{1,2}.jsonl   UseGeo d = 1-2 sepanjang jalur, varian 'B' (acuan) dan 'none'
  ketahanan  results/arev/x3/usegeo__<run>_{final,coarse}__d{1..8}.jsonl   sukses = gate_v2 dan galat kisi-H <= 99,45 px
Metrik varian = rerata atas benih dari metrik per run. AUC@theta atas semua pasangan (gagal = 180 derajat); galat median
pada subset pasangan yang galatnya terdefinisi di semua run yang dibandingkan; sukses pada tampalan terukur [0,2; 0,6).
Bootstrap hierarkis 10.000x (benih 20261101): benih diresampel dengan pengembalian di dalam tiap varian, lalu segmen
jalur diresampel berstrata penerbangan; resampel segmen yang sama dipakai untuk kedua varian (pasangan sama).
Keluaran: revisi_d/results/arev/seed/stats.json (+ ringkasan cetak). Pakai: python revisi_d/scripts/arev_seed_stats.py
"""
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
EV, X3, OUT = f"{RD}/results/arev/eval", f"{RD}/results/arev/x3", f"{RD}/results/arev/seed"
TH = (1, 2, 5, 10, 20)
TAU = 99.45
BIN = (0.2, 0.6)
N_BOOT = int(os.environ.get("AREV_NBOOT", 10000))
GROUPS = {"d8": ["orig_d8", "d8_s2", "d8_s3"], "d18": ["orig_d18", "d18_s2", "d18_s3"],
          "L0": ["L0_s1", "L0_s2"], "L4": ["L4_s1", "L4_s2"], "clean": ["clean_s1", "clean_s2"],
          "s32": ["s32_s1", "s32_s2"], "r480": ["r480_s1", "r480_s2"], "pub": ["pub_s1"]}
COMPARE = [("D1", "d8", "d18"), ("D2", "L0", "d8"), ("D2_L4", "L4", "d8"), ("D3_s32", "s32", "d8"),
           ("D3_r480", "r480", "d8"), ("D3_clean", "clean", "d8"), ("D5", "pub", "clean")]


def pose_auc(errs):
    from e13_refine import pose_auc as pa
    a = pa(list(errs), ths=TH)
    return np.array([a[f"auc@{t}"] for t in TH])


def load_acc(run, variant):
    out = {}
    for d in (1, 2):
        f = f"{EV}/{run}/acc_d{d}.jsonl"
        if not os.path.exists(f"{EV}/{run}/acc_d{d}.done"):
            return None
        for line in open(f):
            r = json.loads(line)
            if r["variant"] != variant:
                continue
            pe, me = r.get("pose_err"), r.get("med_err_inb")
            lk = (1 - r["lk_ok"] / r["n_coarse"]) if (r.get("lk_ok") is not None and r.get("n_coarse")) else np.nan
            out[(r["dataset"], r["frame_a"], d)] = (180.0 if pe is None else pe, np.nan if me is None else me, lk,
                                                    bool(r.get("h0_fail")))
    return out


def load_rob(run, kind):
    lab = f"{run}_{kind}"
    if not os.path.exists(f"{X3}/usegeo__{lab}__d8.done"):
        return None
    out = {}
    for d in range(1, 9):
        for line in open(f"{X3}/usegeo__{lab}__d{d}.jsonl"):
            r = json.loads(line)
            hg = r.get("hgrid")
            ok = bool(r.get("gate_v2")) and hg is not None and np.isfinite(hg) and hg <= TAU
            out[(r["dataset"], r["pair"][0].replace("_res.jpg", ".jpg"), d)] = (r.get("overlap"), ok)
    return out


def seg_map():
    from crefine import data as D
    seg = {}
    for ds in (1, 2, 3):
        s, _ = D.segments(ds)
        seg.update({(ds, k): v for k, v in s.items()})
    return seg


def pct(x, lv):
    a = (100 - lv) / 2
    return np.percentile(x, [a, 100 - a]).tolist()


def verdict_equiv(lo90, hi90, lo95, hi95, margin, loss_sign):
    """D1: TOST dengan margin simetris; loss_sign = -1 untuk AUC (turun = rugi), +1 untuk galat (naik = rugi)."""
    if -margin < lo90 and hi90 < margin:
        return "equivalent within the margin"
    if lo95 <= 0 <= hi95:
        return "no detectable loss in this experiment"
    worse = (hi95 < 0) if loss_sign < 0 else (lo95 > 0)
    return "loss reported" if worse else "favourable difference reported"


def main():
    os.makedirs(OUT, exist_ok=True)
    seg = seg_map()
    res = dict(n_boot=N_BOOT, tau_px=TAU, overlap_bin=list(BIN), th=list(TH), runs={}, groups={}, comparisons={})
    acc, accn, rob, robc = {}, {}, {}, {}
    for g, runs in GROUPS.items():
        for r in runs:
            a, an, b, bc = load_acc(r, "B"), load_acc(r, "none"), load_rob(r, "final"), load_rob(r, "coarse")
            if a is None:
                continue
            acc[r], accn[r] = a, an
            if b is not None:
                rob[r] = b
            if bc is not None:
                robc[r] = bc
    avail = {g: [r for r in runs if r in acc] for g, runs in GROUPS.items()}
    keys = sorted(set.intersection(*[set(v) for v in acc.values()])) if acc else []
    cl_acc = np.array([seg[(k[0], k[1])] for k in keys])
    sel = {}
    for r in acc:
        sp = f"{EV}/{r}/selected.json"
        sel[r] = json.load(open(sp))["best"] if os.path.exists(sp) else None
    # metrik per run (titik)
    E = {r: np.array([acc[r][k][0] for k in keys]) for r in acc}
    M = {r: np.array([acc[r][k][1] for k in keys]) for r in acc}
    En = {r: np.array([accn[r][k][0] for k in keys]) for r in acc}
    Mn = {r: np.array([accn[r][k][1] for k in keys]) for r in acc}
    rkeys = sorted(set.intersection(*[set(v) for v in rob.values()])) if rob else []
    rin = [k for k in rkeys if rob[next(iter(rob))][k][0] is not None and BIN[0] <= rob[next(iter(rob))][k][0] < BIN[1]]
    cl_rob = np.array([seg[(k[0], k[1])] for k in rin])
    S = {r: np.array([rob[r][k][1] for k in rin], float) for r in rob}
    Sc = {r: np.array([robc[r][k][1] for k in rin], float) for r in robc}
    for r in acc:
        okr = ~np.isnan(M[r])
        res["runs"][r] = dict(selected=sel[r], n_pairs=len(keys), auc_B=pose_auc(E[r]).tolist(), auc_none=pose_auc(En[r]).tolist(),
                              med_B=float(np.median(M[r][okr])), med_none=float(np.nanmedian(Mn[r])),
                              n_err_undefined_B=int((~okr).sum()), n_pose_fail_B=int((E[r] >= 180).sum()),
                              lk_fail_median=float(np.nanmedian([acc[r][k][2] for k in keys])),
                              lk_fail_p90=float(np.nanpercentile([acc[r][k][2] for k in keys], 90)),
                              h0_fail=int(sum(acc[r][k][3] for k in keys)),
                              success_final=float(S[r].mean() * 100) if r in S else None,
                              success_coarse=float(Sc[r].mean() * 100) if r in Sc else None, n_rob_pairs=len(rin))
    # gugus segmen berstrata penerbangan; indeks pasangan akurasi dan ketahanan per segmen
    strata = {}
    for c in np.unique(np.concatenate([cl_acc, cl_rob]) if len(cl_rob) else cl_acc):
        strata.setdefault(int(c) // 1000, []).append(int(c))
    ia = {c: np.flatnonzero(cl_acc == c) for cs in strata.values() for c in cs}
    ir = {c: np.flatnonzero(cl_rob == c) for cs in strata.values() for c in cs}

    def group_point(g):
        rs = avail[g]
        return rs

    for name, ga, gb in COMPARE:
        ra, rb = avail[ga], avail[gb]
        if not ra or not rb:
            continue
        sub = np.ones(len(keys), bool)
        for r in ra + rb:
            sub &= ~np.isnan(M[r])
        has_rob = all(r in S for r in ra + rb)

        def metr(rs, ix, jx, seeds):
            auc = np.mean([pose_auc(E[r][ix]) for r in seeds], axis=0)
            ii = ix[sub[ix]]
            med = np.mean([np.median(M[r][ii]) for r in seeds])
            aucn = np.mean([pose_auc(En[r][ix]) for r in seeds], axis=0)
            medn = np.mean([np.nanmedian(Mn[r][ii]) for r in seeds])
            suc = np.mean([S[r][jx].mean() * 100 for r in seeds]) if has_rob and len(jx) else np.nan
            sucn = np.mean([Sc[r][jx].mean() * 100 for r in seeds]) if has_rob and all(r in Sc for r in seeds) and len(jx) else np.nan
            return np.concatenate([auc, [med], aucn, [medn, suc, sucn]])

        ix0, jx0 = np.arange(len(keys)), np.arange(len(rin))
        pa, pb = metr(ra, ix0, jx0, ra), metr(rb, ix0, jx0, rb)
        rng = np.random.default_rng(20261101)
        B = []
        for _ in range(N_BOOT):
            sa = [ra[i] for i in rng.integers(0, len(ra), len(ra))]
            sb = [rb[i] for i in rng.integers(0, len(rb), len(rb))]
            cs = [cs_[j] for cs_ in strata.values() for j in rng.integers(0, len(cs_), len(cs_))]
            ix = np.concatenate([ia[c] for c in cs])
            jx = np.concatenate([ir[c] for c in cs]) if len(rin) else np.array([], int)
            B.append(metr(ra, ix, jx, sa) - metr(rb, ix, jx, sb))
        B = np.array(B)
        nT = len(TH)
        lab = [f"auc@{t}" for t in TH] + ["med_err"] + [f"none_auc@{t}" for t in TH] + ["none_med_err", "success", "success_coarse"]
        dpt = pa - pb
        cmp = dict(a=ga, b=gb, runs_a=ra, runs_b=rb, n_subset=int(sub.sum()), n_rob=len(rin),
                   complete=len(ra) == len(GROUPS[ga]) and len(rb) == len(GROUPS[gb]),
                   point_a=dict(zip(lab, pa.tolist())), point_b=dict(zip(lab, pb.tolist())),
                   delta=dict(zip(lab, dpt.tolist())),
                   ci90={l: pct(B[:, i], 90) for i, l in enumerate(lab)}, ci95={l: pct(B[:, i], 95) for i, l in enumerate(lab)},
                   sd_seed_a={l: float(np.std([metr([r], ix0, jx0, [r])[i] for r in ra], ddof=1)) if len(ra) > 1 else None
                              for i, l in enumerate(lab) if l in ("auc@5", "med_err", "success")},
                   sd_seed_b={l: float(np.std([metr([r], ix0, jx0, [r])[i] for r in rb], ddof=1)) if len(rb) > 1 else None
                              for i, l in enumerate(lab) if l in ("auc@5", "med_err", "success")})
        i5, ie, isu = TH.index(5), nT, 2 * nT + 2
        c90, c95 = cmp["ci90"], cmp["ci95"]
        if name == "D1":
            m_auc = 0.01 * pb[i5]
            cmp["margin_auc5"], cmp["margin_err"] = m_auc, 0.10
            v_auc = verdict_equiv(*c90["auc@5"], *c95["auc@5"], m_auc, -1)
            v_err = verdict_equiv(*c90["med_err"], *c95["med_err"], 0.10, +1)
            order = ["loss reported", "no detectable loss in this experiment", "favourable difference reported",
                     "equivalent within the margin"]
            cmp["verdict"] = dict(auc5=v_auc, med_err=v_err, combined=min((v_auc, v_err), key=order.index))
        elif name == "D2":
            ni = c90["auc@5"][0] > -0.98 and c90["med_err"][1] < 0.10 and c90["success"][0] > -5
            role = c90["success"][1] < -5
            cmp["verdict"] = ("non-inferior: body-necessity claim withdrawn, title and contributions revised" if ni else
                              f"body role stated as the measured difference ({dpt[isu]:.1f} points)" if role else
                              "inconclusive: numbers reported")
        elif name == "D5":
            ok = c90["auc@5"][0] > -0.98 and c90["med_err"][1] < 0.10
            cmp["verdict"] = "replicates within the margin" if ok else "difference reported as the replication bound"
        else:
            cmp["verdict"] = {l: ("CI95 excludes 0" if (c95[l][0] > 0 or c95[l][1] < 0) else "CI95 contains 0")
                              for l in ("auc@5", "med_err", "success", "none_auc@5", "none_med_err")}
        res["comparisons"][name] = cmp
    for g, rs in avail.items():
        if rs:
            res["groups"][g] = rs
    json.dump(res, open(f"{OUT}/stats.json.tmp", "w"), indent=1, default=float)
    os.replace(f"{OUT}/stats.json.tmp", f"{OUT}/stats.json")
    for r, v in res["runs"].items():
        print(f"{r:10s} ep {v['selected']}  AUC@5 B {v['auc_B'][2]:.2f} none {v['auc_none'][2]:.2f}  med B {v['med_B']:.3f} "
              f"none {v['med_none']:.2f}  LKgagal {100 * v['lk_fail_median']:.1f}%  sukses[0.2,0.6) akhir "
              f"{v['success_final'] if v['success_final'] is None else round(v['success_final'], 1)} kasar "
              f"{v['success_coarse'] if v['success_coarse'] is None else round(v['success_coarse'], 1)}")
    for name, c in res["comparisons"].items():
        d, c9 = c["delta"], c["ci90"]
        print(f"{name} {c['a']}-{c['b']}: dAUC5 {d['auc@5']:+.3f} CI90 [{c9['auc@5'][0]:+.3f},{c9['auc@5'][1]:+.3f}] | "
              f"dErr {d['med_err']:+.3f} CI90 [{c9['med_err'][0]:+.3f},{c9['med_err'][1]:+.3f}] | dSukses {d['success']:+.1f} "
              f"CI90 [{c9['success'][0]:+.1f},{c9['success'][1]:+.1f}] | {c['verdict']}")


if __name__ == "__main__":
    main()
