"""Revisi A IJIES -- ringkasan X3/X5: validasi gerbang terhadap GT dan kurva registrasi benar-GT (UseGeo).

Benar-GT (utama, prareg) = galat kisi-H <= 99,45 px (5% lebar: tumpang tindih yang benar ditemukan); sekunder
<= 39,78 px (2% lebar). Galat registrasi yang benar tetapi kasar berada di 13-67 px (kuantisasi sel + galat model
planar yang naik dengan baseline: median oracle 4,6 px di d1, 23 px di d6), kegagalan sungguhan > 100-300 px.
Registrasi sukses (kriteria bersama) = gate_v2 lolos DAN benar-GT.
FAR gerbang = P(tidak benar | gerbang lolos); FRR = P(gerbang tolak | benar).
Per label × d (semua pasangan, sepanjang jalur dan belokan dipisah) dan per bin tampalan terukur, IK 95% bootstrap klaster
segmen jalur (berstrata penerbangan, 2000x).
Pakai: python revisi_d/scripts/arev_x3_summary.py [label ...]   (bawaan: semua label yang lengkap d1..d8)
Keluaran: revisi_d/results/arev/x3/summary.json
"""
import glob
import json
import os
import sys

import numpy as np
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
X3 = f"{RD}/results/arev/x3"
TAU = float(os.environ.get("AREV_TAU", 99.45))
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def seg_map():
    sys.path.insert(0, RD)
    from crefine import data as D
    seg = {}
    for ds in (1, 2, 3):
        s, _ = D.segments(ds)
        seg.update({(ds, k): v for k, v in s.items()})
    return seg


def boot_rate(x, cl, n=2000, seed=0):
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    st = cl // 1000
    groups = {s: [np.flatnonzero(cl == c) for c in np.unique(cl[st == s])] for s in np.unique(st)}
    vals = []
    for _ in range(n):
        ix = np.concatenate([gs[j] for gs in groups.values() for j in rng.integers(0, len(gs), len(gs))])
        vals.append(x[ix].mean())
    return [float(np.percentile(vals, 2.5) * 100), float(np.percentile(vals, 97.5) * 100)]


def main():
    labels = sys.argv[1:] or sorted({os.path.basename(f).split("__")[1] for f in glob.glob(f"{X3}/usegeo__*__d8.done")})
    seg = seg_map()
    out = {}
    for lab in labels:
        R = []
        for d in range(1, 9):
            f = f"{X3}/usegeo__{lab}__d{d}.jsonl"
            if os.path.exists(f):
                R += [json.loads(l) for l in open(f)]
        if not R:
            continue
        res = {}
        for d in range(1, 9):
            rd = [r for r in R if r["d"] == d]
            for part, sel in (("all", rd), ("along", [r for r in rd if r.get("along")]),
                              ("turn", [r for r in rd if not r.get("along")])):
                if not sel:
                    continue
                acc = np.array([bool(r["gate_v2"]) for r in sel])
                cor = np.array([r["hgrid"] <= TAU for r in sel])
                suc = acc & cor
                cl = np.array([seg[(r["dataset"], r["pair"][0].replace("_res.jpg", ".jpg"))] for r in sel])
                res[f"d{d}|{part}"] = dict(
                    n=len(sel), gate=float(acc.mean() * 100), correct=float(cor.mean() * 100),
                    success=float(suc.mean() * 100), success_ci95=boot_rate(suc.astype(float), cl),
                    far=float((acc & ~cor).sum() / max(acc.sum(), 1) * 100),
                    frr=float((~acc & cor).sum() / max(cor.sum(), 1) * 100),
                    n_false_accept=int((acc & ~cor).sum()), n_false_reject=int((~acc & cor).sum()),
                    hgrid_med_success=float(np.median([r["hgrid"] for r, s in zip(sel, suc) if s])) if suc.any() else None,
                    oracle_med=float(np.nanmedian([r["oracle"] if r["oracle"] is not None else np.nan for r in sel])))
        ov = [(r["overlap"], bool(r["gate_v2"]) and r["hgrid"] <= TAU, r) for r in R if r.get("overlap") is not None]
        for lo, hi in BINS:
            sel = [x for x in ov if lo <= x[0] < hi]
            if not sel:
                continue
            suc = np.array([x[1] for x in sel], float)
            cl = np.array([seg[(x[2]["dataset"], x[2]["pair"][0].replace("_res.jpg", ".jpg"))] for x in sel])
            res[f"overlap[{lo:.1f},{hi:.1f})"] = dict(n=len(sel), success=float(suc.mean() * 100),
                                                      success_ci95=boot_rate(suc, cl))
        out[lab] = res
        line = " ".join(f"d{d}:{res[f'd{d}|all']['success']:5.1f}(FA{res[f'd{d}|all']['n_false_accept']})"
                        for d in range(1, 9) if f"d{d}|all" in res)
        print(f"{lab:22s} {line}", flush=True)
    os.makedirs(X3, exist_ok=True)
    json.dump(out, open(f"{X3}/summary.json.tmp", "w"), indent=1)
    os.replace(f"{X3}/summary.json.tmp", f"{X3}/summary.json")


if __name__ == "__main__":
    main()
