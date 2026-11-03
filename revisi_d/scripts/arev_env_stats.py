"""Revisi A IJIES -- batas operasi (R2.6), validasi gerbang (R1.9), tekstur rendah: ringkasan untuk naskah §5.5.

UseGeo d = 1-8, semua pasangan (sepanjang jalur + belokan), catatan X3 (arev_x3_gate.py): sukses = gate_v2 dan galat
kisi-H <= 99,45 px (5% lebar). Per label: sukses per bin tampalan terukur dengan IK 95% bootstrap klaster (segmen
jalur berstrata penerbangan, 2000x), penerimaan palsu gerbang (lolos gerbang tetapi salah) dan penolakan palsu
(ditolak gerbang padahal H benar) sebagai persen pasangan, pasangan belokan terpisah, dan "lantai tampalan" = batas
bawah bin terendah yang suksesnya >= 50%. Tekstur rendah (Seneca, Bellus d = 1-3): laju lolos gerbang dari catatan
registrasi tanpa batas kecocokan (G0, evaluasi usulan, SIFT P30). Kebijakan gerbang (gate_policy): untuk tiap pasangan
label (keluaran kasar, keluaran akhir dengan penghalus), sukses dan penerimaan palsu bila gerbang dinilai pada keluaran
akhir (seperti evaluasi) atau pada keluaran kasar sebelum penghalus, kebenaran selalu dari H akhir.
Keluaran: revisi_d/results/arev/x3/env_stats.json. Pakai: python revisi_d/scripts/arev_env_stats.py
"""
import glob
import json
import os
import sys

import numpy as np
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

RD = AERIAL_ROOT + "/revisi_d"
sys.path.insert(0, f"{RD}/scripts")
sys.path.insert(0, RD)
A = f"{RD}/results/arev"
TAU = 99.45
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
LOWTEX = {"proposed_coarse": f"{A}/eval/orig_d8/reg/{{s}}__orig_d8_trained__d{{d}}.jsonl",
          "sift": f"{RD}/results/p30/{{s}}__sift__d{{d}}.jsonl"}
for _m in ("matchformer", "lgfctr", "aspanformer", "loftr", "efficientloftr", "topicfm_fast", "jamma", "edm",
           "disk_lightglue", "swinmatcher_tgrs"):
    LOWTEX[_m] = f"{A}/g0/{{s}}__{_m}__d{{d}}.jsonl"


def seg_map():
    from crefine import data as D
    seg = {}
    for ds in (1, 2, 3):
        s, _ = D.segments(ds)
        seg.update({(ds, k): v for k, v in s.items()})
    return seg


def boot(x, cl, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    strata = cl // 1000
    groups = {s: [np.flatnonzero(cl == c) for c in np.unique(cl[strata == s])] for s in np.unique(strata)}
    out = []
    for _ in range(n):
        ix = np.concatenate([gs[j] for gs in groups.values() for j in rng.integers(0, len(gs), len(gs))])
        out.append(x[ix].mean() * 100)
    return np.percentile(out, [2.5, 97.5]).tolist()


def main():
    seg = seg_map()
    res = dict(tau_px=TAU, bins=BINS, labels={}, lowtex={})
    labels = sorted({os.path.basename(f).split("__")[1] for f in glob.glob(f"{A}/x3/usegeo__*__d8.done")})
    for lab in labels:
        R = []
        for d in range(1, 9):
            for line in open(f"{A}/x3/usegeo__{lab}__d{d}.jsonl"):
                R.append(json.loads(line))
        hg = np.array([np.inf if (r.get("hgrid") is None or not np.isfinite(r["hgrid"])) else r["hgrid"] for r in R])
        gate = np.array([bool(r.get("gate_v2")) for r in R])
        cor = hg <= TAU
        suc = gate & cor
        ov = np.array([np.nan if r.get("overlap") is None else r["overlap"] for r in R])
        along = np.array([bool(r.get("along")) for r in R])
        cl = np.array([seg[(r["dataset"], r["pair"][0].replace("_res.jpg", ".jpg"))] for r in R])
        L = dict(n=len(R), success=float(suc.mean() * 100), false_accept=float((gate & ~cor).mean() * 100),
                 false_reject=float((~gate & cor).mean() * 100),
                 false_accept_of_accepted=float((gate & ~cor).sum() / max(gate.sum(), 1) * 100),
                 turn=dict(n=int((~along).sum()), success=float(suc[~along].mean() * 100) if (~along).any() else None),
                 bins={})
        L["bins_along"] = {}
        for key, sel0 in (("bins", np.ones(len(R), bool)), ("bins_along", along)):
            for lo, hi in BINS:
                s = (ov >= lo) & (ov < hi) & sel0
                if s.sum():
                    L[key][f"{lo:.1f}-{min(hi, 1.0):.1f}"] = dict(n=int(s.sum()), success=float(suc[s].mean() * 100),
                                                                  ci95=boot(suc[s].astype(float), cl[s]),
                                                                  false_accept=float((gate & ~cor)[s].mean() * 100))
        floor = None
        for lo, hi in BINS:
            b = L["bins_along"].get(f"{lo:.1f}-{min(hi, 1.0):.1f}")
            if b and b["success"] >= 50:
                floor = lo
                break
        L["overlap_floor"] = floor
        res["labels"][lab] = L
    # gerbang sebelum penghalus: terima bila gerbang lolos pada keluaran KASAR, nilai kebenaran pada H AKHIR
    pairs = [(lab[:-len("_final")] + "_coarse", lab) for lab in labels if lab.endswith("_final")]
    pairs += [(lab[:-len("+B")], lab) for lab in labels if lab.endswith("+B")]
    res["gate_policy"] = {}
    for lc, lf in pairs:
        if lc not in labels:
            continue
        Rc, Rf = {}, {}
        for d in range(1, 9):
            for lab_, RR in ((lc, Rc), (lf, Rf)):
                for line in open(f"{A}/x3/usegeo__{lab_}__d{d}.jsonl"):
                    r = json.loads(line)
                    RR[(r["dataset"], r["pair"][0], d)] = r
        ks = sorted(set(Rc) & set(Rf))
        okf = np.array([Rf[k].get("hgrid") is not None and np.isfinite(Rf[k]["hgrid"]) and Rf[k]["hgrid"] <= TAU
                        for k in ks])
        gc = np.array([bool(Rc[k].get("gate_v2")) for k in ks])
        gf = np.array([bool(Rf[k].get("gate_v2")) for k in ks])
        ov = np.array([np.nan if Rc[k].get("overlap") is None else Rc[k]["overlap"] for k in ks])
        s24 = (ov >= 0.2) & (ov < 0.4) & np.array([bool(Rc[k].get("along")) for k in ks])
        res["gate_policy"][lf] = dict(
            n=len(ks), coarse_label=lc,
            gate_final=dict(success=float((gf & okf).mean() * 100), false_accept=float((gf & ~okf).mean() * 100),
                            success_24_along=float((gf & okf)[s24].mean() * 100)),
            gate_coarse=dict(success=float((gc & okf).mean() * 100), false_accept=float((gc & ~okf).mean() * 100),
                             success_24_along=float((gc & okf)[s24].mean() * 100)))
    for lab, fmt in LOWTEX.items():
        T = {}
        for s in ("seneca", "bellus"):
            for d in (1, 2, 3):
                f = fmt.format(s=s, d=d)
                if os.path.exists(f):
                    rs = [json.loads(line) for line in open(f)]
                    T.setdefault(s, {})[str(d)] = float(np.mean([bool(r.get("gate_v2")) for r in rs]) * 100)
        if T:
            res["lowtex"][lab] = T
    os.makedirs(f"{A}/x3", exist_ok=True)
    json.dump(res, open(f"{A}/x3/env_stats.json.tmp", "w"), indent=1)
    os.replace(f"{A}/x3/env_stats.json.tmp", f"{A}/x3/env_stats.json")
    for lab, L in res["labels"].items():
        b = L["bins_along"]
        print(f"{lab:24s} FA {L['false_accept']:4.1f}  belok {L['turn']['success'] or 0:5.1f} (n {L['turn']['n']})  "
              f"lantai {L['overlap_floor']}  jalur: " + " ".join(f"{k}:{v['success']:.0f}" for k, v in b.items()))
    for lab, T in res["lowtex"].items():
        print("lowtex", lab, T)


if __name__ == "__main__":
    main()
