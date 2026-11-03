"""Revisi A IJIES -- skenario terkendali (R1.1, R2.1, prareg D4): ringkasan akurasi UseGeo d = 1-2 untuk
  keluaran lengkap pembanding (tahap halus sendiri)           x4/<m>            'none'
  keluaran lengkap + penghalus CPU identik                    x4/<m>            'B'
  jalur kasar terkompilasi + penghalus identik                x4/<m>_coarse     'none' / 'B'
  pembanding latih ulang udara: lengkap / lengkap + penghalus x4/<m>_ft
  jalur kasar PyTorch, bobot resmi dan latih ulang            eval/tp_<m>_{official,ft_s1}/coarse/*.acc.jsonl
  usulan (checkpoint versi terkirim)                           eval/orig_d8/acc_d*.jsonl
Setiap label: AUC@{1,2,5,10,20} (gagal = 180 derajat) dan galat median per metode; selisih terhadap usulan 'B' dan
selisih D4 (latih ulang - resmi) dengan bootstrap klaster segmen berstrata penerbangan (arev_x12_stats.compare,
10.000x). Aturan D4: "pelatihan udara menjelaskan sebagian selisih" hanya bila latih ulang memperbaiki AUC@5 atau galat
dengan IK 95% yang tidak memuat 0.
Keluaran: revisi_d/results/arev/ctrl/stats.json. Pakai: python revisi_d/scripts/arev_ctrl_stats.py
"""
import json
import os
import sys

import numpy as np
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

RD = AERIAL_ROOT + "/revisi_d"
sys.path.insert(0, f"{RD}/scripts")
sys.path.insert(0, RD)
import arev_x12_stats as X   # noqa: E402

A = f"{RD}/results/arev"
G0 = ("matchformer", "lgfctr", "swinmatcher_tgrs", "aspanformer", "loftr", "efficientloftr", "topicfm_fast", "jamma",
      "disk_lightglue", "edm")
G4 = ("matchformer", "lgfctr", "aspanformer", "loftr", "efficientloftr", "topicfm_fast", "jamma", "usulan")


def load(fmt, variant):
    out = {}
    for d in (1, 2):
        f = fmt.format(d=d)
        done = f.replace(".acc.jsonl", ".done") if f.endswith(".acc.jsonl") else f.replace(".jsonl", ".done")
        if not os.path.exists(f) or not os.path.exists(done):
            return None
        for line in open(f):
            r = json.loads(line)
            if r.get("variant") != variant:
                continue
            pe, me = r.get("pose_err"), r.get("med_err_inb")
            out[(r["dataset"], r["frame_a"], d)] = (180.0 if pe is None else pe, me)
    return out or None


def summ(Dct):
    k = sorted(Dct)
    e = np.array([Dct[x][0] for x in k])
    m = np.array([np.nan if Dct[x][1] is None else Dct[x][1] for x in k])
    return dict(n=len(k), auc=X.pose_auc(e).tolist(), med=float(np.nanmedian(m)), n_fail=int((e >= 180).sum()),
                n_err_undefined=int(np.isnan(m).sum()))


def main():
    seg = X.seg_map()
    src = {"proposed": f"{A}/eval/orig_d8/acc_d{{d}}.jsonl"}
    for m in G0:
        src[m] = f"{A}/x4/{m}/acc_d{{d}}.jsonl"
    for m in G4:
        src[f"{m}_coarse"] = f"{A}/x4/{m}_coarse/acc_d{{d}}.jsonl"
    src["sift"] = f"{A}/x4/sift/acc_d{{d}}.jsonl"
    for m in ("matchformer", "lgfctr"):
        src[f"{m}_ft"] = f"{A}/x4/{m}_ft/acc_d{{d}}.jsonl"
        src[f"{m}_ft_coarse"] = f"{A}/eval/tp_{m}_ft_s1/coarse/usegeo__{m}_ft_coarse__d{{d}}.acc.jsonl"
        src[f"{m}_official_coarse"] = f"{A}/eval/tp_{m}_official/coarse/usegeo__{m}_official_coarse__d{{d}}.acc.jsonl"
    data, res = {}, {}
    for lab, fmt in src.items():
        for v in ("none", "B"):
            d = load(fmt, v)
            if d is not None:
                data[(lab, v)] = d
        if (lab, "none") in data or (lab, "B") in data:
            res[lab] = {}
            for v in ("none", "B"):
                if (lab, v) in data:
                    s = summ(data[(lab, v)])
                    res[lab].update({f"auc_{v}": s["auc"], f"med_{v}": s["med"], f"n_{v}": s["n"], f"fail_{v}": s["n_fail"]})
    ref = data.get(("proposed", "B"))
    if ref is not None:
        for (lab, v), d in data.items():
            if lab == "proposed":
                continue
            c = X.compare(ref, d, seg, n_iid=200, n_cl=10000)
            res[lab][f"vs_proposed_{v}"] = dict(d_sub=c["d_sub"], d_auc5=c["d_auc"][2], ci95_d_sub=c["cluster"]["d_sub_ci95"],
                                                ci95_d_auc5=c["cluster"]["d_auc_ci95"][2], n_subset=c["n_subset"])
    d4 = {}
    for m in ("matchformer", "lgfctr"):
        for cellname, a, b in (("complete", (f"{m}_ft", "none"), (m, "none")),
                               ("coarse_refined", (f"{m}_ft_coarse", "B"), (f"{m}_official_coarse", "B"))):
            if a in data and b in data:
                c = X.compare(data[a], data[b], seg, n_iid=200, n_cl=10000)
                better = (c["cluster"]["d_auc_ci95"][2][0] > 0) or (c["cluster"]["d_sub_ci95"][1] < 0)
                d4[f"{m}|{cellname}"] = dict(d_auc5=c["d_auc"][2], ci95_d_auc5=c["cluster"]["d_auc_ci95"][2],
                                             d_err=c["d_sub"], ci95_d_err=c["cluster"]["d_sub_ci95"],
                                             verdict="aerial training explains part of the gap" if better else
                                             "aerial training does not explain the gap")
    res["_d4"] = d4
    cB = [res[f"{m}_coarse"]["med_B"] for m in G4 if m != "usulan" and f"{m}_coarse" in res and "med_B" in res[f"{m}_coarse"]]
    cA = [res[f"{m}_coarse"]["auc_B"][2] for m in G4 if m != "usulan" and f"{m}_coarse" in res and "auc_B" in res[f"{m}_coarse"]]
    kB = [res[m]["med_B"] for m in G0 if m in res and "med_B" in res[m] and m != "swinmatcher_tgrs"]
    res["_summary"] = dict(coarse_B_err_min=min(cB) if cB else None, coarse_B_err_max=max(cB) if cB else None,
                           coarse_B_auc_min=min(cA) if cA else None, coarse_B_auc_max=max(cA) if cA else None,
                           complete_B_err_min=min(kB) if kB else None, complete_B_err_max=max(kB) if kB else None,
                           n_coarse=len(cB), n_complete=len(kB))
    os.makedirs(f"{A}/ctrl", exist_ok=True)
    json.dump(res, open(f"{A}/ctrl/stats.json.tmp", "w"), indent=1)
    os.replace(f"{A}/ctrl/stats.json.tmp", f"{A}/ctrl/stats.json")
    for lab, v in res.items():
        if lab.startswith("_"):
            continue
        print(f"{lab:26s} " + "  ".join(f"{k}: AUC5 {v[f'auc_{k}'][2]:.2f} err {v[f'med_{k}']:.3f}" for k in ("none", "B") if f"auc_{k}" in v))
    print("ringkasan:", res["_summary"])
    for k, v in d4.items():
        print("D4", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if not isinstance(vv, list)})


if __name__ == "__main__":
    main()
