"""Revisi A IJIES -- G5: kepekaan seleksi checkpoint (R2.4) untuk run versi terkirim (orig_d8).

(i) Akurasi UseGeo d = 1-2 (dengan penghalus) setiap epoch 1-12 (arev_eval.py orig_d8 acc_all) -> sebaran AUC@5 dan
galat median atas epoch; (ii) peluang tiap epoch terpilih oleh aturan q15 (rerata inlier RANSAC tertinggi) bila ke-15
pasangan q15 diresampel dengan pengembalian, 10.000x, benih 20261101, dari inlier per pasangan per epoch
(eval/orig_d8/_curve_tmp/q15__orig_d8_epNNN__d1.jsonl).
Keluaran: revisi_d/results/arev/g5/selection.json. Pakai: python revisi_d/scripts/arev_g5_selection.py
"""
import collections
import glob
import json
import os
import random
import sys
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

RD = AERIAL_ROOT + "/revisi_d"
sys.path.insert(0, f"{RD}/scripts")
sys.path.insert(0, RD)


def main():
    import numpy as np
    from e13_refine import pose_auc
    ev = f"{RD}/results/arev/eval/orig_d8"
    acc = {}
    for ep in sorted(os.listdir(f"{ev}/acc_all")):
        e, m = [], []
        for d in (1, 2):
            for line in open(f"{ev}/acc_all/{ep}/acc_d{d}.jsonl"):
                r = json.loads(line)
                if r["variant"] == "B":
                    e.append(180.0 if r.get("pose_err") is None else r["pose_err"])
                    m.append(np.nan if r.get("med_err_inb") is None else r["med_err_inb"])
        acc[ep] = dict(auc5=pose_auc(e, ths=(5,))["auc@5"], med_err=float(np.nanmedian(m)), n=len(e))
    per = {}
    for f in glob.glob(f"{ev}/_curve_tmp/q15__orig_d8_ep*__d1.jsonl"):
        ep = os.path.basename(f).split("__")[1].split("_")[-1]
        per[ep] = {json.loads(line)["i"]: json.loads(line)["n_inliers_ransac"] for line in open(f)}
    eps = sorted(e for e in per if e != "ep000")
    ids = sorted(set.intersection(*[set(per[e]) for e in eps]))
    rng, cnt = random.Random(20261101), collections.Counter()
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        cnt[max(eps, key=lambda e: sum(per[e][i] for i in s))] += 1
    prob = {e: cnt[e] / 100 for e in eps}
    likely = [e for e in eps if prob[e] >= 5]
    res = dict(per_epoch=acc, selection_prob_pct=prob, n_q15_pairs=len(ids), selected=json.load(open(f"{ev}/selected.json"))["best"],
               auc5_range=[min(v["auc5"] for v in acc.values()), max(v["auc5"] for v in acc.values())],
               err_range=[min(v["med_err"] for v in acc.values()), max(v["med_err"] for v in acc.values())],
               likely_epochs=likely, likely_auc5_range=[min(acc[e]["auc5"] for e in likely), max(acc[e]["auc5"] for e in likely)],
               likely_err_range=[min(acc[e]["med_err"] for e in likely), max(acc[e]["med_err"] for e in likely)])
    os.makedirs(f"{RD}/results/arev/g5", exist_ok=True)
    json.dump(res, open(f"{RD}/results/arev/g5/selection.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_epoch"}, indent=1))


if __name__ == "__main__":
    main()
