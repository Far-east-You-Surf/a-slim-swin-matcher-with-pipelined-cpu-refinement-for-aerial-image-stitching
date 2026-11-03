"""Revisi A IJIES -- G4: jalur KASAR terkompilasi tiap pembanding (dan usulan) pada protokol registrasi yang sama.

Engine kasar TensorRT FP16 yang sama dengan P2/P9 (results/p2c/engines/<m>_coarse.engine; usulan b8_s16) -> matriks
keyakinan -> mutual-NN tanpa ambang -> pusat sel (kelas p9_coarse_B.Coarse, diimpor). Untuk tiap pasangan disimpan
seperti evaltool registrasi: H MAGSAC++ 5 px, gate_v2, inlier, titik (npz) -> X3 menghitung galat kisi-H dan
menerapkan penghalus identik ('refine') untuk protokol kasar vs akhir yang sama bagi semua pencocok.
Urutan: UseGeo d = 1..8 (semua pasangan, termasuk belokan), sheffield/Seneca/Bellus d = 1..3.
Unit = (pencocok, urutan, d), atomik + .done; aman diulang.
Pakai: python revisi_d/scripts/arev_g4_coarse.py <pencocok> [--plan usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3]
"""
import argparse
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g4"
LOWTEX = {"seneca": AERIAL_DATA + "/odm_lowtex/odm_data_seneca/images",
          "bellus": AERIAL_DATA + "/odm_lowtex/odm_data_bellus/images"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matcher")
    ap.add_argument("--plan", default="usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3")
    a = ap.parse_args()
    sys.path.insert(0, f"{RD}/scripts")
    import p9_coarse_B as P9          # memuat e13_refine (chdir ke ROOT) dan kelas Coarse
    import numpy as np
    import stitch_demo as SD
    import evaltool as ET
    SD.FLIGHTS.update(LOWTEX)
    from arev_eval_units import parse_plan
    os.makedirs(OUT, exist_ok=True)
    units = parse_plan(a.plan)
    C = None
    label = f"{a.matcher}_coarse"
    for k, (seq, d) in enumerate(units, 1):
        tag = f"{seq}__{label}__d{d}"
        if os.path.exists(f"{OUT}/{tag}.done"):
            print(f"PROGRESS {k}/{len(units)} {tag} (sudah)", flush=True)
            continue
        if C is None:
            C = P9.Coarse(a.matcher)
        if seq == "usegeo":
            plist = [p for ds in (1, 2, 3) for p in ET.usegeo_pairs(ds, d)]
        else:
            plist = ET.flight_pairs(seq, d)
        pts, n_ok = {}, 0
        tmp = f"{OUT}/{tag}.jsonl.tmp"
        with open(tmp, "w") as fj:
            for p in plist:
                r = dict(seq=seq, d=d, i=p["i"], matcher=label, protocol="registrasi",
                         pair=[os.path.basename(p["p1"]), os.path.basename(p["p2"])])
                for kk in ("dataset", "along", "dkappa_max"):
                    if kk in p:
                        r[kk] = p[kk]
                t0 = time.perf_counter()
                pa, pb, ow, oh = C.match(p["p1"], p["p2"])
                r["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                r["n_matches"], r["image_wh"] = int(len(pa)), [int(ow), int(oh)]
                H, mask = ET.magsac(pa, pb)
                r["n_inliers"] = int(mask.sum()) if mask is not None else 0
                r["H"] = H.tolist() if H is not None else None
                r.update(ET.gates(H, r["n_inliers"], ow, oh))
                n_ok += bool(r["gate_v2"])
                pts[f"{p.get('dataset', 0)}_{p['i']}"] = np.concatenate([pa, pb], 1).astype(np.float32)
                fj.write(json.dumps(r) + "\n")
            fj.flush()
            os.fsync(fj.fileno())
        np.savez_compressed(f"{OUT}/{tag}.npz.tmp.npz", **pts)
        os.replace(f"{OUT}/{tag}.npz.tmp.npz", f"{OUT}/{tag}.npz")
        os.replace(tmp, f"{OUT}/{tag}.jsonl")
        with open(f"{OUT}/{tag}.done", "w") as f:
            json.dump(dict(n=len(plist), gate_v2=n_ok, grid=list(C.grid), t=time.strftime("%Y-%m-%d %H:%M:%S")), f)
        print(f"PROGRESS {k}/{len(units)} {tag} gate_v2 {n_ok}/{len(plist)}", flush=True)
    with open(f"{OUT}/{label}.alldone", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
