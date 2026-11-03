"""Revisi A IJIES -- X3/X4/X5: registrasi benar-GT, validasi gerbang, penghalus identik, dan tampalan terukur (UseGeo).

Untuk tiap pasangan UseGeo (d = 1..8, sepanjang jalur dan belokan) dan tiap keluaran pencocok:
  hgrid  = median |H(x) - GT(x)| atas kisi 32x24 titik ber-LiDAR yang jatuh di citra 2 (px pada 1989x1320;
           crefine.geom.HGridGT, hanya diimpor)
  oracle = galat kisi homografi terbaik terhadap GT itu sendiri (batas model planar = ukuran relief/paralaks adegan)
  correct@τ = gate_v2 lolos DAN hgrid <= τ, τ ∈ {5, 10, 19.89, 39.78 px} (1% lebar = kriteria P39; 2% = registrasi kasar)
  overlap = tampalan terukur dari pose GT + kedalaman (results/a4/overlap_usegeo_gt.json, kisi 60x40)
Mode:
  hgrid  --src-dir D --name M [--label L]    : pakai H yang tersimpan di JSONL registrasi (keluaran apa adanya)
  refine --src-dir D --name M [--label L]    : titik tersimpan (npz) -> penghalus CPU identik (varian B e13) -> H baru
Unit = (label, d), ditulis atomik dengan penanda .done; aman diulang sesudah putus daya.
Pakai: python revisi_d/scripts/arev_x3_gate.py hgrid --src-dir revisi_d/results/arev/g0 --name matchformer
"""
import argparse
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/x3"
TAUS = (5.0, 10.0, 19.89, 39.78)   # px pada 1989: 0,25%, 0,5%, 1% (P39), 2% lebar (registrasi kasar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hgrid", "refine"])
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--name", required=True, help="nama matcher di berkas sumber usegeo__<name>__d<d>")
    ap.add_argument("--label", default=None)
    ap.add_argument("--d", default="1,2,3,4,5,6,7,8")
    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, RD)
    sys.path.insert(0, f"{RD}/scripts")
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from crefine import data as D
    from crefine import geom as Gm
    label = a.label or (a.name if a.mode == "hgrid" else f"{a.name}+B")
    src = a.src_dir if os.path.isabs(a.src_dir) else os.path.join(ROOT, a.src_dir)
    os.makedirs(OUT, exist_ok=True)
    ov = json.load(open(f"{RD}/results/a4/overlap_usegeo_gt.json"))
    if a.mode == "refine":
        import evaltool as ET
        import e13_refine as E
    cache = D.DepthCache()
    dl = [int(x) for x in a.d.split(",")]
    for k, d in enumerate(dl, 1):
        tag = f"usegeo__{label}__d{d}"
        done = f"{OUT}/{tag}.done"
        if os.path.exists(done):
            print(f"PROGRESS {k}/{len(dl)} {tag} (sudah)", flush=True)
            continue
        rows = {}
        for line in open(f"{src}/usegeo__{a.name}__d{d}.jsonl"):
            r = json.loads(line)
            rows[(r["dataset"], r["i"])] = r
        npz = np.load(f"{src}/usegeo__{a.name}__d{d}.npz") if a.mode == "refine" else None
        poses = {ds: D.load_poses(ds) for ds in (1, 2, 3)}
        tmp = f"{OUT}/{tag}.jsonl.tmp"
        n_ok = 0
        with open(tmp, "w") as fj:
            for (ds, i), r in sorted(rows.items()):
                P = poses[ds]
                A, B = P[i], P[i + d]
                depthA = cache.get(ds, A[0])
                hg = Gm.HGridGT(A, B, depthA)
                o = dict(dataset=ds, i=i, d=d, along=r.get("along"), label=label, pair=r["pair"],
                         overlap=ov.get(f"{ds}|{r['pair'][0]}|{r['pair'][1]}"), n_grid=int(len(hg.x)))
                Ho = hg.oracle()
                o["oracle"] = hg.err(Ho) if Ho is not None else None
                if a.mode == "hgrid":
                    H = np.array(r["H"]) if r.get("H") is not None else None
                    o.update(gate_v2=bool(r.get("gate_v2")), n_inliers=r.get("n_inliers"), n_matches=r.get("n_matches"))
                else:
                    key = f"{ds}_{i}"
                    pts = npz[key] if key in npz.files else np.zeros((0, 4), np.float32)
                    pa, pb = pts[:, :2].astype(np.float64), pts[:, 2:].astype(np.float64)
                    H, lk_ok, h0_fail = None, None, True
                    if len(pa) >= 8:
                        H0, _ = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 5.0)
                        if H0 is not None:
                            h0_fail = False
                            pA, pB = D.img_path(ds, A[0]), D.img_path(ds, B[0])
                            pb2, lk_ok = E.prewarp_refine(E.gray(pA), E.gray(pB), pa, pb, H0, "lk")
                            pb = pb2
                    Hm, mask = ET.magsac(pa, pb)
                    inl = int(mask.sum()) if mask is not None else 0
                    g = ET.gates(Hm, inl, 1989, 1320)
                    H = Hm
                    o.update(gate_v2=bool(g["gate_v2"]), n_inliers=inl, n_matches=int(len(pa)),
                             lk_ok=(int(lk_ok) if lk_ok is not None else None), h0_fail=h0_fail,
                             H=(Hm.tolist() if Hm is not None else None))
                o["hgrid"] = hg.err(H)
                for t in TAUS:
                    o[f"correct@{t:g}"] = bool(o["gate_v2"] and o["hgrid"] <= t)
                n_ok += o["correct@19.89"]
                fj.write(json.dumps(o) + "\n")
            fj.flush()
            os.fsync(fj.fileno())
        os.replace(tmp, f"{OUT}/{tag}.jsonl")
        with open(done, "w") as f:
            json.dump(dict(n=len(rows), correct_1pct=n_ok, t=time.strftime("%Y-%m-%d %H:%M:%S")), f)
        print(f"PROGRESS {k}/{len(dl)} {tag} benar@1% {n_ok}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
