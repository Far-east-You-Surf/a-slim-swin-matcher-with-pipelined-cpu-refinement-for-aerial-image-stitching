"""Revisi A IJIES -- X4: akurasi UseGeo d = 1-2 (sepanjang jalur) dari titik tersimpan, tanpa dan dengan penghalus
CPU identik dengan usulan ('none' / 'B'), untuk skenario terkendali (R1.1, R2.6).

Sumber titik (npz kunci "<dataset>_<i>", larik [ax, ay, bx, by] dalam piksel citra 1989x1320):
  keluaran LENGKAP pembanding sebagaimana terbit, tanpa batas kecocokan  results/arev/g0       --name <m>
  jalur KASAR terkompilasi pembanding (TensorRT, MNN tanpa ambang)       results/arev/g4       --name <m>_coarse
  SIFT (OpenCV, rasio Lowe)                                              results/p30           --name sift
Penghalus dan penilai = kode yang sama dengan evaluasi usulan dan pembanding latih ulang (e13_refine.prewarp_refine
'lk', e13_refine.score; H0 = MAGSAC++ 5 px), jadi kolom 'B' setiap pencocok dihitung dengan prosedur identik.
Keluaran: revisi_d/results/arev/x4/<label>/acc_d{1,2}.jsonl (+ .done), format = results/arev/eval/<run>/acc_d*.jsonl.
Pakai: python revisi_d/scripts/arev_x4_refine_acc.py --src-dir <dir> --name <nama> --label <label>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    import cv2
    import numpy as np
    import e13_refine as E
    import evaltool as ET
    import stitch_demo as SD
    import usegeo_gt_eval as UG
    cv2.setNumThreads(1)
    out = f"{RD}/results/arev/x4/{a.label}"
    os.makedirs(out, exist_ok=True)
    cache = UG.DepthCache()
    for d in (1, 2):
        f = f"{out}/acc_d{d}.jsonl"
        if os.path.exists(f"{out}/acc_d{d}.done"):
            continue
        z = np.load(f"{a.src_dir}/usegeo__{a.name}__d{d}.npz")
        pairs = [p for ds in (1, 2, 3) for p in ET.usegeo_pairs(ds, d) if p.get("along")]
        done = set()
        if os.path.exists(f):                         # lanjut: baris utuh dibaca, ekor rusak dipotong
            good = []
            for line in open(f):
                try:
                    r = json.loads(line)
                    good.append(line)
                    done.add((r["dataset"], r["frame_a"], r["variant"]))
                except json.JSONDecodeError:
                    break
            open(f, "w").writelines(good)
        n_missing = 0
        with open(f, "a") as fo:
            for k, p in enumerate(pairs):
                A, B = p["A"], p["B"]
                if (p["dataset"], A[0], "B") in done:
                    continue
                key = f"{p['dataset']}_{p['i']}"
                if key in z.files and len(z[key]):
                    pts = z[key].astype(np.float64)
                    pa, pb = pts[:, :2].copy(), pts[:, 2:].copy()
                else:
                    pa = pb = np.zeros((0, 2))
                    n_missing += 1
                g1, g2 = E.gray(p["p1"]), E.gray(p["p2"])
                oh, ow = g1.shape[:2]
                depthA = cache.get(p["dataset"], A[0])
                H0 = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, SD.RANSAC_THR)[0] if len(pa) >= 8 else None
                outs = {"none": pb, "B": E.prewarp_refine(g1, g2, pa, pb, H0, "lk")[0] if H0 is not None else pb}
                for v, bb in outs.items():
                    if (p["dataset"], A[0], v) in done:
                        continue
                    s = E.score(pa, bb, ow, oh, A, B, depthA) if len(pa) else dict(n=0, med_err_inb=None, pose_err=None)
                    s.update(model=a.label, variant=v, d=d, dataset=p["dataset"], frame_a=A[0], frame_b=B[0],
                             h0_fail=H0 is None and len(pa) > 0)
                    fo.write(json.dumps(s) + "\n")
                fo.flush()
                if k % 100 == 0:
                    os.fsync(fo.fileno())
                    print(f"PROGRESS {k}/{len(pairs)} d{d} {a.label}", flush=True)
            os.fsync(fo.fileno())
        open(f"{out}/acc_d{d}.done", "w").write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), src=a.src_dir,
                                                                 name=a.name, n_pairs=len(pairs), n_missing=n_missing)))
    print(f"[x4] SELESAI {a.label}", flush=True)


if __name__ == "__main__":
    main()
