"""Revisi A IJIES -- X6: uji tingkat urutan mosaik pada UseGeo (butir R2.5).

Untuk tiap segmen jalur lurus UseGeo (`crefine/data.py:segments`, >= 10 bingkai) dan tiap pencocok:
  rantai  : T_k = T_{k-1} · H_{k-1->k}^{-1} memetakan bingkai k ke bingkai awal segmen s (H = homografi registrasi
            d = 1 yang tersimpan, piksel 1989x1320)
  GT      : pusat bingkai k diproyeksikan dengan kedalaman LiDAR + pose GT ke bidang citra bingkai s
            (crefine.data.gt_project; boleh di luar bingkai)
  drift   : |T_k(c) - GT_s(c_k)| dalam piksel bingkai s dan meter (GSD = kedalaman median bingkai s / f)
            -> drift akhir segmen, drift berjendela 10 bingkai (rantai dari k-10), sebaran
  pemulihan saat link (k, k+1) ditolak gerbang: P0 putus (rantai baru), P1 jembatan d = 2, P2 jembatan d <= 3;
            dilaporkan fraksi bingkai tertempatkan, jumlah putus, drift
  sambungan: NCC abu-abu pada skala 1/4 di daerah tumpang tindih bingkai k dan k+1 sesudah dilengkungkan dengan H link
Pakai: python revisi_d/scripts/arev_x6_sequence.py <label>=<dir_registrasi>:<nama_berkas> [...]
  mis.  proposed_coarse=revisi_d/results/eval/registrasi:p16_nofpn  proposed_final=revisi_d/results/arev/x3:proposed_final
Keluaran: revisi_d/results/arev/x6/<label>.json
"""
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/x6"
W, H = 1989, 1320
WIN = 10


def load_links(src, name, d):
    f = f"{src}/usegeo__{name}__d{d}.jsonl"
    out = {}
    if not os.path.exists(f):
        return out
    for line in open(f):
        r = json.loads(line)
        Hm = r.get("H")
        out[(r["dataset"], r["i"])] = (None if Hm is None else Hm, bool(r.get("gate_v2")))
    return out


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, RD)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    from crefine import data as D
    os.makedirs(OUT, exist_ok=True)
    f_px = 1160.075                     # fokus pada 1989 px (intrinsik UseGeo dibagi 4)
    cache = D.DepthCache()
    gray_cache = {}

    def gray_small(ds, name):
        k = (ds, name)
        if k not in gray_cache:
            im = cv2.imread(D.img_path(ds, name), cv2.IMREAD_GRAYSCALE)
            gray_cache[k] = cv2.resize(im, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
            if len(gray_cache) > 64:
                gray_cache.pop(next(iter(gray_cache)))
        return gray_cache[k]

    c = np.array([[W / 2 - 0.5, H / 2 - 0.5]])
    stride = int(os.environ.get("AREV_STRIDE", "1"))   # subsampel bingkai: link d = stride (tampalan lebih tipis)
    for spec in sys.argv[1:]:
        label, rest = spec.split("=", 1)
        src, name = rest.rsplit(":", 1)
        src = src if os.path.isabs(src) else os.path.join(ROOT, src)
        outp = f"{OUT}/{label}" + (f"__s{stride}" if stride > 1 else "") + ".json"
        if os.path.exists(outp):
            print("lewat (sudah):", label, flush=True)
            continue
        L = {d: load_links(src, name, d) for d in range(1, 9)}
        if not L[1]:
            print("tidak ada berkas d1 untuk", label, flush=True)
            continue
        segs_out = []
        for ds in (1, 2, 3):
            rows = D.load_poses(ds)
            seg, _ = D.segments(ds)
            ids = [seg[r[0]] for r in rows]
            for sid in sorted(set(ids)):
                idx = [i for i, s in enumerate(ids) if s == sid]
                if len(idx) < WIN:              # segmen yang sama untuk semua langkah subsampel
                    continue
                idx = idx[::stride]
                if len(idx) < 3:
                    continue
                s0 = idx[0]
                depth_s = cache.get(ds, rows[s0][0])
                gsd = float(np.median(depth_s[np.isfinite(depth_s) & (depth_s > 0)])) / f_px
                res = {}
                for pol, dmax in (("P0", 1), ("P1", 2), ("P2", 3)):
                    T = {s0: np.eye(3)}
                    start = {s0: s0}
                    breaks, k = 0, s0
                    while k < idx[-1]:
                        placed = False
                        for d in range(1, dmax + 1):
                            dd = d * stride
                            if k + dd > idx[-1] or dd > 8:
                                break
                            Hm, ok = L[dd].get((ds, k), (None, False))
                            if ok and Hm is not None:
                                T[k + dd] = T[k] @ np.linalg.inv(np.array(Hm))
                                start[k + dd] = start[k]
                                k += dd
                                placed = True
                                break
                        if not placed:
                            breaks += 1
                            k += stride
                            T[k], start[k] = np.eye(3), k
                    drift_end, drift_win = [], []
                    for kk in sorted(T):
                        st = start[kk]
                        if kk == st:
                            continue
                        dk = cache.get(ds, rows[kk][0])
                        ub, vb, valid = D.gt_project(rows[kk], rows[st], dk, c[:, 0], c[:, 1])
                        if not valid[0]:
                            continue
                        p = T[kk] @ np.array([c[0, 0], c[0, 1], 1.0])
                        p = p[:2] / p[2]
                        e = float(np.hypot(p[0] - ub[0], p[1] - vb[0]))
                        if kk == max(x for x in T if start[x] == st):
                            drift_end.append(e * gsd)
                        kw = kk - stride * max(1, round(WIN / stride))   # jendela ±10 bingkai asli
                        if kw in T and start[kw] == st:
                            Tw = np.linalg.inv(T[kw]) @ T[kk]
                            ub2, vb2, v2 = D.gt_project(rows[kk], rows[kw], dk, c[:, 0], c[:, 1])
                            if v2[0]:
                                q = Tw @ np.array([c[0, 0], c[0, 1], 1.0])
                                drift_win.append(float(np.hypot(q[0] / q[2] - ub2[0], q[1] / q[2] - vb2[0])) * gsd)
                    res[pol] = dict(n_frames=len(idx), placed=len(T), breaks=breaks,
                                    drift_end_m=drift_end, drift_win10_m=drift_win)
                # kualitas sambungan pada link d = 1 yang lolos gerbang
                ncc = []
                for k in idx[:-1]:
                    Hm, ok = L[stride].get((ds, k), (None, False))
                    if not ok or Hm is None:
                        continue
                    A, B = gray_small(ds, rows[k][0]), gray_small(ds, rows[k + stride][0])
                    S = np.diag([0.25, 0.25, 1.0])
                    Hs = S @ np.array(Hm) @ np.linalg.inv(S)
                    Bw = cv2.warpPerspective(B, np.linalg.inv(Hs), (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR,
                                             borderValue=-1)
                    m = Bw >= 0
                    if m.sum() < 1000:
                        continue
                    a, b = A[m], Bw[m]
                    a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
                    ncc.append(float((a * b).mean()))
                segs_out.append(dict(dataset=ds, segment=sid, frames=len(idx), gsd_m=gsd, policies=res, ncc=ncc))
        def agg(pol):
            fr = sum(s["frames"] for s in segs_out)
            pl = sum(s["policies"][pol]["placed"] for s in segs_out)
            de = [x for s in segs_out for x in s["policies"][pol]["drift_end_m"]]
            dw = [x for s in segs_out for x in s["policies"][pol]["drift_win10_m"]]
            return dict(frames=fr, placed_frac=pl / max(fr, 1), breaks=sum(s["policies"][pol]["breaks"] for s in segs_out),
                        drift_end_med_m=float(np.median(de)) if de else None,
                        drift_end_p90_m=float(np.percentile(de, 90)) if de else None,
                        drift_win10_med_m=float(np.median(dw)) if dw else None,
                        drift_win10_p90_m=float(np.percentile(dw, 90)) if dw else None)
        allncc = [x for s in segs_out for x in s["ncc"]]
        summ = dict(label=label, src=src, name=name, stride=stride, segments=len(segs_out), P0=agg("P0"), P1=agg("P1"), P2=agg("P2"),
                    ncc_med=float(np.median(allncc)) if allncc else None,
                    ncc_p10=float(np.percentile(allncc, 10)) if allncc else None, t=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(dict(summary=summ, segments=segs_out), open(outp + ".tmp", "w"), indent=1)
        os.replace(outp + ".tmp", outp)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
