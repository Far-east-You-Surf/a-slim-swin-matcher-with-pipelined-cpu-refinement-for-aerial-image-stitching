"""Alat evaluasi TUNGGAL revisi Artikel D (Tahap 0.3).

evaluate(matcher, urutan, d, protokol) -> satu rekaman JSON per pasangan (+ titik kecocokan .npz).

Menyatukan jalur yang sebelumnya tersebar (`collect_p16_curve`, `collect_depth_multipair`,
`requal_multipair`, `collect_thirdparty_multipair`, `stitch_demo.run_flight`, `usegeo_gt_eval`).
Matcher TIDAK ditulis ulang: semua dibangun lewat `stitch_demo.build_matchers`, jadi bobot,
praproses, dan seleksi mutual-NN identik dgn sumber angka naskah.

PROTOKOL (pra-registrasi amandemen 1, butir 11)
  kualitas   15 pasangan luar-domain (+5 in-domain) `collect_pilar3_multipair.PAIR_SETS`.
             Model kami: jalur LAMA persis (`gate_arch_probe.load_pair_at` = cv2.imread DGN rotasi
             EXIF, top-K mutual-NN K=3200, koordinat POJOK sel). Pihak ketiga: `match()`.
             Metrik primer = inlier RANSAC 5 px arah pts2->pts1 (`quant_sweep.evaluate_matches`).
  registrasi pasangan (i, i+d) sepanjang urutan. Titik dari `match()` (EXIF diabaikan, koordinat
             PUSAT sel), MAGSAC++ 5 px arah a->b, dua gerbang dicatat berdampingan:
               gate_v1 = `stitch_demo.h_sanity` (cond(H) < 1e7 -- CACAT, §60.12, hanya pembanding)
               gate_v2 = `recompute_sanity.sanity` + inlier >= 8 (gerbang yang berlaku)
             UseGeo: ditambah metrik kebenaran-dasar `usegeo_gt_eval.score_pair` (galat lokalisasi,
             n_correct1/2/5, AUC pose) pada pasangan sepanjang-jalur.

Kedua protokol menyimpan untuk SETIAP pasangan: n_matches, inlier RANSAC (pts2->pts1), inlier MAGSAC
(a->b), H MAGSAC, gate_v1, gate_v2, hull, dan titik kecocokan (npz) -- cukup untuk menghitung ulang
gerbang, A4.3-A4.4, dan penghalusan tanpa inferensi ulang.

MAGSAC/RANSAC OpenCV 4.13 deterministik untuk masukan identik (diuji 0.2: setRNGSeed tak berpengaruh);
`cv2.setRNGSeed(0)` tetap dipanggil per pasangan sbg penanda protokol.

Pakai (dari akar proyek, venv aerial_fp8_env):
  python revisi_d/scripts/evaltool.py --protocol kualitas --seq q15 --matchers p16_nofpn
  python revisi_d/scripts/evaltool.py --protocol registrasi --seq "G Uji" --d 1,2,3,4 --matchers baseline_p32,p16_nofpn
  python revisi_d/scripts/evaltool.py --protocol registrasi --seq usegeo --datasets 1,2,3 --d 1,2,3,4,5 ...
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "revisi_d", "scripts"))

import cv2
import numpy as np
import torch

import stitch_demo as SD
from recompute_sanity import sanity as _gate_v2
from collect_pilar3_multipair import PAIR_SETS

OUT_ROOT = "revisi_d/results/eval"
# Checkpoint DIKUNCI (Tahap 0.1, OPSI A). stitch_demo memilihnya dari kurva; di sini dipaku eksplisit.
LOCKED = {"p16_nofpn": "ckpt_archive_p16_nofpn/ep008_val4.3737.pth"}
SD.best_nofpn_ckpt = lambda: LOCKED["p16_nofpn"]

# Varian /32 dipangkas zero-shot (baris 12 dan 8 blok Tabel 2): bobot baseline, stage 3 dipangkas
# lewat `gate_pilar3._prune_stage3` -- sama persis dgn `collect_depth_multipair.build_variant`.
_orig_build = SD.build_matchers


def _p32_pruned(keep):
    from gate_pilar3 import _prune_stage3
    m = SD.OurModel(f"p32_depth{keep}", "p32", "aerial_stitching_fp16_best.pth", grid=(24, 32))
    net = m.model.float().cpu()
    net, _ = _prune_stage3(net, keep)
    m.model = net.to(m.device).half().eval()
    return m


TP_NAMES = ("efficientloftr", "lgfctr", "swinmatcher_tgrs", "matchformer", "aspanformer")


class _Capped:
    """Pembungkus E1.2 (`tp_matchers`) + pemotongan top-K ke CAP menurut `last_conf`, identik dgn
    kelas pihak ketiga stitch_demo (praktis no-op: kecocokannya ~2000-2400 < 3200)."""

    def __init__(self, inner):
        self.inner, self.label = inner, inner.label

    def match(self, p1, p2):
        a, b, ow, oh = self.inner.match(p1, p2)
        c = getattr(self.inner, "last_conf", None)
        if c is not None and len(a) > SD.CAP:
            c = c.detach().float().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
            idx = np.argsort(-c)[:SD.CAP]
            a, b = a[idx], b[idx]
        return a, b, ow, oh


def _p16_ckpt(spec):
    """Spesifikasi 'p16ckpt=<blok>=<path>=<label>': model_p16 tanpa fusi dgn jumlah blok tahap 3 bebas (E3.1)."""
    _, blocks, path, label = spec.split("=")
    import model_p16
    blocks = int(blocks)
    m = SD.OurModel.__new__(SD.OurModel)
    from quant_sweep import DEVICE
    m.label, m.grid, m.device, m.half = label, (48, 64), DEVICE, True
    net = model_p16.AerialStitchingModelP16(d_model=128, nhead=8, num_layers=8, stage3_blocks=blocks,
                                            use_fpn=False, flash=True, pretrained_backbone=False)
    sd = torch.load(path, map_location="cpu")
    miss, unexp = net.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise RuntimeError(f"{label}: checkpoint tidak cocok {miss[:3]} {unexp[:3]}")
    m.model = net.to(DEVICE).half().eval()
    return m


def _bb_ckpt(spec):
    """Spesifikasi 'bbckpt=<preset>=<path>=<label>': model_bb (tulang punggung alternatif, butir 7 tahap 3)."""
    _, preset, path, label = spec.split("=")
    import model_bb
    m = SD.OurModel.__new__(SD.OurModel)
    from quant_sweep import DEVICE
    m.label, m.grid, m.device, m.half = label, (48, 64), DEVICE, True
    base = "__".join(x for x in preset.split("__") if not x.startswith(("dh", "db")))   # pangkas sesudah muat (P14)
    net = model_bb.build(base, pretrained=False)
    sd = torch.load(path, map_location="cpu")
    miss, unexp = net.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise RuntimeError(f"{label}: checkpoint tidak cocok {miss[:3]} {unexp[:3]}")
    model_bb.prune_after_load(net, preset)
    m.model = net.to(DEVICE).half().eval()
    return m


def build_matchers(which):
    extra = {"p32_depth12": lambda: _p32_pruned(12), "p32_depth8": lambda: _p32_pruned(8)}
    for n in which.split(","):
        if n.startswith("p16ckpt="):
            extra[n] = (lambda n=n: _p16_ckpt(n))
        if n.startswith("bbckpt="):
            extra[n] = (lambda n=n: _bb_ckpt(n))
    for n in TP_NAMES:
        extra[n] = (lambda n=n: _Capped(__import__("tp_matchers").build(n)))
    names = which.split(",")
    out = []
    for n in names:
        if n in extra:
            out.append(extra[n]()); print(f"[model] {n} siap")
        else:
            out += _orig_build(n)
    return out

MIN_INL = 8


def sha1(path, n=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]


def ransac_legacy(a, b):
    """quant_sweep.evaluate_matches persis: RANSAC 5 px, arah pts2 -> pts1."""
    if len(a) < 4:
        return 0, None
    H, m = cv2.findHomography(b, a, cv2.RANSAC, 5.0)
    return (int(m.sum()) if m is not None else 0), H


def magsac(a, b):
    if len(a) < SD.MIN_MATCHES:
        return None, None
    H, m = cv2.findHomography(a, b, cv2.USAC_MAGSAC, SD.RANSAC_THR)
    if H is None or m is None:
        return None, None
    return H, m.reshape(-1).astype(bool)


def gates(H, inl, ow, oh):
    if H is None:
        return dict(gate_v1=False, gate_v2=False, v2_reason="tanpa H")
    g1 = SD.h_sanity(H, ow, oh)
    g2 = _gate_v2(H, ow, oh)
    return dict(gate_v1=bool(g1["valid"] and inl >= SD.MIN_MATCHES),
                gate_v2=bool(g2["valid"] and inl >= MIN_INL),
                cond=g1.get("cond"), warp_area_ratio=g2.get("warp_area_ratio"),
                convex=g2.get("convex"), v2_reason=g2.get("reason"))


# ------------------------------------------------------------------ titik kecocokan
def points_kualitas(m, p1, p2):
    """Model kami -> jalur lama kualitas; pihak ketiga -> match() (= collect_thirdparty)."""
    if isinstance(m, SD.OurModel):
        from gate_arch_probe import load_pair_at
        from gate_arch_probe2 import topk_points
        t1, t2, ow, oh = load_pair_at(p1, p2, SD.INPUT_HW)
        with torch.no_grad():
            conf = m.model(t1.to(m.device).half(), t2.to(m.device).half()).float().cpu()
        a, b = topk_points(conf, ow, oh, m.grid, SD.SAT_K, mutual=True)
        return np.asarray(a, np.float64), np.asarray(b, np.float64), ow, oh
    a, b, ow, oh = m.match(p1, p2)
    return np.asarray(a, np.float64), np.asarray(b, np.float64), ow, oh


def points_registrasi(m, p1, p2):
    a, b, ow, oh = m.match(p1, p2)
    return np.asarray(a, np.float64), np.asarray(b, np.float64), ow, oh


# ------------------------------------------------------------------ daftar pasangan
def flight_pairs(seq, d):
    folder = SD.FLIGHTS[seq]
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg")))
    return [dict(i=i, p1=files[i], p2=files[i + d]) for i in range(len(files) - d)]


def quality_pairs(seq):
    key = {"q15": "g_uji_out_of_domain", "q5in": "aerial234_in_domain"}[seq]
    return [dict(i=k, p1=p1, p2=p2) for k, (p1, p2) in enumerate(PAIR_SETS[key])]


def usegeo_pairs(ds, d):
    import usegeo_gt_eval as UG
    rows = UG.load_poses(ds)
    out = []
    for i in range(len(rows) - d):
        steps = [(UG.dkappa(rows[k][6], rows[k + 1][6]),
                  float(np.hypot(rows[k][1] - rows[k + 1][1], rows[k][2] - rows[k + 1][2])))
                 for k in range(i, i + d)]
        along = all(dk < UG.DKAPPA_MAX and st < UG.STEP_MAX for dk, st in steps)
        out.append(dict(i=i, p1=UG.img_path(ds, rows[i][0]), p2=UG.img_path(ds, rows[i + d][0]),
                        A=rows[i], B=rows[i + d], along=along, dataset=ds,
                        dkappa_max=round(max(dk for dk, _ in steps), 2)))
    return out


# ------------------------------------------------------------------ inti
def evaluate(m, seq, d, protocol, out_dir, datasets=(1, 2, 3), along_only=False, want_pose=True):
    """Satu (matcher, urutan, d, protokol) -> JSONL per pasangan + npz titik. Mengembalikan ringkasan."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{seq.replace(' ', '_')}__{m.label}__d{d}"
    jl, npz = f"{out_dir}/{tag}.jsonl", f"{out_dir}/{tag}.npz"
    if protocol == "kualitas":
        plist, getpts = quality_pairs(seq), points_kualitas
    elif seq == "usegeo":
        plist = [p for ds in datasets for p in usegeo_pairs(ds, d)]
        if along_only:
            plist = [p for p in plist if p["along"]]
        getpts = points_registrasi
    else:
        plist, getpts = flight_pairs(seq, d), points_registrasi
    if seq == "usegeo":
        import usegeo_gt_eval as UG
        cache = UG.DepthCache()

    recs, pts_store = [], {}
    with open(jl, "w") as fj:
        for p in plist:
            cv2.setRNGSeed(0)
            r = dict(seq=seq, d=d, i=p["i"], matcher=m.label, protocol=protocol,
                     pair=[os.path.basename(p["p1"]), os.path.basename(p["p2"])])
            for k in ("dataset", "along", "dkappa_max"):
                if k in p:
                    r[k] = p[k]
            t0 = time.perf_counter()
            try:
                a, b, ow, oh = getpts(m, p["p1"], p["p2"])
            except Exception as e:
                r.update(error=f"{type(e).__name__}: {e}", gate_v1=False, gate_v2=False, n_matches=0)
                recs.append(r); fj.write(json.dumps(r) + "\n"); continue
            r["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            r["n_matches"] = int(len(a))
            r["image_wh"] = [int(ow), int(oh)]
            r["n_inliers_ransac"], _ = ransac_legacy(a, b)
            H, mask = magsac(a, b)
            r["n_inliers"] = int(mask.sum()) if mask is not None else 0
            r["H"] = H.tolist() if H is not None else None
            r.update(gates(H, r["n_inliers"], ow, oh))
            if mask is not None and mask.sum() >= 3:
                r["hull_frac"] = SD.hull_area_frac(a[mask], ow, oh)
                r["grid_occ"] = SD.grid_occupancy(a[mask], ow, oh)
            if seq == "usegeo":
                import usegeo_gt_eval as UG
                try:
                    depthA = cache.get(p["dataset"], p["A"][0])
                    gt = UG.score_pair(a, b, ow, oh, p["A"], p["B"], depthA, want_pose)
                    for k, v in gt.items():
                        if k not in ("n_matches", "n_inliers", "inlier_ratio", "gate"):
                            r[k] = v
                    r["gate_v1_usegeo_check"] = gt.get("gate")
                except RuntimeError as e:
                    r["gt_error"] = str(e)
            pts_store[f"{p.get('dataset', 0)}_{p['i']}"] = np.concatenate([a, b], 1).astype(np.float32)
            recs.append(r)
            fj.write(json.dumps(r) + "\n")
    np.savez_compressed(npz, **pts_store)
    ok = [r for r in recs if "error" not in r]
    summ = dict(seq=seq, d=d, matcher=m.label, protocol=protocol, n=len(recs), n_error=len(recs) - len(ok),
                mean_inliers_ransac=float(np.mean([r["n_inliers_ransac"] for r in ok])) if ok else None,
                mean_inliers_magsac=float(np.mean([r["n_inliers"] for r in ok])) if ok else None,
                success_v1=float(np.mean([r["gate_v1"] for r in recs]) * 100) if recs else None,
                success_v2=float(np.mean([r["gate_v2"] for r in recs]) * 100) if recs else None,
                jsonl=jl)
    if seq == "usegeo":
        al = [r for r in recs if r.get("along")]
        summ["n_along"] = len(al)
        summ["success_v2_along"] = float(np.mean([r["gate_v2"] for r in al]) * 100) if al else None
        summ["success_v1_along"] = float(np.mean([r["gate_v1"] for r in al]) * 100) if al else None
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["kualitas", "registrasi"])
    ap.add_argument("--seq", required=True, help="q15 | q5in | 'G Uji' | G6 | sheffield | usegeo")
    ap.add_argument("--d", default="1")
    ap.add_argument("--matchers", required=True)
    ap.add_argument("--datasets", default="1,2,3")
    ap.add_argument("--along-only", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or f"{OUT_ROOT}/{a.protocol}"
    ds = tuple(int(x) for x in a.datasets.split(","))
    meta = dict(argv=sys.argv, t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), cv2=cv2.__version__,
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                input_hw=SD.INPUT_HW, sat_k=SD.SAT_K, locked=LOCKED,
                locked_sha1={k: sha1(v) for k, v in LOCKED.items()})
    summaries = []
    for m in build_matchers(a.matchers):
        for d in [int(x) for x in a.d.split(",")]:
            s = evaluate(m, a.seq, d, a.protocol, out_dir, ds, a.along_only, not a.no_pose)
            print(json.dumps(s), flush=True)
            summaries.append(s)
        del m
        torch.cuda.empty_cache()
    import re as _re
    tag = f"{a.protocol}__{a.seq.replace(' ', '_')}__{a.matchers.replace(',', '+')}__d{a.d.replace(',', '')}"
    tag = _re.sub(r"[^A-Za-z0-9_+.-]", "_", tag)[:180]
    json.dump(dict(meta=meta, summaries=summaries), open(f"{out_dir}/summary__{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
