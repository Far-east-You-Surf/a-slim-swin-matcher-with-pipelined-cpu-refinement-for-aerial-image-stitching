"""Revisi A IJIES -- evaluasi satu run sapuan (paket T), bertahap dan tahan putus daya.

Tahap (masing-masing berpenanda selesai di revisi_d/results/arev/eval/<tag>/):
  curve  : protokol kualitas q15 untuk tiap arsip epoch (aturan seleksi usulan = rerata inlier RANSAC tertinggi)
  select : epoch terbaik di ep001..ep012 (+ epoch terakhir dicatat)
  acc    : UseGeo d = 1–2 sepanjang jalur, keluaran kasar ('none') dan dengan penghalus CPU ('B'), penilai e13
           (identik dengan Tabel 2/3 naskah)
  reg    : registrasi UseGeo d = 1–8, sheffield d = 1–3, Seneca/Bellus d = 1–3 (arev_eval_units, per unit)
Pakai: python revisi_d/scripts/arev_eval.py <tag> [curve|select|acc|reg|all]
"""
import glob
import json
import os
import subprocess
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
PY = f"{ROOT}/aerial_fp8_env/bin/python"


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    tag = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    out = f"{RD}/results/arev/eval/{tag}"
    if "--sub" in sys.argv:
        out = sys.argv[sys.argv.index("--sub") + 1]
    os.makedirs(out, exist_ok=True)
    arch = f"{ROOT}/ckpt_archive_p16_arev_{tag}"
    _dn = f"{RD}/results/arev/train/{tag}.train.done"
    if os.path.exists(_dn) and json.load(open(_dn)).get("archive"):
        arch = os.path.join(ROOT, json.load(open(_dn))["archive"])   # run lama (seed 1) memakai arsipnya sendiri
    curve_f, sel_f = f"{out}/curve.json", f"{out}/selected.json"

    if mode in ("curve", "all"):
        import torch
        import arev_models
        ET = arev_models.install_into_evaltool()
        curve = json.load(open(curve_f)) if os.path.exists(curve_f) else {}
        eps = sorted(glob.glob(f"{arch}/ep*.pth"))
        for k, p in enumerate(eps, 1):
            ep = os.path.basename(p).split("_")[0]
            if ep in curve:
                print(f"PROGRESS {k}/{len(eps)} kurva {ep} (sudah)", flush=True)
                continue
            lab = f"{tag}_{ep}"
            m = ET.build_matchers(f"arev={tag}={p}={lab}")[0]
            tmpd = f"{out}/_curve_tmp"
            os.makedirs(tmpd, exist_ok=True)
            ET.evaluate(m, "q15", 1, "kualitas", tmpd)
            rr = [json.loads(l) for l in open(f"{tmpd}/q15__{lab}__d1.jsonl")]
            curve[ep] = dict(ckpt=p, inliers=sum(r["n_inliers_ransac"] for r in rr) / len(rr),
                             per_pair=[r["n_inliers_ransac"] for r in rr])
            write_json(curve_f, curve)
            del m
            torch.cuda.empty_cache()
            print(f"PROGRESS {k}/{len(eps)} kurva {ep} inlier {curve[ep]['inliers']:.1f}", flush=True)
        if len(curve) < 12:
            raise SystemExit(f"kurva baru {len(curve)}/12 epoch")
        with open(f"{out}/curve.done", "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    if mode in ("select", "all"):
        curve = json.load(open(curve_f))
        win = {e: v for e, v in curve.items() if 1 <= int(e[2:5]) <= 12}
        best = max(win, key=lambda e: win[e]["inliers"])
        last = max(win)
        if tag.startswith("pub_"):   # prareg amandemen 1: resep publik penuh -> epoch 12, q15 hanya sekunder
            write_json(sel_f, dict(best=last, ckpt=win[last]["ckpt"], best_inliers=win[last]["inliers"], last=last,
                                   q15_best=best, q15_best_ckpt=win[best]["ckpt"], q15_best_inliers=win[best]["inliers"],
                                   rule="epoch 12 tanpa seleksi pada data privat (prareg amandemen 1)"))
        else:
            write_json(sel_f, dict(best=best, ckpt=win[best]["ckpt"], best_inliers=win[best]["inliers"], last=last,
                                   last_inliers=win[last]["inliers"],
                                   rule="rerata inlier RANSAC q15 tertinggi di ep001-ep012 (aturan seleksi usulan)"))

    if mode in ("acc", "all"):
        acc_done = f"{out}/acc.done"
        if not os.path.exists(acc_done):
            import numpy as np
            import torch
            import cv2
            import arev_models
            arev_models.install_into_evaltool()
            import stitch_demo as SD
            import usegeo_gt_eval as UG
            import e13_refine as E
            sel = json.load(open(sel_f))
            m = arev_models.build_matcher(f"arev={tag}={sel['ckpt']}={tag}_trained")
            cache = UG.DepthCache()
            h, w = m.grid
            for d in (1, 2):
                fdone = f"{out}/acc_d{d}.done"
                if os.path.exists(fdone):
                    continue
                pairs = E.along_pairs(d, (1, 2, 3))
                tmp = f"{out}/acc_d{d}.jsonl.tmp"
                with open(tmp, "w") as fj:
                    for k, (ds, A, B) in enumerate(pairs, 1):
                        pA, pB = UG.img_path(ds, A[0]), UG.img_path(ds, B[0])
                        t1, t2, ow, oh = SD.load_pair_tensor(pA, pB)
                        with torch.no_grad():
                            conf = m.model(t1.to(m.device).half(), t2.to(m.device).half()).float().cpu()
                        c = conf[0]
                        bj, bi = c.argmax(1), c.argmax(0)
                        rows = torch.arange(c.shape[0])
                        keep = bi[bj[rows]] == rows
                        i1, i2 = rows[keep].numpy(), bj[rows[keep]].numpy()
                        a = np.stack([(i1 % w + 0.5) * ow / w, (i1 // w + 0.5) * oh / h], 1).astype(np.float64)
                        b = np.stack([(i2 % w + 0.5) * ow / w, (i2 // w + 0.5) * oh / h], 1).astype(np.float64)
                        depthA = cache.get(ds, A[0])
                        H0, _ = (cv2.findHomography(a, b, cv2.USAC_MAGSAC, SD.RANSAC_THR) if len(a) >= 8
                                 else (None, None))
                        outs = {"none": b}
                        n_ok = None
                        if H0 is not None:
                            outs["B"], n_ok = E.prewarp_refine(E.gray(pA), E.gray(pB), a, b, H0, "lk")
                        else:
                            outs["B"] = b
                        for v in ("none", "B"):
                            r = E.score(a, outs[v], ow, oh, A, B, depthA)
                            r.update(model=f"{tag}_trained", variant=v, d=d, dataset=ds, frame_a=A[0], frame_b=B[0],
                                     n_coarse=int(len(a)), lk_ok=(int(n_ok) if (v == "B" and n_ok is not None) else None),
                                     h0_fail=H0 is None)
                            fj.write(json.dumps(r) + "\n")
                        if k % 50 == 0:
                            print(f"PROGRESS {k + (d - 1) * 729}/{1418} akurasi UseGeo d{d}", flush=True)
                    fj.flush()
                    os.fsync(fj.fileno())
                os.replace(tmp, f"{out}/acc_d{d}.jsonl")
                with open(fdone, "w") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            with open(acc_done, "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    if mode == "acc_all":
        import glob as _g
        eps = sorted(_g.glob(f"{arch}/ep*.pth"))
        for p in eps:
            ep = os.path.basename(p).split("_")[0]
            sub = f"{out}/acc_all/{ep}"
            if os.path.exists(f"{sub}/acc.done"):
                continue
            os.makedirs(sub, exist_ok=True)
            write_json(f"{sub}/selected.json", dict(best=ep, ckpt=p, rule="semua epoch (G5, kepekaan)"))
            subprocess.run([PY, __file__, tag, "acc", "--sub", sub], check=True)
        with open(f"{out}/acc_all.done", "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    if mode in ("reg", "all"):
        sel = json.load(open(sel_f))
        subprocess.run([PY, f"{RD}/scripts/arev_eval_units.py", "--matcher", f"arev={tag}={sel['ckpt']}={tag}_trained",
                        "--plan", "usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3", "--out", f"{out}/reg",
                        "--label", f"{tag}_trained"], check=True)
        with open(f"{out}/reg.done", "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if mode == "all":
        with open(f"{out}/all.done", "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
