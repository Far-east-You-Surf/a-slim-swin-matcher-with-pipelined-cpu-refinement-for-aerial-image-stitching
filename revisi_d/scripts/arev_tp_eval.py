"""Revisi A IJIES (T3) -- evaluasi pembanding hasil latih ulang pada data udara (LGFCTR, MatchFormer).

Tahap (penanda di revisi_d/results/arev/eval/<tag>/):
  curve  : q15 protokol kualitas untuk epoch 0 (bobot resmi; kontrol) dan tiap arsip epoch -> rerata inlier RANSAC
  select : epoch terbaik ep001..ep012 (aturan yang sama dengan usulan)
  reg    : keluaran LENGKAP (tahap halus sendiri, tanpa batas kecocokan) -- UseGeo d1-8 (+ metrik GT LiDAR/pose di
           pasangan sepanjang jalur), sheffield d1-3, Seneca/Bellus d1-3 (arev_eval_units, AREV_TP_WEIGHTS)
  coarse : jalur KASAR PyTorch (p2c_<m>.CoarseConf = rumus dual-softmax repo) -> mutual-NN tanpa ambang -> pusat
           sel; disimpan seperti G4 (H, gate_v2, npz) untuk X3, ditambah skor GT UseGeo d1-2 tanpa/dengan penghalus
           identik (pola p9_coarse_B)
Pakai: python revisi_d/scripts/arev_tp_eval.py <lgfctr|matchformer> <tag> [curve|select|reg|coarse|all]
       python revisi_d/scripts/arev_tp_eval.py <lgfctr|matchformer> tp_<m>_official coarse
         -> sel tabel 2x2 D4 "bobot resmi x jalur kasar + penghalus identik": jalur kode SAMA dengan sel latih ulang,
            bobot = TM.WEIGHTS[m] resmi, label <m>_official_coarse (tanpa kurva/seleksi)
"""
import glob
import json
import os
import subprocess
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
PY = f"{ROOT}/aerial_fp8_env/bin/python"
LOWTEX = {"seneca": AERIAL_DATA + "/odm_lowtex/odm_data_seneca/images",
          "bellus": AERIAL_DATA + "/odm_lowtex/odm_data_bellus/images"}


def wj(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def main():
    model, tag = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    os.environ["STITCH_CAP"] = str(10 ** 9)
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    out = f"{RD}/results/arev/eval/{tag}"
    os.makedirs(out, exist_ok=True)
    arch = f"{ROOT}/ckpt_archive_{tag}"
    curve_f, sel_f = f"{out}/curve.json", f"{out}/selected.json"
    official = tag.endswith("_official")
    if official:
        if mode != "coarse":
            raise SystemExit("tag *_official hanya untuk tahap coarse")
        import tp_matchers as TM0
        wj(sel_f, dict(best="official", ckpt=TM0.WEIGHTS[model], rule="bobot resmi (sel kontrol D4)"))

    if mode in ("curve", "all") and not os.path.exists(f"{out}/curve.done"):
        import torch
        import evaltool as ET
        import tp_matchers as TM
        curve = json.load(open(curve_f)) if os.path.exists(curve_f) else {}
        eps = [("ep000", TM.WEIGHTS[model])] + [(os.path.basename(p)[:5], p) for p in sorted(glob.glob(f"{arch}/ep*.pth"))]
        for k, (ep, p) in enumerate(eps, 1):
            if ep in curve:
                continue
            TM.WEIGHTS[model] = p
            m = ET.build_matchers(model)[0]
            m.label = m.inner.label = f"{model}_ft_{ep}"
            tmpd = f"{out}/_curve_tmp"
            os.makedirs(tmpd, exist_ok=True)
            ET.evaluate(m, "q15", 1, "kualitas", tmpd)
            rr = [json.loads(l) for l in open(f"{tmpd}/q15__{m.label}__d1.jsonl")]
            curve[ep] = dict(ckpt=p, inliers=sum(r["n_inliers_ransac"] for r in rr) / len(rr))
            wj(curve_f, curve)
            del m
            torch.cuda.empty_cache()
            print(f"PROGRESS {k}/{len(eps)} kurva {ep} inlier {curve[ep]['inliers']:.1f}", flush=True)
        if sum(1 for e in curve if e != "ep000") >= 12:
            open(f"{out}/curve.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            raise SystemExit("kurva belum lengkap")

    if mode in ("select", "all"):
        curve = json.load(open(curve_f))
        win = {e: v for e, v in curve.items() if e != "ep000" and 1 <= int(e[2:5]) <= 12}
        best = max(win, key=lambda e: win[e]["inliers"])
        wj(sel_f, dict(best=best, ckpt=win[best]["ckpt"], best_inliers=win[best]["inliers"],
                       zero_shot_inliers=curve["ep000"]["inliers"], rule="rerata inlier RANSAC q15 tertinggi ep001-ep012"))

    if mode in ("reg", "all") and not os.path.exists(f"{out}/reg.done"):
        sel = json.load(open(sel_f))
        env = dict(os.environ, AREV_TP_WEIGHTS=f"{model}={sel['ckpt']}")
        subprocess.run([PY, f"{RD}/scripts/arev_eval_units.py", "--matcher", model, "--label", f"{model}_ft",
                        "--plan", "usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3", "--out", f"{out}/reg"],
                       check=True, env=env)
        open(f"{out}/reg.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))

    if mode in ("coarse", "all") and not os.path.exists(f"{out}/coarse.done"):
        import cv2
        import numpy as np
        import torch
        import e13_refine as E
        import evaltool as ET
        import stitch_demo as SD
        import tp_matchers as TM
        import usegeo_gt_eval as UG
        SD.FLIGHTS.update(LOWTEX)
        sel = json.load(open(sel_f))
        TM.WEIGHTS[model] = sel["ckpt"]
        P2 = __import__(f"p2c_{model}")
        raw, _ = TM.BUILDERS[model]()
        net = P2.CoarseConf(raw.cuda().eval()).eval()
        lab = f"{model}_official_coarse" if official else f"{model}_ft_coarse"
        cdir = f"{out}/coarse"
        os.makedirs(cdir, exist_ok=True)
        cache = UG.DepthCache()

        def match(p1, p2):
            g = []
            for p in (p1, p2):
                im = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                oh, ow = im.shape[:2]
                x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (640, 480))
                g.append(torch.from_numpy(x).cuda().float()[None, None] / 255)
            with torch.no_grad():
                c = net(g[0], g[1])[0].float()
            L = c.shape[0]
            s = int(round((480 * 640 / L) ** 0.5))
            h, w = 480 // s, 640 // s
            bj, bi = c.argmax(1), c.argmax(0)
            rows = torch.arange(L, device="cuda")
            keep = bi[bj] == rows
            i1, i2 = rows[keep], bj[keep]
            a = torch.stack([(i1 % w + 0.5) * ow / w, (i1 // w + 0.5) * oh / h], 1)
            b = torch.stack([(i2 % w + 0.5) * ow / w, (i2 // w + 0.5) * oh / h], 1)
            return a.cpu().double().numpy(), b.cpu().double().numpy(), ow, oh

        from arev_eval_units import parse_plan
        for seq, d in parse_plan("usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3"):
            utag = f"{seq}__{lab}__d{d}"
            if os.path.exists(f"{cdir}/{utag}.done"):
                continue
            plist = [p for ds in (1, 2, 3) for p in ET.usegeo_pairs(ds, d)] if seq == "usegeo" else ET.flight_pairs(seq, d)
            pts = {}
            with open(f"{cdir}/{utag}.jsonl.tmp", "w") as fj, open(f"{cdir}/{utag}.acc.jsonl.tmp", "w") as fa:
                for p in plist:
                    pa, pb, ow, oh = match(p["p1"], p["p2"])
                    r = dict(seq=seq, d=d, i=p["i"], matcher=lab, pair=[os.path.basename(p["p1"]), os.path.basename(p["p2"])])
                    for kk in ("dataset", "along"):
                        if kk in p:
                            r[kk] = p[kk]
                    r["n_matches"] = int(len(pa))
                    H, mask = ET.magsac(pa, pb)
                    r["n_inliers"] = int(mask.sum()) if mask is not None else 0
                    r["H"] = H.tolist() if H is not None else None
                    r.update(ET.gates(H, r["n_inliers"], ow, oh))
                    pts[f"{p.get('dataset', 0)}_{p['i']}"] = np.concatenate([pa, pb], 1).astype(np.float32)
                    fj.write(json.dumps(r) + "\n")
                    if seq == "usegeo" and d <= 2 and p.get("along"):
                        A, B = p["A"], p["B"]
                        depthA = cache.get(p["dataset"], A[0])
                        H0, _ = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, SD.RANSAC_THR) if len(pa) >= 8 else (None, None)
                        outs = {"none": pb, "B": (E.prewarp_refine(E.gray(p["p1"]), E.gray(p["p2"]), pa, pb, H0, "lk")[0]
                                                  if H0 is not None else pb)}
                        for v, bb in outs.items():
                            s = E.score(pa, bb, ow, oh, A, B, depthA)
                            s.update(model=lab, variant=v, d=d, dataset=p["dataset"], frame_a=A[0], frame_b=B[0])
                            fa.write(json.dumps(s) + "\n")
            np.savez_compressed(f"{cdir}/{utag}.npz.tmp.npz", **pts)
            os.replace(f"{cdir}/{utag}.npz.tmp.npz", f"{cdir}/{utag}.npz")
            os.replace(f"{cdir}/{utag}.jsonl.tmp", f"{cdir}/{utag}.jsonl")
            os.replace(f"{cdir}/{utag}.acc.jsonl.tmp", f"{cdir}/{utag}.acc.jsonl")
            open(f"{cdir}/{utag}.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"PROGRESS kasar {utag}", flush=True)
        open(f"{out}/coarse.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if mode == "all":
        open(f"{out}/all.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
