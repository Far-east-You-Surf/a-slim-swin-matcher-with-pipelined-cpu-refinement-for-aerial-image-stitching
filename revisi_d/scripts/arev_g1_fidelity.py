"""Revisi A IJIES -- G1: kesetiaan engine statis top-K pembanding terhadap PyTorch asli (butir R1.3).

Engine e2e TensorRT FP16 yang dipakai untuk latensi/energi naskah (results/p2c/engines/<m>_e2e.engine, K = 3072 kandidat
kasar tetap; entri bantalan ber-mconf = 0 ikut tahap halus lalu dibuang) dijalankan pada 1418 pasangan UseGeo
sepanjang jalur d = 1–2 dengan praproses pembungkus asli (abu-abu [0,1] cv2.resize 480x640; JamMa: RGB ImageNet).
Pembanding rujukan = keluaran PyTorch asli TANPA batas kecocokan (G0, results/arev/g0/usegeo__<m>__d<d>.npz/.jsonl).
Per pasangan: jumlah kecocokan engine vs PyTorch, kecocokan bersama (titik citra 0 <= 4 px di bingkai 640, = sel kasar
sama; p2c_common.match_compare), selisih titik citra 1, dan metrik GT (usegeo_gt_eval.score_pair: galat LiDAR, pose).
Keluaran: revisi_d/results/arev/g1/<m>.jsonl + ringkasan <m>.json (ΔAUC, Δgalat, sebaran jumlah kecocokan nyata).
Pakai: python revisi_d/scripts/arev_g1_fidelity.py <m>
"""
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g1"


def main():
    m = sys.argv[1]
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    import cv2
    import numpy as np
    import tensorrt as trt
    import torch
    import usegeo_gt_eval as UG
    import evaltool as ET
    from p2c_common import match_compare
    from e13_refine import pose_auc
    os.makedirs(OUT, exist_ok=True)
    jl = f"{OUT}/{m}.jsonl"
    if not os.path.exists(jl + ".done"):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = rt.deserialize_cuda_engine(open(f"{RD}/results/p2c/engines/{m}_e2e.engine", "rb").read())
        ctx = eng.create_execution_context()
        bufs = {}
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            dt = {trt.float16: torch.float16, trt.float32: torch.float32}[eng.get_tensor_dtype(n)]
            bufs[n] = torch.empty(tuple(eng.get_tensor_shape(n)), dtype=dt, device="cuda")
            ctx.set_tensor_address(n, bufs[n].data_ptr())
        rgb = bufs["image0"].shape[1] == 3
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def prep(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            oh, ow = im.shape[:2]
            if rgb:
                r = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255, (640, 480))
                x = (torch.from_numpy(r).cuda().permute(2, 0, 1)[None] - mean) / std
            else:
                g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (640, 480))
                x = torch.from_numpy(g).cuda().float()[None, None] / 255
            return x, ow, oh

        cache = UG.DepthCache()
        with open(jl + ".tmp", "w") as fj:
            n = 0
            for d in (1, 2):
                ref = np.load(f"{RD}/results/arev/g0/usegeo__{m}__d{d}.npz")
                refj = {(r["dataset"], r["i"]): r for r in map(json.loads, open(f"{RD}/results/arev/g0/usegeo__{m}__d{d}.jsonl"))}
                for ds in (1, 2, 3):
                    for p in ET.usegeo_pairs(ds, d):
                        if not p["along"]:
                            continue
                        xa, ow, oh = prep(p["p1"])
                        xb, _, _ = prep(p["p2"])
                        bufs["image0"].copy_(xa.to(bufs["image0"].dtype))
                        bufs["image1"].copy_(xb.to(bufs["image1"].dtype))
                        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                        torch.cuda.synchronize()
                        mc = bufs["mconf"].float().cpu().numpy()
                        ok = mc > 0
                        s = np.array([ow / 640, oh / 480])
                        k0 = bufs["mkpts0_f"].float().cpu().numpy()[ok]
                        k1 = bufs["mkpts1_f"].float().cpu().numpy()[ok]
                        pts = ref[f"{ds}_{p['i']}"]
                        r0, r1 = pts[:, :2] / s, pts[:, 2:] / s
                        cmpd = match_compare(k0, k1, r0, r1)
                        a, b = (k0 * s).astype(np.float64), (k1 * s).astype(np.float64)
                        gt = UG.score_pair(a, b, ow, oh, p["A"], p["B"], cache.get(ds, p["A"][0]), True) if len(a) >= 8 else {}
                        rr = refj.get((ds, p["i"]), {})
                        rec = dict(dataset=ds, i=p["i"], d=d, n_engine=int(ok.sum()), n_torch=int(len(pts)),
                                   n_torch_over_K=bool(len(pts) > len(mc)), **{f"cmp_{k}": v for k, v in cmpd.items()},
                                   pose_err_engine=gt.get("pose_err"), med_err_engine=gt.get("med_err_inbounds"),
                                   pose_err_torch=rr.get("pose_err"), med_err_torch=rr.get("med_err_inbounds"))
                        fj.write(json.dumps(rec) + "\n")
                        n += 1
                        if n % 100 == 0:
                            print(f"PROGRESS {n}/1418 kesetiaan {m}", flush=True)
        os.replace(jl + ".tmp", jl)
        open(jl + ".done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    R = [json.loads(l) for l in open(jl)]
    pe = lambda k: [180.0 if r[k] is None else r[k] for r in R]  # noqa: E731
    ae, at = pose_auc(pe("pose_err_engine"), ths=(1, 2, 5)), pose_auc(pe("pose_err_torch"), ths=(1, 2, 5))
    me = [r["med_err_engine"] for r in R if r["med_err_engine"] is not None]
    mt = [r["med_err_torch"] for r in R if r["med_err_torch"] is not None]
    nt = np.array([r["n_torch"] for r in R])
    summ = dict(matcher=m, n_pairs=len(R), auc_engine=ae, auc_torch=at,
                med_err_engine=float(np.median(me)), med_err_torch=float(np.median(mt)),
                frac_pairs_torch_over_K=float(np.mean([r["n_torch_over_K"] for r in R])),
                n_torch_p50=float(np.percentile(nt, 50)), n_torch_p90=float(np.percentile(nt, 90)), n_torch_max=int(nt.max()),
                n_engine_p50=float(np.median([r["n_engine"] for r in R])),
                common_frac_med=float(np.median([r["cmp_n_common"] / max(r["n_engine"], 1) for r in R])),
                pt1_within_1px_med=float(np.nanmedian([r.get("cmp_frac_pt1_within_1px") or np.nan for r in R])))
    json.dump(summ, open(f"{OUT}/{m}.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
