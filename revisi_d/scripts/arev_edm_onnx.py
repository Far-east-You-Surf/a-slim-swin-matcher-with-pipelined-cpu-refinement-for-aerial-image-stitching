"""Revisi A IJIES (G2/G3) -- ekspor EDM ke ONNX statis FP16 (mode DEPLOY repo) lalu engine TensorRT, protokol sama.

Konfigurasi = arev_tp_extra._build_edm (outdoor, 480x640, TOPK 1680, NPE [832, 832, 480, 640]) + EDM.DEPLOY = True
(jalur ekspor resmi repo deploy/export_onnx.py: masukan [1, 2, H, W] abu-abu, keluaran tetap TOPK). Model dicast FP16
seperti engine pembanding lain (STRONGLY_TYPED, build_engines.build).
PERBAIKAN 2 Nov 2026: jalur ds_opt EDM (exp(S) lalu normalisasi L1) meluap pada FP16 (exp(S/T) > 65504 -> mconf = inf,
seleksi TOPK tak bermakna; engine 31 Okt rusak). Ekspor kini memakai softmax ganda bawaan (ds_opt = False), yang identik
secara matematis (normalisasi L1 dari exp(S) = softmax(S)) dan stabil pada FP16 karena dikurangi maksimum.
Kesetiaan (edm_fidelity.json): engine vs PyTorch FP32 (konfigurasi resmi, ds_opt bawaan) pada N_FID pasangan UseGeo
d = 1 -> mconf hingga, sel kasar sama (%), dan titik akhir citra 1 dalam 1 px (%) setelah pascaproses resmi.
Pakai: python revisi_d/scripts/arev_edm_onnx.py
Keluaran: revisi_d/results/arev/g2/{onnx/edm_e2e.onnx, engines/edm_e2e.engine, edm_export.json}
"""
import json
import os
import sys
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g2"
N_FID = 100


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    sys.path.insert(0, f"{RD}/cloud_e14/scripts")
    import cv2
    import numpy as np
    import torch
    import arev_tp_extra as X
    import tp_matchers as TM
    X.install()
    TM.REPOS["edm"] = X.EDM_ROOT
    TM._enter_repo("edm")
    X._lightning_shim()
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config
    c = get_cfg_defaults()
    c.merge_from_file(os.path.join(X.EDM_ROOT, "configs", "edm", "outdoor", "edm_base.py"))
    h, w = 480, 640
    c.EDM.COARSE.TOPK = int(h / 8 * w / 8 * 0.35)
    c.EDM.TRAIN_RES_H, c.EDM.TRAIN_RES_W, c.EDM.TEST_RES_H, c.EDM.TEST_RES_W = 832, 832, h, w
    c.EDM.NECK.NPE = [832, 832, h, w]
    c.EDM.DEPLOY = True
    net = EDM(config=lower_config(c)["edm"])
    sd = torch.load(X.EDM_W, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("state_dict", sd))
    net = net.eval().cuda()
    ref_net = EDM(config=lower_config(c)["edm"])          # rujukan FP32 resmi (ds_opt bawaan) untuk uji kesetiaan
    ref_net.load_state_dict(sd.get("state_dict", sd))
    ref_net = ref_net.eval().cuda()
    net.coarse_matching.ds_opt = False                     # softmax ganda bawaan: sama secara matematis, stabil FP16
    import usegeo_gt_eval as UG
    rows = UG.load_poses(1)
    g = []
    for k in (0, 1):
        im = cv2.imread(UG.img_path(1, rows[k][0]), cv2.IMREAD_GRAYSCALE)
        g.append(torch.from_numpy(cv2.resize(im, (w, h))).float()[None, None] / 255)
    x = torch.cat(g, 1).cuda()
    with torch.no_grad():
        ref = net(x)
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    os.makedirs(f"{OUT}/engines", exist_ok=True)
    onnx_p = f"{OUT}/onnx/edm_e2e.onnx"
    neth = net.half()
    with torch.no_grad():
        torch.onnx.export(neth, (x.half(),), onnx_p, input_names=["input"], output_names=["output"], opset_version=17,
                          dynamo=False)
    from build_engines import build
    eng = f"{OUT}/engines/edm_e2e.engine"
    info = build(onnx_p, eng + ".tmp", f"{OUT}/engines/edm_e2e.build.log")
    os.replace(eng + ".tmp", eng)
    rep = dict(onnx=onnx_p, engine=eng, topk=int(c.EDM.COARSE.TOPK), ref_shape=list(ref.shape) if torch.is_tensor(ref) else str(type(ref)),
               build=str(info)[:300], ds_opt=False)
    json.dump(rep, open(f"{OUT}/edm_export.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)
    # ---- kesetiaan engine vs PyTorch FP32 resmi
    import arev_g2_bench as GB
    E = GB.EDMEngine(eng)
    same, near, finite, n_e, n_r = [], [], [], [], []
    for k in range(min(N_FID, len(rows) - 1)):
        g = []
        for kk in (k, k + 1):
            im = cv2.imread(UG.img_path(1, rows[kk][0]), cv2.IMREAD_GRAYSCALE)
            g.append(torch.from_numpy(cv2.resize(im, (w, h))).float()[None, None] / 255)
        xf = torch.cat(g, 1).cuda()
        with torch.no_grad():
            r = ref_net(xf)
        E.inp.copy_(xf.half())
        E.c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        o = E.out.float()
        finite.append(bool(torch.isfinite(o).all()))
        ce = {tuple(v) for v in o[:, 0:4].round().int().tolist()}
        cr = {tuple(v) for v in r.float()[:, 0:4].round().int().tolist()}
        same.append(len(ce & cr) / max(len(cr), 1))
        e0, e1 = GB.edm_post(o)
        r0, r1 = GB.edm_post(r)
        n_e.append(len(e0))
        n_r.append(len(r0))
        if len(e0) and len(r0):
            d0 = torch.cdist(e0, r0)                        # punca citra 0 identik bila sel sama -> pasangan terdekat
            j = d0.argmin(1)
            ok = d0.gather(1, j[:, None])[:, 0] < 0.5
            near.append(float(((e1[ok] - r1[j[ok]]).norm(dim=1) <= 1.0).float().mean()) if ok.any() else 0.0)
    fid = dict(n_pairs=len(same), all_finite=all(finite), coarse_cells_same_med=float(np.median(same)),
               coarse_cells_same_min=float(np.min(same)), pts1_within_1px_med=float(np.median(near)) if near else None,
               pts1_within_1px_min=float(np.min(near)) if near else None, n_matches_engine_med=float(np.median(n_e)),
               n_matches_torch_med=float(np.median(n_r)))
    json.dump(fid, open(f"{OUT}/edm_fidelity.json.tmp", "w"), indent=1)
    os.replace(f"{OUT}/edm_fidelity.json.tmp", f"{OUT}/edm_fidelity.json")
    print(json.dumps(fid), flush=True)


if __name__ == "__main__":
    main()
