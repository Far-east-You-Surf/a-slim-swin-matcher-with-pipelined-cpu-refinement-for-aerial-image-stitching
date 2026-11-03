"""Revisi A IJIES (G2) -- ekspor ONNX + engine TensorRT FP16 untuk varian sapuan, prosedur IDENTIK engine usulan.

Ekspor = gate_pilar3.export_onnx (masukan 480x640 FP16, opset 17, keluaran conf_matrix) seperti b8_s16u
(export_uniform_p16.py); build = cloud_e14/scripts/build_engines.build (STRONGLY_TYPED, workspace 8 GiB).
Bobot: checkpoint terpilih bila run sudah dievaluasi (results/arev/eval/<run>/selected.json), selain itu acak --
latensi dan energi tidak bergantung pada nilai bobot. GFLOPs (FlopCounterMode) dan jumlah parameter dicatat.
Pakai: python revisi_d/scripts/arev_export_variants.py [tag ...]   (bawaan: d8 d18 L0 L4 s32 r480)
Keluaran: revisi_d/results/arev/g2/{onnx,engines}/<tag>.{onnx,engine} + export_info.json
"""
import json
import os
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g2"
VARIANTS = {"d8": dict(blocks=8, layers=8, stride=16, res="768x1024", run="d8_s2"),
            "d18": dict(blocks=18, layers=8, stride=16, res="768x1024", run="d18_s2"),
            "L0": dict(blocks=8, layers=0, stride=16, res="768x1024", run="L0_s1"),
            "L4": dict(blocks=8, layers=4, stride=16, res="768x1024", run="L4_s1"),
            "s32": dict(blocks=8, layers=8, stride=32, res="768x1024", run="s32_s1"),
            "r480": dict(blocks=8, layers=8, stride=16, res="480x640", run="r480_s1")}


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    sys.path.insert(0, f"{RD}/cloud_e14/scripts")
    import torch
    from gate_pilar3 import export_onnx, measure_flops, count_params
    from build_engines import build
    import arev_models as AM
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    os.makedirs(f"{OUT}/engines", exist_ok=True)
    info_p = f"{OUT}/export_info.json"
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    onnx_only = "--onnx-only" in sys.argv
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or list(VARIANTS)
    for tag in tags:
        eng = f"{OUT}/engines/{tag}.engine"
        if onnx_only and os.path.exists(f"{OUT}/onnx/{tag}.onnx") and tag in info:
            print("lewat (onnx ada):", tag, flush=True)
            continue
        if not onnx_only and os.path.exists(eng) and tag in info:
            print("lewat:", tag, flush=True)
            continue
        v = VARIANTS[tag]
        net, grid = AM.build_net(dict(v, init="pred"))
        sel = f"{RD}/results/arev/eval/{v['run']}/selected.json"
        wsrc = "acak"
        if os.path.exists(sel):
            sd = torch.load(json.load(open(sel))["ckpt"], map_location="cpu")
            net.load_state_dict(sd)
            wsrc = json.load(open(sel))["ckpt"]
        net = net.cuda().half().eval()
        onnx_p = f"{OUT}/onnx/{tag}.onnx"
        export_onnx(net, onnx_p)
        rec = dict(v, grid=list(grid), weights=wsrc, params=count_params(net), gflops=measure_flops(net))
        if onnx_only:
            info[tag] = dict(info.get(tag, {}), **rec)
            json.dump(info, open(info_p + ".tmp", "w"), indent=1, default=str)
            os.replace(info_p + ".tmp", info_p)
            print(f"PROGRESS onnx {tag} kisi {grid} GFLOPs {rec['gflops']}", flush=True)
            del net
            torch.cuda.empty_cache()
            continue
        t0 = time.time()
        b = build(onnx_p, eng + ".tmp", f"{OUT}/engines/{tag}.build.log")
        os.replace(eng + ".tmp", eng)
        rec.update(build_s=round(time.time() - t0, 1), build=b if isinstance(b, dict) else str(b))
        info[tag] = rec
        json.dump(info, open(info_p + ".tmp", "w"), indent=1, default=str)
        os.replace(info_p + ".tmp", info_p)
        print(f"PROGRESS {len(info)}/{len(VARIANTS)} {tag} kisi {grid} GFLOPs {rec['gflops']}", flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
