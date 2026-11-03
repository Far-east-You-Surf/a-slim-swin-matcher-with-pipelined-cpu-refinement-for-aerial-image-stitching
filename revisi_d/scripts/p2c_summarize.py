"""P2c -- gabungkan compile.json (ekspor/build/validasi) + latency/*.json (3 proses, median) -> compile.json
(field 'latency') dan compile.md (tabel Indonesia).
Pakai: python revisi_d/scripts/p2c_summarize.py
"""
import json, os, glob
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
OUT = f"{ROOT}/revisi_d/results/p2c"
ORDER = ["efficientloftr", "topicfm_fast", "loftr", "jamma", "matchformer", "aspanformer", "lgfctr"]
OURS = {"proposal /16 8-blok (b8_s16)": 10.84, "baseline": 13.28}


def f(x, n=2):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def galat(r):
    v = r.get("validation")
    if not v:
        return "—"
    if r["level"].startswith("coarse"):
        s = f"conf maks |Δ| {v['max_abs_vs_torch_fp16']:.3g} vs PyTorch FP16"
        if "max_abs_vs_torch_fp32" in v:
            s += f"; {v['max_abs_vs_torch_fp32']:.3g} vs FP32"
        if v.get("nan_engine"):
            s += f"; **NaN {v['nan_engine']}**"
        return s
    a = v.get("vs_torch_fp32_original", {})
    s = (f"{v['n_valid_engine']} kecocokan engine vs {v.get('n_ref_full_fp32')} PyTorch-FP32 asli; "
         f"{a.get('n_common', 0)} sel sama")
    if a.get("mean_pt1_diff_px") is not None:
        s += (f", Δpt1 rerata {a['mean_pt1_diff_px']:.3f} px (median {a['median_pt1_diff_px']:.3f}), "
              f"{100 * a['frac_pt0_and_pt1_within_1px']:.1f}% ≤1 px")
    if v.get("nan_engine"):
        s += f"; **NaN {v['nan_engine']}**"
    return s


def main():
    d = json.load(open(f"{OUT}/compile.json"))
    lat = {}
    for p in glob.glob(f"{OUT}/latency/*.json"):
        j = json.load(open(p))
        lat[os.path.basename(p)[:-5]] = j
    rows = []
    for m in ORDER + sorted(k for k in d if k not in ORDER):
        if m not in d:
            continue
        for lvl in sorted(d[m], key=lambda s: (not s.startswith("coarse"), s)):
            r = d[m][lvl]
            L = lat.get(f"{m}_{lvl}")
            if L:
                r["latency"] = {k: L[k] for k in ("latency_ms_median", "latency_ms_reps", "per_iter_median_ms_median",
                                                  "device_memory_size_v2_bytes", "engine_file_bytes", "protocol")}
                r["latency"]["control_b8_s16_ms"] = L.get("control_b8_s16_ms")
            trt = r.get("trt") or {}
            rows.append(dict(
                m=m, lvl=lvl, onnx="ya" if r.get("onnx_ok") else ("tidak" if r.get("onnx_ok") is False else "—"),
                trt="ya" if r.get("trt_ok") else ("tidak" if r.get("trt_ok") is False else "—"),
                chg=r.get("changes") or "—", gal=galat(r),
                lat=f(L["latency_ms_median"]) if L else "—",
                mem=f(L["device_memory_size_v2_bytes"] / 2**20, 1) if L else (
                    f(r["device_memory_size_v2_bytes"] / 2**20, 1) if r.get("device_memory_size_v2_bytes") else "—"),
                emb=f(r["engine_file_bytes"] / 2**20, 1) if r.get("engine_file_bytes") else "—",
                note=(r.get("note") or "") + ((" " + r["error"][:300]) if r.get("error") else "") +
                     (f" build {trt.get('build_s')} s" if trt.get("build_s") is not None else "")))
    json.dump(d, open(f"{OUT}/compile.json", "w"), indent=1, default=str)
    H = ("| metode | level | ekspor ONNX ok? | build TRT ok? | perubahan agar statis | galat vs PyTorch | "
         "latensi (ms) | memori engine (MB) | berkas engine (MB) | catatan/penyebab gagal |")
    lines = [H, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['m']} | {r['lvl']} | {r['onnx']} | {r['trt']} | {r['chg']} | {r['gal']} | {r['lat']} | "
                     f"{r['mem']} | {r['emb']} | {r['note'].strip() or '—'} |")
    open(f"{OUT}/compile_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
