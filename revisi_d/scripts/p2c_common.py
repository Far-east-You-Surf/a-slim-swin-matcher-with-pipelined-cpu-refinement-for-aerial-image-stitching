"""P2c -- utilitas bersama: latensi TERKOMPILASI pembanding (ONNX opset 17 FP16 -> TensorRT
STRONGLY_TYPED, workspace 8 GiB; builder = cloud_e14/scripts/build_engines.build, identik engine kami).

Isi:
  load_pair_gray()        pasangan nyata G Uji (collect_pilar3_multipair.PAIR_SETS['g_uji_out_of_domain'][0])
                          -> 2 tensor abu-abu [0,1] (1,1,480,640), praproses = pembungkus stitch_demo/tp_matchers
  static_coarse_match()   pengganti statis get_coarse_match (mutual-NN + ambang + border) dgn top-K tetap
  export_build()          ekspor ONNX FP16 + build engine + simpan info
  run_engine()            jalankan engine dgn masukan NYATA, kembalikan semua keluaran (numpy)
  record()                tulis/menggabung revisi_d/results/p2c/compile.json
Satu metode per proses (bentrok paket `src`).
"""
import json
import os
import shutil
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
for p in (ROOT, f"{RD}/scripts", f"{RD}/cloud_e14/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = f"{RD}/results/p2c"
ENG = f"{OUT}/engines"
ONNX_TP = f"{RD}/cloud_e14/onnx_tp"
os.makedirs(ENG, exist_ok=True)
os.makedirs(ONNX_TP, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

H, W = 480, 640
GUJI_DIR = AERIAL_DATA_UJICOBA + "/G Uji"
PAIR = (f"{GUJI_DIR}/DSC00003.JPG", f"{GUJI_DIR}/DSC00004.JPG")   # = PAIR_SETS['g_uji_out_of_domain'][0]


def imread(p):
    return cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def load_pair_gray(device="cuda"):
    g = []
    for p in PAIR:
        x = cv2.cvtColor(imread(p), cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (W, H))
        g.append(torch.from_numpy(x).float()[None, None].to(device) / 255.0)
    return g


def load_pair_rgb_imagenet(device="cuda"):
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    t = []
    for p in PAIR:
        x = cv2.cvtColor(imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = cv2.resize(x, (W, H))
        x = (x - mean) / std
        t.append(torch.from_numpy(x).permute(2, 0, 1)[None].to(device))
    return t


# ----------------------------------------------------------------------------- statis
def static_coarse_match(conf, thr, border_rm, hw0_c, hw1_c, K, scale):
    """Setara get_coarse_match LoFTR (inferensi, bs=1, tanpa mask) TETAPI berukuran tetap.

    Asli: mask = (conf>thr) & border & (conf==max baris) & (conf==max kolom); torch.where -> M dinamis.
    Statis: per baris i ambil v_i,j_i = max_j conf (mutual-NN => paling banyak satu j per i);
    valid_i = v_i>thr & border0[i] & border1[j_i] & (v_i == max kolom j_i). Lalu top-K baris menurut
    v_i*valid_i (TopK, K konstan). Entri tak valid tetap dihitung (mconf=0) -> bentuk statis.
    Mengembalikan b_ids, i_ids, j_ids (int64, K), mconf (K; 0 = tak valid), mkpts0_c, mkpts1_c (K,2)."""
    h0, w0 = hw0_c
    h1, w1 = hw1_c
    dev = conf.device
    v, j = conf.max(dim=2)                       # (1,L)
    colmax = conf.max(dim=1)[0]                  # (1,S)
    mutual = v == torch.gather(colmax, 1, j)

    def bmask(h, w):                              # konstanta (numpy) -> tidak menambah op dinamis
        m = np.ones((h, w), bool)
        if border_rm > 0:
            m[:border_rm] = False
            m[-border_rm:] = False
            m[:, :border_rm] = False
            m[:, -border_rm:] = False
        return torch.from_numpy(m.flatten()).to(dev)

    b0, b1 = bmask(h0, w0), bmask(h1, w1)
    valid = (v > thr) & mutual & b0[None] & torch.gather(b1[None].expand(conf.shape[0], -1), 1, j)
    score = torch.where(valid, v, torch.zeros_like(v))
    mconf, i_ids = torch.topk(score, K, dim=1)   # (1,K)
    j_ids = torch.gather(j, 1, i_ids)
    i_ids, j_ids, mconf = i_ids[0], j_ids[0], mconf[0]
    b_ids = torch.zeros_like(i_ids)
    mk0 = torch.stack([i_ids % w0, torch.div(i_ids, w0, rounding_mode="floor")], 1) * scale
    mk1 = torch.stack([j_ids % w1, torch.div(j_ids, w1, rounding_mode="floor")], 1) * scale
    return dict(b_ids=b_ids, i_ids=i_ids, j_ids=j_ids, mconf=mconf, m_bids=b_ids,
                mkpts0_c=mk0.float(), mkpts1_c=mk1.float())   # koordinat FP32 (FP16 di ~640 px = langkah 0,5 px)


# ----------------------------------------------------------------------------- ekspor/build
def export_onnx(net, inputs, path, in_names, out_names):
    t = time.time()
    with torch.no_grad():
        torch.onnx.export(net, tuple(inputs), path, opset_version=17, dynamo=False,
                          input_names=in_names, output_names=out_names)
    return dict(ok=True, s=round(time.time() - t, 1), mb=round(os.path.getsize(path) / 2**20, 2),
                path=path)


def onnx_ops(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    h = {}
    for n in m.graph.node:
        h[n.op_type] = h.get(n.op_type, 0) + 1
    return dict(n_nodes=len(m.graph.node), ops=dict(sorted(h.items(), key=lambda kv: -kv[1])[:25]))


def build_engine(onnx_path, eng_path):
    from build_engines import build
    log = eng_path.replace(".engine", ".build.log")
    info = build(onnx_path, eng_path, log)
    try:
        info["log_tail"] = open(log).read().strip().splitlines()[-8:]
    except Exception:
        pass
    return info


def run_engine(eng_path, feeds):
    """feeds: dict nama->np.ndarray. Mengembalikan dict nama->np.ndarray keluaran + device_memory."""
    import tensorrt as trt
    from cuda.bindings import runtime as cr
    from trt_common import ck
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(open(eng_path, "rb").read())
    ctx = eng.create_execution_context()
    stream = ck(cr.cudaStreamCreate())[0]
    ptrs, outs = [], {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shp = tuple(eng.get_tensor_shape(n))
        dt = trt.nptype(eng.get_tensor_dtype(n))
        nb = int(np.prod(shp)) * np.dtype(dt).itemsize
        p = ck(cr.cudaMalloc(max(nb, 1)))[0]
        ptrs.append(p)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            a = np.ascontiguousarray(feeds[n].astype(dt).reshape(shp))
            ck(cr.cudaMemcpy(p, a.ctypes.data, nb, cr.cudaMemcpyKind.cudaMemcpyHostToDevice))
        else:
            outs[n] = (p, np.empty(shp, dt), nb)
        ctx.set_tensor_address(n, int(p))
    ctx.execute_async_v3(int(stream))
    ck(cr.cudaStreamSynchronize(stream))
    res = {}
    for n, (p, host, nb) in outs.items():
        ck(cr.cudaMemcpy(host.ctypes.data, p, nb, cr.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        res[n] = host
    dm = int(eng.device_memory_size_v2)
    for p in ptrs:
        cr.cudaFree(p)
    cr.cudaStreamDestroy(stream)
    return res, dm


def copy_onnx(onnx_path, method, level):
    """Salin ONNX (berkas nyata) ke cloud_e14/onnx_tp/<method>_<level>.onnx (+ .data bila ada)."""
    dst = f"{ONNX_TP}/{method}_{level}.onnx"
    shutil.copyfile(onnx_path, dst)
    if os.path.exists(onnx_path + ".data"):
        shutil.copyfile(onnx_path + ".data", dst + ".data")
    return dst


def match_compare(k0e, k1e, k0r, k1r, tol=4.0):
    """Cocokkan kecocokan engine ke rujukan via mkpts0 terdekat (<= tol px = setengah sel kasar 8 px, jadi sel kasar yang sama), lalu selisih mkpts0 dan mkpts1."""
    if len(k0e) == 0 or len(k0r) == 0:
        return dict(n_engine=int(len(k0e)), n_ref=int(len(k0r)), n_common=0)
    d = np.linalg.norm(k0e[:, None, :] - k0r[None, :, :], axis=2)
    nn_ = d.argmin(1)
    ok = d[np.arange(len(k0e)), nn_] <= tol
    d0 = d[np.arange(len(k0e)), nn_][ok]
    d1 = np.linalg.norm(k1e[ok] - k1r[nn_[ok]], axis=1)
    return dict(n_engine=int(len(k0e)), n_ref=int(len(k0r)), n_common=int(ok.sum()),
                mean_pt0_diff_px=float(d0.mean()) if ok.any() else None,
                mean_pt1_diff_px=float(d1.mean()) if ok.any() else None,
                median_pt1_diff_px=float(np.median(d1)) if ok.any() else None,
                frac_pt1_within_1px=float((d1 <= 1).mean()) if ok.any() else None,
                frac_pt0_and_pt1_within_1px=float(((d1 <= 1) & (d0 <= 1)).mean()) if ok.any() else None,
                max_pt1_diff_px=float(d1.max()) if ok.any() else None)


def record(method, level, rec):
    p = f"{OUT}/compile.json"
    allr = json.load(open(p)) if os.path.exists(p) else {}
    allr.setdefault(method, {})[level] = rec
    json.dump(allr, open(p, "w"), indent=1, default=str)


def short(e):
    return f"{type(e).__name__}: {str(e)}"[:1500]


# ----------------------------------------------------------------------------- driver satu level
def do_level(method, level, net16, inputs16, in_names, out_names, ref32=None, ref_full=None, note=None,
             changes=None, extra=None):
    """net16: modul FP16 yang diekspor; inputs16: tensor nyata FP16.
    level 'coarse': out = conf_matrix; dibandingkan dgn net16 (PyTorch FP16) dan ref32 (tensor FP32).
    level 'e2e'   : out = (mkpts0_f, mkpts1_f, mconf) K tetap; valid = mconf>0. Dibandingkan dgn
                    net16 (graf sama di PyTorch) dan ref_full = (k0,k1) keluaran model ASLI FP32."""
    rec = dict(method=method, level=level, input="480x640", opset=17, precision="FP16 (model.half())",
               changes=changes, note=note)
    rec.update(extra or {})
    tag = f"{method}_{level}"
    onnx_p, eng_p = f"{ENG}/{tag}.onnx", f"{ENG}/{tag}.engine"
    try:
        with torch.no_grad():
            ref16 = net16(*inputs16)
        ref16 = [ref16] if torch.is_tensor(ref16) else list(ref16)
        rec["eager_fp16_ok"] = True
    except Exception as e:
        rec.update(eager_fp16_ok=False, error=short(e)); record(method, level, rec); print(rec); return rec
    try:
        rec["onnx"] = export_onnx(net16, inputs16, onnx_p, in_names, out_names)
        rec["onnx"].update(onnx_ops(onnx_p))
    except Exception as e:
        import traceback
        rec.update(onnx_ok=False, error=short(e), tb=traceback.format_exc()[-2500:])
        record(method, level, rec); print(rec["error"][:800]); return rec
    rec["onnx_ok"] = True
    try:
        rec["trt"] = build_engine(onnx_p, eng_p)
    except Exception as e:
        rec["trt"] = dict(ok=False, error=short(e))
    rec["trt_ok"] = bool(rec["trt"].get("ok"))
    if rec["trt_ok"]:
        rec["onnx_tp_copy"] = copy_onnx(onnx_p, method, level)
        feeds = {n: t.detach().cpu().numpy() for n, t in zip(in_names, inputs16)}
        outs, dm = run_engine(eng_p, feeds)
        rec["device_memory_size_v2_bytes"] = dm
        rec["engine_file_bytes"] = os.path.getsize(eng_p)
        o = [outs[n].astype(np.float32) for n in out_names]
        r16 = [t.float().cpu().numpy() for t in ref16]
        v = {}
        if level == "coarse":
            v["max_abs_vs_torch_fp16"] = float(np.abs(o[0] - r16[0]).max())
            v["mean_abs_vs_torch_fp16"] = float(np.abs(o[0] - r16[0]).mean())
            if ref32 is not None:
                r = ref32.float().cpu().numpy()
                v["max_abs_vs_torch_fp32"] = float(np.abs(o[0] - r).max())
            v["ref_max"] = float(r16[0].max())
            v["nan_engine"] = int(np.isnan(o[0]).sum())
        else:
            k0e, k1e, ce = o[0].reshape(-1, 2), o[1].reshape(-1, 2), o[2].reshape(-1)
            k0t, k1t, ct = r16[0].reshape(-1, 2), r16[1].reshape(-1, 2), r16[2].reshape(-1)
            ve, vt = ce > 0, ct > 0
            v["K"] = int(len(ce))
            v["n_valid_engine"] = int(ve.sum())
            v["n_valid_torch_fp16_static"] = int(vt.sum())
            v["nan_engine"] = int(np.isnan(o[0]).sum() + np.isnan(o[1]).sum())
            v["vs_torch_fp16_static"] = match_compare(k0e[ve], k1e[ve], k0t[vt], k1t[vt])
            if ref_full is not None:
                v["n_ref_full_fp32"] = int(len(ref_full[0]))
                v["vs_torch_fp32_original"] = match_compare(k0e[ve], k1e[ve], ref_full[0], ref_full[1])
        rec["validation"] = v
    record(method, level, rec)
    print(json.dumps({k: rec.get(k) for k in ("method", "level", "onnx_ok", "trt_ok", "validation")}, default=str)[:3000])
    if rec["trt_ok"] is False:
        print("TRT:", json.dumps(rec["trt"], default=str)[:2000])
    return rec


def static_vs_original(net32_e2e, inputs32, ref_full):
    """Cek penulisan-ulang statis di FP32 PyTorch: keluaran statis vs model asli (tanpa TRT)."""
    with torch.no_grad():
        k0, k1, c = [t.float().cpu().numpy() for t in net32_e2e(*inputs32)]
    k0, k1, c = k0.reshape(-1, 2), k1.reshape(-1, 2), c.reshape(-1)
    v = c > 0
    return dict(n_valid_static_fp32=int(v.sum()), **{"cmp": match_compare(k0[v], k1[v], ref_full[0], ref_full[1])})


class _FShim:
    """Pengganti nama `F` di modul pihak ketiga: F.unfold dgn stride/kernel/padding tensor-0d (hasil
    shape saat tracing) -> int konstan (bentuk statis 480x640). Selain itu diteruskan ke torch.nn.functional."""
    def __getattr__(self, k):
        import torch.nn.functional as F
        return getattr(F, k)

    @staticmethod
    def _i(x):
        if torch.is_tensor(x):
            return int(x)
        if isinstance(x, (tuple, list)):
            return type(x)(int(v) if torch.is_tensor(v) else v for v in x)
        return x

    def unfold(self, x, kernel_size, dilation=1, padding=0, stride=1):
        import torch.nn.functional as F
        return F.unfold(x, self._i(kernel_size), self._i(dilation), self._i(padding), self._i(stride))


def shim_F(module):
    module.F = _FShim()


def wrap_int_hw(submodule, keys=("hw0_i", "hw1_i", "hw0_c", "hw1_c", "hw0_f", "hw1_f")):
    """Bungkus submodule.forward(..., data): ukuran hw* di `data` (tensor-0d saat tracing) -> int konstan."""
    orig = submodule.forward

    def f(*a, **kw):
        data = a[-1] if a and isinstance(a[-1], dict) else kw.get("data")
        for k in keys:
            if data is not None and k in data:
                data[k] = tuple(int(v) for v in data[k])
        return orig(*a, **kw)
    submodule.forward = f
