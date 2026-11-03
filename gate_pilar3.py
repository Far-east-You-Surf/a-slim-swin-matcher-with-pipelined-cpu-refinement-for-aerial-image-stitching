"""
Gerbang kelayakan ekspor untuk kandidat reduksi komputasi Pilar #3 (B1-B4).

Tujuan: menjawab SEBELUM training apa pun -- apakah perubahan struktural yang
kita niatkan benar-benar SAMPAI ke engine TensorRT, atau hilang di jalur ekspor
(pelajaran log §20: flash attention hilang krn legacy TorchScript exporter
mendekomposisi SDPA, jadi ORIGINAL & FLASH menghasilkan engine identik).

Kandidat:
  b1_sparse24     -- structured sparsity 2:4 (ModelOpt magnitude) + TRT SPARSE_WEIGHTS
  b2_depth12      -- depth pruning Swin stage 3: 18 -> 12 blok
  b2_depth8       -- depth pruning Swin stage 3: 18 -> 8 blok
  b3_stage3_half  -- reduksi jumlah token stage 3 (jalankan stage 3 di 1/2 resolusi)
  b4_512x640      -- resize internal 768x1024 -> 512x640
  b4_384x512      -- resize internal 768x1024 -> 384x512

Metodologi ekspor & benchmark identik dgn quant_sweep.py / flash_level3_5070ti.py
supaya angkanya langsung comparable dgn tabel §18/§23.

Pakai: python gate_pilar3.py <varian> [results_dir]
"""

import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import utils as onnx_utils

from model import AerialStitchingModel
from quant_sweep import benchmark_engine_with_vram, RES_HW, DEVICE

CKPT = "aerial_stitching_fp16_best.pth"
DEFAULT_OUT = "gate_pilar3_results"


# --------------------------------------------------------------------------
# Pembangun varian
# --------------------------------------------------------------------------

def load_baseline():
    model = AerialStitchingModel(d_model=128, nhead=8, num_layers=8).to(DEVICE).half().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    return model


def variant_baseline(model):
    return model, {}


def variant_sparse24(model):
    """B1: structured sparsity 2:4 lewat ModelOpt magnitude pruning.

    Tidak mengubah struktur graf sama sekali -- hanya menolkan bobot dgn pola 2:4.
    Pertanyaan gerbangnya: apakah TensorRT (flag SPARSE_WEIGHTS) benar-benar
    memanfaatkan pola itu di Blackwell, atau cuma jadi bobot nol biasa.
    """
    import modelopt.torch.sparsity as mts

    out = mts.sparsify(model, "sparse_magnitude", config={})
    model = out[0] if isinstance(out, tuple) else out
    model = mts.export(model)  # lipat mask ke bobot

    # ukur rasio nol nyata + kepatuhan pola 2:4
    total, zeros, checked_2of4, ok_2of4 = 0, 0, 0, 0
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            w = mod.weight.detach().float().reshape(mod.weight.shape[0], -1)
            total += w.numel()
            zeros += (w == 0).sum().item()
            if w.shape[1] % 4 == 0:
                g = w.reshape(-1, 4)
                nz = (g != 0).sum(dim=1)
                checked_2of4 += g.shape[0]
                ok_2of4 += (nz <= 2).sum().item()
    info = dict(
        weight_zero_ratio=zeros / total if total else None,
        pattern_2of4_compliance=ok_2of4 / checked_2of4 if checked_2of4 else None,
    )
    return model, info


def _prune_stage3(model, keep_blocks):
    """B2: buang blok dari Swin stage 3 (features[5], 18 blok).

    Blok Swin berselang-seling shifted/non-shifted (genap = tanpa shift, ganjil =
    shift). Pemangkasan dilakukan per-PASANGAN supaya pola selang-seling itu utuh,
    karena itulah yg bikin window attention Swin punya receptive field global.
    """
    stage3 = model.backbone[5]
    n_pairs_avail = len(stage3) // 2
    n_pairs_keep = keep_blocks // 2
    assert keep_blocks % 2 == 0, "keep_blocks harus genap (jaga pasangan shift)"
    pair_idx = np.linspace(0, n_pairs_avail - 1, n_pairs_keep).round().astype(int)
    block_idx = []
    for p in pair_idx:
        block_idx += [2 * p, 2 * p + 1]
    model.backbone[5] = nn.Sequential(*[stage3[i] for i in block_idx])
    return model, dict(stage3_blocks_before=len(stage3), stage3_blocks_after=keep_blocks,
                        kept_block_indices=[int(i) for i in block_idx])


class Stage3HalfRes(nn.Module):
    """B3: jalankan stage 3 pada 1/2 resolusi spasial (jumlah token turun 4x),
    lalu kembalikan ke resolusi asli sebelum stage berikutnya.

    Ini pengganti ToMe/token-merging yang KOMPATIBEL dengan Swin: window attention
    Swin butuh token tersusun di grid H x W, jadi penggabungan token arbitrer
    (bipartite soft matching ala ToMe) akan merusak partisi window-nya. Reduksi
    spasial seragam mempertahankan grid, tetap memangkas jumlah token, dan
    menghasilkan shape statis (aman utk TensorRT).
    """

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, x):  # x: [B, H, W, C]
        B, H, W, C = x.shape
        xt = x.permute(0, 3, 1, 2)
        xd = F.avg_pool2d(xt, kernel_size=2, stride=2)
        y = self.stage(xd.permute(0, 2, 3, 1))
        yt = y.permute(0, 3, 1, 2)
        yu = F.interpolate(yt, size=(H, W), mode="bilinear", align_corners=False)
        return yu.permute(0, 2, 3, 1)


def variant_stage3_half(model):
    model.backbone[5] = Stage3HalfRes(model.backbone[5])
    return model, dict(stage3_token_reduction="4x (resolusi spasial 1/2)")


def variant_resize(model, size):
    model.target_size = size
    return model, dict(target_size=list(size))


VARIANTS = {
    "baseline":       variant_baseline,
    "b1_sparse24":    variant_sparse24,
    "b2_depth12":     lambda m: _prune_stage3(m, 12),
    "b2_depth8":      lambda m: _prune_stage3(m, 8),
    "b3_stage3_half": variant_stage3_half,
    "b4_512x640":     lambda m: variant_resize(m, (512, 640)),
    "b4_384x512":     lambda m: variant_resize(m, (384, 512)),
}

SPARSE_VARIANTS = {"b1_sparse24"}


# --------------------------------------------------------------------------
# Ekspor, inspeksi graf, build engine
# --------------------------------------------------------------------------

def export_onnx(model, onnx_path):
    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(),
             torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=17,
                          input_names=["img1", "img2"], output_names=["conf_matrix"])


def inspect_onnx(onnx_path):
    """Cek apakah perubahan struktural kelihatan di graf, dan apakah shape-nya statis."""
    import onnx
    m = onnx.load(onnx_path)
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    def dims_of(vi):
        return [(d.dim_value if d.HasField("dim_value") else d.dim_param)
                for d in vi.type.tensor_type.shape.dim]

    dynamic = []
    for vi in list(m.graph.input) + list(m.graph.output):
        for d in dims_of(vi):
            if isinstance(d, str) or d == 0:
                dynamic.append(vi.name)
                break

    init_elems = sum(int(np.prod(i.dims)) for i in m.graph.initializer if i.dims)
    return dict(
        total_nodes=len(m.graph.node),
        op_histogram=dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        n_softmax=ops.get("Softmax", 0),
        n_matmul=ops.get("MatMul", 0),
        n_gemm=ops.get("Gemm", 0),
        n_conv=ops.get("Conv", 0),
        initializer_elements=init_elems,
        dynamic_shape_tensors=dynamic,
        static_shapes=len(dynamic) == 0,
        onnx_file_mb=os.path.getsize(onnx_path) / (1024 ** 2),
    )


def build_engine_ex(onnx_path, engine_path, sparse=False, workspace_gb=8):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return False, errs
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))
    if sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False, ["build_serialized_network mengembalikan None"]
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return True, []


def engine_device_memory_mb(engine_path):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    return engine.device_memory_size_v2 / (1024 ** 2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model):
    """FLOPs analitik via FlopCounterMode -- metodologi sama dgn log §8."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        img1 = torch.randn(1, 3, *RES_HW, device=DEVICE).half()
        img2 = torch.randn(1, 3, *RES_HW, device=DEVICE).half()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(img1, img2)
        return counter.get_total_flops() / 1e9
    except Exception as e:
        return f"gagal: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------

def main():
    variant = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    if variant not in VARIANTS:
        print(f"varian tidak dikenal: {variant}. Pilihan: {list(VARIANTS)}")
        sys.exit(2)

    onnx_path = f"{out_dir}/{variant}.onnx"
    engine_path = f"{out_dir}/{variant}.engine"
    json_path = f"{out_dir}/{variant}.json"

    result = dict(variant=variant, res_hw=list(RES_HW), gpu=torch.cuda.get_device_name(0))
    print(f"[GATE] {variant}: bangun varian model...")

    model = load_baseline()
    try:
        model, vinfo = VARIANTS[variant](model)
        result["variant_info"] = vinfo
    except Exception as e:
        result["error"] = f"build_variant: {type(e).__name__}: {e}"
        json.dump(result, open(json_path, "w"), indent=2)
        print(f"[GAGAL] bangun varian: {e}")
        sys.exit(1)

    model = model.to(DEVICE).half().eval()
    result["params"] = count_params(model)

    print(f"[GATE] {variant}: hitung FLOPs...")
    result["gflops"] = measure_flops(model)

    print(f"[GATE] {variant}: ekspor ONNX...")
    try:
        export_onnx(model, onnx_path)
    except Exception as e:
        result["error"] = f"export_onnx: {type(e).__name__}: {e}"
        json.dump(result, open(json_path, "w"), indent=2)
        print(f"[GAGAL] ekspor ONNX: {e}")
        sys.exit(1)

    result["onnx"] = inspect_onnx(onnx_path)
    del model
    torch.cuda.empty_cache()

    sparse = variant in SPARSE_VARIANTS
    print(f"[GATE] {variant}: build engine TensorRT (sparse_weights={sparse})...")
    ok, errs = build_engine_ex(onnx_path, engine_path, sparse=sparse)
    result["trt_sparse_weights_flag"] = sparse
    if not ok:
        result["error"] = "build_engine gagal"
        result["trt_errors"] = errs
        json.dump(result, open(json_path, "w"), indent=2)
        print(f"[GAGAL] build engine: {errs}")
        sys.exit(1)

    print(f"[GATE] {variant}: benchmark...")
    result.update(benchmark_engine_with_vram(engine_path))
    result["engine_device_memory_mb"] = engine_device_memory_mb(engine_path)
    result["engine_file_mb"] = os.path.getsize(engine_path) / (1024 ** 2)

    json.dump(result, open(json_path, "w"), indent=2)
    gf = result["gflops"]
    gf_str = gf if isinstance(gf, str) else f"{gf:.1f} GFLOPs"
    print(f"[SELESAI] {variant}: {result['latency_ms']:.4f} ms | "
          f"{result['engine_device_memory_mb']:.1f} MB | {gf_str}")


if __name__ == "__main__":
    main()
