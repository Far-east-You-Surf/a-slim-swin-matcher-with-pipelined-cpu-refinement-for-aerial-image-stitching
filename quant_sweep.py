"""
Sweep kuantisasi tersatukan: 3 varian model (old baseline, Model A data-only,
Model B coarse-only) x ~10 varian presisi (FP16 ref, INT8 W+A/W-only, FP8
W+A/W-only, FP4 weight-only, FP4 W4A8, FP4 W4A4 loftr-only/swin-only/full),
di resolusi native 480x640 (comparable langsung ke LoFTR/LightGlue & tabel
historis evaluate_native640.py).

Setiap kombinasi (model, presisi) dijalankan sbg PROSES TERPISAH (dipanggil
lewat subprocess oleh run_sweep_all.py) -- pola yang sama dgn skrip run_*.py
individual sebelumnya, supaya state ModelOpt/TensorRT/custom-op-registration
antar varian tidak saling bocor (proven pattern di proyek ini).

Dead-code-elimination Model B coarse-only tervalidasi (2026-07-12): jumlah
node & Gemm/MatMul di graph ONNX forward()-only Model B IDENTIK dgn old
baseline -- fine_proj/fine_head tidak ikut terekspor, jadi perbandingan adil.

Output: 1 file JSON per kombinasi di results_dir/{model}__{precision}.json
berisi latency_ms, fps, peak_vram_mb, dan per-pair (aerial234/g_uji): mse,
cos_sim, matches, inliers, inlier_ratio (semua relatif thd FP16 PyTorch
model YANG SAMA -- bukan lintas model).

Pemakaian: python quant_sweep.py <model_variant> <precision> [results_dir]
  model_variant  : old | dataonly | finematch_coarse
  precision      : fp16 | int8_wa | int8_wo | fp8_wa | fp8_wo |
                    fp4_wo | fp4_w4a8 | fp4_w4a4_loftr | fp4_w4a4_swin | fp4_w4a4_full
"""
import os
import sys
import json
import time
import threading
import warnings

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

import modelopt.torch.quantization as mtq
from torch.onnx import utils as onnx_utils
from torch.onnx import register_custom_op_symbolic
from modelopt.torch.quantization.export_onnx import export_fp4, configure_linear_module_onnx_quantizers

from model import AerialStitchingModel
from model_finematch import AerialStitchingModelFineMatch
from model_linear import AerialStitchingModelLinear
from calib_utils import make_real_calibrate_loop
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

RES_HW = (480, 640)
DEVICE = "cuda"

MODEL_VARIANTS = {
    "old":              dict(cls=AerialStitchingModel, ckpt="aerial_stitching_fp16_best.pth"),
    "dataonly":         dict(cls=AerialStitchingModel, ckpt="aerial_stitching_fp16_dataonly_best.pth"),
    "finematch_coarse": dict(cls=AerialStitchingModelFineMatch, ckpt="aerial_stitching_fp16_finematch_best.pth"),
    # §56 ablasi terbalik: bobot IDENTIK dgn "old", hanya operator attention yang
    # ditukar softmax -> linear (LinearMHA, tanpa parameter, load strict 1:1).
    # Pasangannya dgn "old" adalah ablasi variabel tunggal, arah kebalikan §26.
    "old_linear":       dict(cls=AerialStitchingModelLinear, ckpt="aerial_stitching_fp16_best.pth"),
}

PAIRS = {
    "aerial234": (
        AERIAL_DATA_UJICOBA + "/Dataset Aerial234/001.JPG",
        AERIAL_DATA_UJICOBA + "/Dataset Aerial234/002.JPG",
    ),
    "g_uji": (
        AERIAL_DATA_UJICOBA + "/G Uji/DSC00003.JPG",
        AERIAL_DATA_UJICOBA + "/G Uji/DSC00004.JPG",
    ),
}

_FP4_SYMBOLIC_REGISTERED = False


def _ensure_fp4_symbolic():
    global _FP4_SYMBOLIC_REGISTERED
    if _FP4_SYMBOLIC_REGISTERED:
        return

    def export_fp4_wrapper(g, inputs, block_size, amax, num_bits, trt_high_precision_dtype, onnx_quantizer_type):
        res = export_fp4(g, inputs, block_size, amax, num_bits, trt_high_precision_dtype, onnx_quantizer_type)
        if hasattr(res, "type"):
            if hasattr(inputs.type(), 'sizes'):
                res.setType(inputs.type().with_sizes(inputs.type().sizes()))
            else:
                res.setType(inputs.type())
        return res

    register_custom_op_symbolic("trt::TRT_FP4QDQ", export_fp4_wrapper, 18)
    _FP4_SYMBOLIC_REGISTERED = True


def load_model(variant):
    spec = MODEL_VARIANTS[variant]
    model = spec["cls"](d_model=128, nhead=8, num_layers=8).to(DEVICE).half().eval()
    model.load_state_dict(torch.load(spec["ckpt"], map_location="cpu"))
    return model


def disable_conv2d_weight_quantizer(model):
    for name, module in model.named_modules():
        if "Conv2d" in module.__class__.__name__:
            if hasattr(module, "weight_quantizer"):
                module.weight_quantizer.disable()


def apply_dtype_patches(onnx_model):
    import onnx
    new_nodes = []
    for node in onnx_model.graph.node:
        new_nodes.append(node)
        if node.op_type == "DequantizeLinear":
            dq_out = node.output[0]
            cast_out = dq_out + "_casted_fp16"
            cast_node = onnx.helper.make_node(
                "Cast", inputs=[dq_out], outputs=[cast_out],
                to=onnx.TensorProto.FLOAT16, name=node.name + "_CastToFP16"
            )
            new_nodes.append(cast_node)
            for consumer in onnx_model.graph.node:
                if consumer is node:
                    continue
                for i in range(len(consumer.input)):
                    if consumer.input[i] == dq_out:
                        consumer.input[i] = cast_out
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    for init in onnx_model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT:
            arr = onnx.numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(onnx.numpy_helper.from_array(arr, init.name))

    for node in onnx_model.graph.node:
        if node.op_type not in ("Constant", "ConstantOfShape"):
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
                arr = onnx.numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(onnx.numpy_helper.from_array(arr))

    for node in onnx_model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                attr.i = onnx.TensorProto.FLOAT16

    return onnx_model


def dedup_fp4_qdq(onnx_model):
    fp4_qdq_seen = {}
    nodes_to_remove = []
    for node in onnx_model.graph.node:
        if node.op_type == "TRT_FP4QDQ":
            weight_name = node.input[0]
            if weight_name in fp4_qdq_seen:
                original_output = fp4_qdq_seen[weight_name]
                duplicate_output = node.output[0]
                for consumer in onnx_model.graph.node:
                    for i in range(len(consumer.input)):
                        if consumer.input[i] == duplicate_output:
                            consumer.input[i] = original_output
                nodes_to_remove.append(node)
            else:
                fp4_qdq_seen[weight_name] = node.output[0]
    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)
    return onnx_model


# --------------------------------------------------------------------------
# Fungsi quantize_and_export per varian presisi. Semua menerima `model`
# (sudah .half().eval(), belum dikuantisasi) dan `onnx_path`, mengembalikan
# tidak ada (menulis onnx_path di disk).
# --------------------------------------------------------------------------

def qe_int8(model, onnx_path, weight_only):
    cfg = mtq.config.INT8_WEIGHT_ONLY_CFG if weight_only else mtq.config.INT8_DEFAULT_CFG
    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, cfg, forward_loop=calibrate_loop)
    disable_conv2d_weight_quantizer(model)

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=17,
                           input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.int8_exporter import INT8QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = INT8QuantExporter.process_model(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


def qe_fp16_engine(model, onnx_path):
    """FP16 TANPA kuantisasi sama sekali, tapi lewat jalur ekspor+build yang SAMA
    PERSIS dgn varian terkuantisasi (pengekspor legacy, opset 17, nama I/O sama).

    Ditambahkan 2026-08-05 (log §34.1). Baris `fp16` di sweep dijalankan di PyTorch
    eager, sehingga kolom speedup selama ini membandingkan engine-terkuantisasi vs
    PyTorch -- mencampur manfaat kompilasi dgn manfaat kuantisasi. Varian ini
    menyediakan pembanding yang benar: engine vs engine. Tidak ada kalibrasi, tidak
    ada mtq.quantize, tidak ada exporter ModelOpt -- graf FP16 apa adanya.
    """
    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=17,
                           input_names=['img1', 'img2'], output_names=['conf_matrix'])


def qe_fp8(model, onnx_path, weight_only):
    if weight_only:
        import copy
        cfg = copy.deepcopy(mtq.config.FP8_DEFAULT_CFG)
        for entry in cfg["quant_cfg"]:
            if entry.get("quantizer_name") == "*input_quantizer":
                entry.pop("cfg", None)
                entry["enable"] = False
    else:
        cfg = mtq.config.FP8_DEFAULT_CFG

    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, cfg, forward_loop=calibrate_loop)
    disable_conv2d_weight_quantizer(model)

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=17,
                           input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = FP8QuantExporter.process_model(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


def qe_fp4_wo(model, onnx_path):
    _ensure_fp4_symbolic()
    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, mtq.config.NVFP4_DEFAULT_CFG, forward_loop=calibrate_loop)
    for name, module in model.named_modules():
        if hasattr(module, "input_quantizer"):
            module.input_quantizer.disable()
        if "Conv2d" in module.__class__.__name__:
            if hasattr(module, "weight_quantizer"):
                module.weight_quantizer.disable()

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=18, custom_opsets={"trt": 18},
                           input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = dedup_fp4_qdq(onnx_model)
    onnx_model = NVFP4QuantExporter.compute_scales(onnx_model)
    onnx_model = NVFP4QuantExporter.compress_weights(onnx_model)
    onnx_model = NVFP4QuantExporter.post_process(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


def qe_fp4_w4a8(model, onnx_path):
    _ensure_fp4_symbolic()
    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, mtq.config.W4A8_NVFP4_FP8_CFG, forward_loop=calibrate_loop)
    disable_conv2d_weight_quantizer(model)

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_utils.export(model, dummy, onnx_path, opset_version=18, custom_opsets={"trt": 18},
                           input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter
    from modelopt.onnx.export.fp8_exporter import FP8QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = dedup_fp4_qdq(onnx_model)
    onnx_model = NVFP4QuantExporter.compute_scales(onnx_model)
    onnx_model = NVFP4QuantExporter.compress_weights(onnx_model)
    onnx_model = NVFP4QuantExporter.post_process(onnx_model)
    onnx_model = FP8QuantExporter.process_model(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


def qe_fp4_w4a4_loftr(model, onnx_path):
    """Varian 'true' lama: hanya modul rank 2D/3D (blok LoFTR custom) yang FP4 aktif;
    Swin backbone (rank-4) otomatis gugur -- TIDAK pakai reshape trick."""
    _ensure_fp4_symbolic()
    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, mtq.config.NVFP4_DEFAULT_CFG, forward_loop=calibrate_loop)

    rank_by_module = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                rank_by_module[name] = inputs[0].dim()
        return hook

    for name, module in model.named_modules():
        if hasattr(module, 'input_quantizer') or hasattr(module, 'weight_quantizer'):
            hook_handles.append(module.register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        calibrate_loop(model)
    for h in hook_handles:
        h.remove()

    for name, module in model.named_modules():
        rank = rank_by_module.get(name)
        if rank is not None and rank not in (2, 3):
            if hasattr(module, 'weight_quantizer'):
                module.weight_quantizer.disable()
            if hasattr(module, 'input_quantizer'):
                module.input_quantizer.disable()

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with configure_linear_module_onnx_quantizers(model):
            onnx_utils.export(model, dummy, onnx_path, opset_version=18, custom_opsets={"trt": 18},
                               input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = dedup_fp4_qdq(onnx_model)
    onnx_model = NVFP4QuantExporter.compute_scales(onnx_model)
    onnx_model = NVFP4QuantExporter.compress_weights(onnx_model)
    onnx_model = NVFP4QuantExporter.post_process(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


def install_reshape_hooks(model, target_names):
    handles = []
    shape_cache = {}

    def pre_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            if x.dim() == 4:
                b, h, w, c = x.shape
                shape_cache[name] = (b, h, w)
                x2 = x.reshape(b, h * w, c)
                return (x2,) + tuple(inputs[1:])
            return inputs
        return hook

    def post_hook(name):
        def hook(module, inputs, output):
            if name in shape_cache:
                b, h, w = shape_cache.pop(name)
                c_out = output.shape[-1]
                return output.reshape(b, h, w, c_out)
            return output
        return hook

    for name, module in model.named_modules():
        if name in target_names:
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))
    return handles


def qe_fp4_w4a4_scope(model, onnx_path, scenario):
    """scenario: 'swin_only' atau 'full'. Reshape-trick utk backbone 4D
    (name.startswith('backbone') -- cocok utk 'backbone' (old/dataonly) MAUPUN
    'backbone_fine'/'backbone_coarse' (finematch_coarse), tervalidasi string
    prefix match sblm dipakai)."""
    _ensure_fp4_symbolic()
    calibrate_loop = make_real_calibrate_loop(n_pairs=16, size=RES_HW)
    model = mtq.quantize(model, mtq.config.NVFP4_DEFAULT_CFG, forward_loop=calibrate_loop)

    rank_by_module = {}
    cls_by_module = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                rank_by_module[name] = inputs[0].dim()
        return hook

    for name, module in model.named_modules():
        if hasattr(module, 'input_quantizer') or hasattr(module, 'weight_quantizer'):
            cls_by_module[name] = module.__class__.__name__
            hook_handles.append(module.register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        calibrate_loop(model)
    for h in hook_handles:
        h.remove()

    reshape_targets = set()
    for name, rank in rank_by_module.items():
        if rank == 4 and name.startswith("backbone"):
            cls = cls_by_module.get(name, "")
            if "Linear" in cls or "LayerNorm" in cls:
                reshape_targets.add(name)
    install_reshape_hooks(model, reshape_targets)

    for name, module in model.named_modules():
        rank = rank_by_module.get(name)
        if rank is None:
            continue
        is_backbone = name.startswith("backbone")
        is_reshaped = name in reshape_targets
        if scenario == "swin_only":
            keep = is_backbone and (is_reshaped or rank in (2, 3))
        else:  # full
            keep = is_reshaped or rank in (2, 3)
        if not keep:
            if hasattr(module, 'weight_quantizer'):
                module.weight_quantizer.disable()
            if hasattr(module, 'input_quantizer'):
                module.input_quantizer.disable()

    dummy = (torch.randn(1, 3, *RES_HW, device=DEVICE).half(), torch.randn(1, 3, *RES_HW, device=DEVICE).half())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with configure_linear_module_onnx_quantizers(model):
            onnx_utils.export(model, dummy, onnx_path, opset_version=18, custom_opsets={"trt": 18},
                               input_names=['img1', 'img2'], output_names=['conf_matrix'])

    import onnx
    from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter
    onnx_model = onnx.load(onnx_path)
    onnx_model = dedup_fp4_qdq(onnx_model)
    onnx_model = NVFP4QuantExporter.compute_scales(onnx_model)
    onnx_model = NVFP4QuantExporter.compress_weights(onnx_model)
    onnx_model = NVFP4QuantExporter.post_process(onnx_model)
    onnx_model = apply_dtype_patches(onnx_model)
    onnx.save(onnx_model, onnx_path)


PRECISION_FUNCS = {
    "fp16_engine":     qe_fp16_engine,
    "int8_wa":         lambda m, p: qe_int8(m, p, weight_only=False),
    "int8_wo":         lambda m, p: qe_int8(m, p, weight_only=True),
    "fp8_wa":          lambda m, p: qe_fp8(m, p, weight_only=False),
    "fp8_wo":          lambda m, p: qe_fp8(m, p, weight_only=True),
    "fp4_wo":          qe_fp4_wo,
    "fp4_w4a8":        qe_fp4_w4a8,
    "fp4_w4a4_loftr":  qe_fp4_w4a4_loftr,
    "fp4_w4a4_swin":   lambda m, p: qe_fp4_w4a4_scope(m, p, "swin_only"),
    "fp4_w4a4_full":   lambda m, p: qe_fp4_w4a4_scope(m, p, "full"),
}


# --------------------------------------------------------------------------
# Build, benchmark (+ VRAM), quality eval
# --------------------------------------------------------------------------

def build_engine(onnx_path, engine_path, workspace_gb=8):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    try:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    except AttributeError:
        network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return True


def benchmark_engine_with_vram(engine_path, n_iters=100, n_warmup=10):
    import tensorrt as trt
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pid = os.getpid()

    def sample_proc_vram():
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            return None
        for p in procs:
            if p.pid == pid and p.usedGpuMemory:
                return p.usedGpuMemory / (1024 ** 2)
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape = engine.get_tensor_shape("img1")
    output_shape = engine.get_tensor_shape("conf_matrix")

    img1 = torch.randn(*input_shape, dtype=torch.float16, device="cuda")
    img2 = torch.randn(*input_shape, dtype=torch.float16, device="cuda")
    output = torch.empty(tuple(output_shape), dtype=torch.float16, device="cuda")

    context.set_tensor_address("img1", img1.data_ptr())
    context.set_tensor_address("img2", img2.data_ptr())
    context.set_tensor_address("conf_matrix", output.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(n_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    vram_samples = []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.is_set():
            v = sample_proc_vram()
            if v is not None:
                vram_samples.append(v)
            time.sleep(0.01)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    end = time.time()

    stop_flag.set()
    t.join()

    avg_ms = (end - start) / n_iters * 1000
    fps = 1000 / avg_ms
    peak_vram_mb = max(vram_samples) if vram_samples else None
    return dict(latency_ms=avg_ms, fps=fps, peak_vram_mb=peak_vram_mb)


def benchmark_pytorch_with_vram(model, n_iters=100, n_warmup=10):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pid = os.getpid()

    def sample_proc_vram():
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            return None
        for p in procs:
            if p.pid == pid and p.usedGpuMemory:
                return p.usedGpuMemory / (1024 ** 2)
        return None

    img1 = torch.randn(1, 3, *RES_HW, device=DEVICE).half()
    img2 = torch.randn(1, 3, *RES_HW, device=DEVICE).half()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(img1, img2)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        vram_samples = []
        stop_flag = threading.Event()

        def sampler():
            while not stop_flag.is_set():
                v = sample_proc_vram()
                if v is not None:
                    vram_samples.append(v)
                time.sleep(0.01)

        t = threading.Thread(target=sampler, daemon=True)
        t.start()

        start = time.time()
        for _ in range(n_iters):
            model(img1, img2)
        torch.cuda.synchronize()
        end = time.time()

        stop_flag.set()
        t.join()
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    avg_ms = (end - start) / n_iters * 1000
    fps = 1000 / avg_ms
    peak_vram_mb = max(vram_samples) if vram_samples else None
    return dict(latency_ms=avg_ms, fps=fps, peak_vram_mb=peak_vram_mb,
                torch_peak_allocated_mb=torch_peak_allocated_mb)


def load_pair(img1_path, img2_path):
    img1_cv = cv2.imread(img1_path)
    img2_cv = cv2.imread(img2_path)
    orig_h, orig_w = img1_cv.shape[:2]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(RES_HW)])
    img1 = transform(cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    img2 = transform(cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    return img1, img2, orig_w, orig_h


def extract_matches(conf_matrix, orig_w, orig_h, threshold=0.1):
    conf = conf_matrix[0]
    mask = conf > threshold
    indices = torch.nonzero(mask)
    H_feat, W_feat = 24, 32
    scale_x, scale_y = orig_w / W_feat, orig_h / H_feat
    pts1, pts2 = [], []
    for idx in indices:
        idx1, idx2 = idx[0].item(), idx[1].item()
        y1, x1 = divmod(idx1, W_feat)
        y2, x2 = divmod(idx2, W_feat)
        pts1.append([x1 * scale_x, y1 * scale_y])
        pts2.append([x2 * scale_x, y2 * scale_y])
    return np.array(pts1), np.array(pts2)


def evaluate_matches(pts1, pts2):
    if len(pts1) < 4:
        return len(pts1), 0, 0.0
    H_matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    ratio = (inliers / len(pts1)) * 100 if len(pts1) > 0 else 0.0
    return len(pts1), inliers, ratio


def compare(conf_ref, conf_test):
    ref_flat = conf_ref.flatten().double()
    test_flat = conf_test.flatten().double()
    mse = torch.mean((ref_flat - test_flat) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), test_flat.unsqueeze(0)).item()
    return mse, cos_sim


def run_pytorch_conf(model, img1, img2):
    with torch.no_grad():
        conf = model(img1.to(DEVICE).half(), img2.to(DEVICE).half())
    return conf.float().cpu()


def run_engine_conf(engine_path, img1, img2):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    output_shape = tuple(engine.get_tensor_shape("conf_matrix"))
    img1_d = img1.to(DEVICE).half().contiguous()
    img2_d = img2.to(DEVICE).half().contiguous()
    output_d = torch.empty(output_shape, dtype=torch.float16, device=DEVICE)
    context.set_tensor_address("img1", img1_d.data_ptr())
    context.set_tensor_address("img2", img2_d.data_ptr())
    context.set_tensor_address("conf_matrix", output_d.data_ptr())
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return output_d.float().cpu()


def evaluate_quality(conf_ref_by_pair, conf_test_by_pair):
    results = {}
    for pair_name, (img1, img2, orig_w, orig_h) in PAIRS_LOADED.items():
        conf_ref = conf_ref_by_pair[pair_name]
        conf_test = conf_test_by_pair[pair_name]
        mse, cos_sim = compare(conf_ref, conf_test)
        pts1, pts2 = extract_matches(conf_test, orig_w, orig_h)
        matches, inliers, ratio = evaluate_matches(pts1, pts2)
        results[pair_name] = dict(mse=mse, cos_sim=cos_sim, matches=matches,
                                   inliers=inliers, inlier_ratio=ratio)
    return results


PAIRS_LOADED = {}


def main():
    global PAIRS_LOADED
    model_variant = sys.argv[1]
    precision = sys.argv[2]
    results_dir = sys.argv[3] if len(sys.argv) > 3 else "quant_sweep_results"
    os.makedirs(results_dir, exist_ok=True)

    PAIRS_LOADED = {name: load_pair(p1, p2) for name, (p1, p2) in PAIRS.items()}

    print(f"[SWEEP] model={model_variant} precision={precision}")
    model = load_model(model_variant)

    conf_ref_by_pair = {name: run_pytorch_conf(model, img1, img2)
                         for name, (img1, img2, ow, oh) in PAIRS_LOADED.items()}

    result = dict(model=model_variant, precision=precision, res_hw=list(RES_HW))

    if precision == "fp16":
        perf = benchmark_pytorch_with_vram(model)
        result.update(perf)
        quality = evaluate_quality(conf_ref_by_pair, conf_ref_by_pair)  # identik dgn dirinya sendiri
        result["quality"] = quality
    else:
        onnx_path = f"{results_dir}/{model_variant}__{precision}.onnx"
        engine_path = f"{results_dir}/{model_variant}__{precision}.engine"

        model_for_quant = load_model(model_variant)  # instance segar, blm ada quantizer nempel
        try:
            PRECISION_FUNCS[precision](model_for_quant, onnx_path)
        except Exception as e:
            result["error"] = f"quantize_export: {type(e).__name__}: {e}"
            with open(f"{results_dir}/{model_variant}__{precision}.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"[GAGAL] quantize/export: {e}")
            sys.exit(1)

        ok = build_engine(onnx_path, engine_path)
        if not ok:
            result["error"] = "build_engine gagal"
            with open(f"{results_dir}/{model_variant}__{precision}.json", "w") as f:
                json.dump(result, f, indent=2)
            print("[GAGAL] build_engine")
            sys.exit(1)

        perf = benchmark_engine_with_vram(engine_path)
        result.update(perf)

        import tensorrt as trt
        _rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            _engine = _rt.deserialize_cuda_engine(f.read())
        result["engine_device_memory_mb"] = _engine.device_memory_size_v2 / (1024 ** 2)

        conf_test_by_pair = {}
        for name, (img1, img2, ow, oh) in PAIRS_LOADED.items():
            conf_test_by_pair[name] = run_engine_conf(engine_path, img1, img2)
        quality = evaluate_quality(conf_ref_by_pair, conf_test_by_pair)
        result["quality"] = quality

    with open(f"{results_dir}/{model_variant}__{precision}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SELESAI] {model_variant}__{precision}: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
