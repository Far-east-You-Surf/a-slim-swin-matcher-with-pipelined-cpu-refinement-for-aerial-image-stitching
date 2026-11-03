"""Bangun 8 engine dari onnx/ dengan metode IDENTIK sumber angka naskah
(gate_pilar3.build_engine_ex): STRONGLY_TYPED, workspace 8 GiB, tanpa flag presisi
(ONNX sudah FP16). Log per konfigurasi -> results/<sesi>/build/<cfg>.log

Pakai (dari cloud_e14/): python3 scripts/build_engines.py <sesi> [cfg ...]
"""
import json, os, sys, time
import tensorrt as trt

CFGS = ["b18_s32", "b12_s32", "b8_s32", "b18_s16", "b12_s16", "b8_s16", "b6_s16", "b4_s16"]


def build(onnx_path, engine_path, log_path, workspace_gb=8):
    lines = []

    class L(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, sev, msg):
            if sev <= trt.ILogger.Severity.WARNING:
                lines.append(f"[{sev}] {msg}")

    logger = L()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        lines += [str(parser.get_error(i)) for i in range(parser.num_errors)]
    t0 = time.time()
    ser = None
    if ok:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
        ser = builder.build_serialized_network(network, config)
        if ser is not None:
            with open(engine_path, "wb") as f:
                f.write(ser)
    dt = time.time() - t0
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(ok=ser is not None, build_s=round(dt, 1), trt=trt.__version__,
                engine_mb=round(os.path.getsize(engine_path) / 2**20, 3) if ser is not None else None,
                n_log=len(lines))


def main():
    sesi = sys.argv[1]
    cfgs = sys.argv[2:] or CFGS
    os.makedirs(f"results/{sesi}/build", exist_ok=True)
    os.makedirs(f"engines/{sesi}", exist_ok=True)
    summary = {}
    for c in cfgs:
        print(f"[build] {c} ...", flush=True)
        summary[c] = build(f"onnx/{c}.onnx", f"engines/{sesi}/{c}.engine",
                           f"results/{sesi}/build/{c}.log")
        print(f"   -> {summary[c]}", flush=True)
    with open(f"results/{sesi}/build/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
