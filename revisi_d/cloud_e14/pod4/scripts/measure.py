"""Ukur latensi SATU engine per proses (dipanggil per konfigurasi oleh run_measure.sh).

Angka PRIMER `latency_ms` = protokol naskah persis (quant_sweep.benchmark_engine_with_vram):
10 pemanasan, lalu 100 execute_async_v3 beruntun di satu stream, waktu dinding / 100.
Tambahan: 100 iterasi bertimer CUDA event per-iterasi -> median, p95, sd (sesudah angka primer,
tidak mempengaruhinya).

Pakai: python3 scripts/measure.py <sesi> <cfg>   -> results/<sesi>/latency/<cfg>.json
"""
import json, os, sys, time
import numpy as np
from cuda.bindings import runtime as cr
from trt_common import Runner, ck

N_WARM, N_ITER = 10, 100


def main():
    sesi, cfg = sys.argv[1], sys.argv[2]
    eng = f"engines/{sesi}/{cfg}.engine"
    if not os.path.exists(eng):
        eng = f"engines/{sys.argv[3]}/{cfg}.engine" if len(sys.argv) > 3 else eng
    r = Runner(eng)
    r.launch(N_WARM); r.sync()
    ck(cr.cudaDeviceSynchronize())
    t0 = time.time()
    r.launch(N_ITER); r.sync()
    primary = (time.time() - t0) / N_ITER * 1000

    e0, e1 = ck(cr.cudaEventCreate())[0], ck(cr.cudaEventCreate())[0]
    per = []
    for _ in range(N_ITER):
        ck(cr.cudaEventRecord(e0, r.stream)); r.launch(1); ck(cr.cudaEventRecord(e1, r.stream))
        ck(cr.cudaEventSynchronize(e1))
        per.append(ck(cr.cudaEventElapsedTime(e0, e1))[0])
    per = np.array(per)
    out = dict(cfg=cfg, engine=eng, latency_ms=primary, n_warmup=N_WARM, n_iter=N_ITER,
               per_iter_mean_ms=float(per.mean()), per_iter_median_ms=float(np.median(per)),
               per_iter_p95_ms=float(np.percentile(per, 95)), per_iter_sd_ms=float(per.std(ddof=1)),
               device_memory_size_v2_bytes=int(r.engine.device_memory_size_v2),
               engine_file_bytes=os.path.getsize(eng), t=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    os.makedirs(f"results/{sesi}/latency", exist_ok=True)
    json.dump(out, open(f"results/{sesi}/latency/{cfg}.json", "w"), indent=1)
    print(f"{cfg}: {primary:.3f} ms (median/iter {out['per_iter_median_ms']:.3f}, p95 {out['per_iter_p95_ms']:.3f})")
    r.close()


if __name__ == "__main__":
    main()
