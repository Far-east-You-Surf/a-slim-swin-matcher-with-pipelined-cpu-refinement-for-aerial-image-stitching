"""P2c -- ukur latensi engine dgn protokol PRIMER yang SAMA dgn cloud_e14/scripts/measure.py:
10 pemanasan, lalu 100 execute_async_v3 beruntun di satu stream, waktu dinding / 100 (masukan acak,
trt_common.Runner). Ditambah 100 iterasi CUDA-event (median/p95). 3 ulangan di PROSES TERPISAH -> median.

Pakai: python p2c_measure.py <engine> [<engine> ...]      -> revisi_d/results/p2c/latency/<nama>.json
       python p2c_measure.py --one <engine>               (satu proses = satu ulangan, cetak JSON)
"""
import json, os, subprocess, sys, time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
ROOT = AERIAL_ROOT + ""
sys.path.insert(0, f"{ROOT}/revisi_d/cloud_e14/scripts")
OUT = f"{ROOT}/revisi_d/results/p2c/latency"
N_WARM, N_ITER, N_REP = 10, 100, 3


def one(eng):
    import numpy as np
    from cuda.bindings import runtime as cr
    from trt_common import Runner, ck
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
    out = dict(latency_ms=primary, per_iter_median_ms=float(np.median(per)),
               per_iter_p95_ms=float(np.percentile(per, 95)),
               device_memory_size_v2_bytes=int(r.engine.device_memory_size_v2),
               engine_file_bytes=os.path.getsize(eng), t=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    r.close()
    return out


def gpu_snapshot():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def wait_idle(max_util=15, n_ok=5, timeout_s=3600):
    """Tunggu GPU menganggur (utilisasi <= max_util % pada n_ok sampel beruntun @0,5 s); proses lain
    (subagen P1) berbagi GPU ini. Kembalikan detik menunggu (None bila timeout)."""
    t0 = time.time(); ok = 0
    while time.time() - t0 < timeout_s:
        try:
            u = int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip().splitlines()[0])
        except Exception:
            return None
        ok = ok + 1 if u <= max_util else 0
        if ok >= n_ok:
            return round(time.time() - t0, 1)
        time.sleep(0.5)
    return None


def main():
    if sys.argv[1] == "--one":
        print("JSON" + json.dumps(one(sys.argv[2])))
        return
    os.makedirs(OUT, exist_ok=True)
    for eng in sys.argv[1:]:
        reps = []
        for k in range(N_REP):
            waited = wait_idle()
            snap = gpu_snapshot()
            p = subprocess.run([sys.executable, __file__, "--one", eng], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
            if not line:
                print(p.stdout[-2000:], p.stderr[-3000:]); raise SystemExit(f"gagal: {eng}")
            d = json.loads(line[0][4:]); d["gpu_before"] = snap; d["waited_idle_s"] = waited
            reps.append(d)
        import statistics as st
        res = dict(engine=eng, protocol="10 warmup + 100 back-to-back execute_async_v3, wall/100; 3 proses, median",
                   latency_ms_median=st.median(r["latency_ms"] for r in reps),
                   latency_ms_reps=[r["latency_ms"] for r in reps],
                   per_iter_median_ms_median=st.median(r["per_iter_median_ms"] for r in reps),
                   device_memory_size_v2_bytes=reps[0]["device_memory_size_v2_bytes"],
                   engine_file_bytes=reps[0]["engine_file_bytes"], reps=reps)
        name = os.path.basename(eng).replace(".engine", "")
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
        print(f"{name}: {res['latency_ms_median']:.3f} ms (reps {[round(x,3) for x in res['latency_ms_reps']]}), "
              f"devmem {res['device_memory_size_v2_bytes']/2**20:.1f} MB")


if __name__ == "__main__":
    main()
