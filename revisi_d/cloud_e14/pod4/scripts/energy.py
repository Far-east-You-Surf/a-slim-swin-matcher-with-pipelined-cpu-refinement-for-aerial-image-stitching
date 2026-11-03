"""P3/P4 -- energi & daya per inferensi (protokol Artikel A, langkah 23), portabel lokal & pod.

NVML dibaca lewat ctypes (libnvidia-ml.so.1) -> tanpa dependensi tambahan.
  idle   : daya idle rerata selama 20 s (GPU tanpa beban)
  engine : engine TensorRT apa pun (nama I/O bebas), masukan acak, dijalankan TERUS >= 30 s;
           energi/inferensi = dE/N (nvmlDeviceGetTotalEnergyConsumption, mJ kumulatif);
           energi dinamis = (dE - P_idle * dt) / N; daya rerata & puncak, clock SM, suhu disampel 50 ms;
           memori: device_memory_size_v2 engine + memori proses (NVML) puncak.
  eager  : pembanding PyTorch eager (forward saja) dari revisi_d/scripts/tp_matchers.py atau stitch_demo,
           protokol sama (hanya lokal).
Pakai (dari cloud_e14/): python3 scripts/energy.py idle <sesi> <rep>
                          python3 scripts/energy.py engine <sesi> <tag> <engine_path> <rep>
                          python3 scripts/energy.py eager <sesi> <label> <rep>
Keluaran: results/<sesi>/energy/<tag>_rep<rep>.json
"""
import ctypes, json, os, sys, threading, time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", ".."))

_nv = ctypes.CDLL("libnvidia-ml.so.1")
_nv.nvmlInit_v2(); _h = ctypes.c_void_p(); _nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(_h))


def energy_mj():
    v = ctypes.c_ulonglong(); _nv.nvmlDeviceGetTotalEnergyConsumption(_h, ctypes.byref(v)); return v.value


def power_w():
    v = ctypes.c_uint(); _nv.nvmlDeviceGetPowerUsage(_h, ctypes.byref(v)); return v.value / 1000.0


def clock_sm():
    v = ctypes.c_uint(); _nv.nvmlDeviceGetClockInfo(_h, 1, ctypes.byref(v)); return v.value


def temp_c():
    v = ctypes.c_uint(); _nv.nvmlDeviceGetTemperature(_h, 0, ctypes.byref(v)); return v.value


def gpu_name():
    b = ctypes.create_string_buffer(96); _nv.nvmlDeviceGetName(_h, b, 96); return b.value.decode()


def proc_mem_mb():
    """Memori terpakai GPU keseluruhan (MB) -- proksi memori proses (satu proses aktif)."""
    class M(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
    m = M(); _nv.nvmlDeviceGetMemoryInfo(_h, ctypes.byref(m)); return m.used / 2**20


class Sampler:
    def __init__(self, period=0.05):
        self.p, self.s, self.stop = period, [], threading.Event()

    def _run(self):
        while not self.stop.is_set():
            self.s.append((power_w(), clock_sm(), temp_c(), proc_mem_mb())); time.sleep(self.p)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop.set(); self.t.join()

    def summary(self):
        import statistics as st
        P = [x[0] for x in self.s]
        return dict(power_mean_w=st.mean(P), power_max_w=max(P), sm_clock_mean=st.mean(x[1] for x in self.s),
                    temp_max_c=max(x[2] for x in self.s), mem_used_max_mb=max(x[3] for x in self.s), n_samples=len(P))


def out_path(sesi, tag, rep):
    d = f"results/{sesi}/energy"; os.makedirs(d, exist_ok=True); return f"{d}/{tag}_rep{rep}.json"


def idle_power(sesi):
    f = sorted(p for p in os.listdir(f"results/{sesi}/energy") if p.startswith("idle_"))
    vals = [json.load(open(f"results/{sesi}/energy/{x}"))["power_mean_w"] for x in f]
    return sum(vals) / len(vals) if vals else None


def run_loop(step, sync, sesi, tag, rep, extra, min_s=30.0):
    for _ in range(20):
        step()
    sync()
    n = 0
    with Sampler() as S:
        e0, t0 = energy_mj(), time.perf_counter()
        while time.perf_counter() - t0 < min_s:
            for _ in range(10):
                step()
            sync(); n += 10
        e1, t1 = energy_mj(), time.perf_counter()
    dt, de = t1 - t0, (e1 - e0) / 1000.0
    pi = idle_power(sesi)
    res = dict(tag=tag, rep=rep, gpu=gpu_name(), n=n, seconds=dt, latency_ms=1000 * dt / n,
               energy_j_per_inf=de / n, power_from_energy_w=de / dt,
               energy_dyn_j_per_inf=(de - pi * dt) / n if pi else None, idle_w=pi, **S.summary(), **extra)
    json.dump(res, open(out_path(sesi, tag, rep), "w"), indent=1)
    print(f"{tag} rep{rep}: {res['latency_ms']:.3f} ms, {res['energy_j_per_inf']*1000:.1f} mJ/inf "
          f"(dinamis {res['energy_dyn_j_per_inf']*1000 if pi else float('nan'):.1f}), {res['power_mean_w']:.0f} W, "
          f"mem {res['mem_used_max_mb']:.0f} MB")


def main():
    mode, sesi = sys.argv[1], sys.argv[2]
    if mode == "idle":
        rep = sys.argv[3]
        with Sampler() as S:
            time.sleep(20)
        json.dump(dict(tag="idle", rep=rep, gpu=gpu_name(), **S.summary()), open(out_path(sesi, "idle", rep), "w"), indent=1)
        print("idle", S.summary()["power_mean_w"], "W")
    elif mode == "engine":
        tag, eng, rep = sys.argv[3], sys.argv[4], sys.argv[5]
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trt_common import Runner
        r = Runner(eng)
        run_loop(lambda: r.launch(1), r.sync, sesi, tag, rep,
                 dict(engine=eng, device_memory_mb=r.engine.device_memory_size_v2 / 2**20,
                      engine_file_mb=os.path.getsize(eng) / 2**20, kind="engine"))
    elif mode == "eager":
        label, rep = sys.argv[3], sys.argv[4]
        root = AERIAL_ROOT + ""
        sys.path.insert(0, root); sys.path.insert(0, f"{root}/revisi_d/scripts")
        import torch
        from collect_pilar3_multipair import PAIR_SETS
        if label in ("loftr", "topicfm_fast", "topicfm_plus", "jamma", "disk_lightglue"):
            import stitch_demo as SD
            m = SD.build_matchers(label)[0]
        else:
            import tp_matchers as TM
            m = TM.build(label)
        p1, p2 = PAIR_SETS["g_uji_out_of_domain"][0]
        fn = m._prep_for_timing(p1, p2)
        torch.cuda.reset_peak_memory_stats()

        def step():
            with torch.no_grad():
                fn()
        run_loop(step, torch.cuda.synchronize, sesi, label, rep,
                 dict(kind="eager", torch_peak_alloc_mb=None))
        # memori puncak alokasi torch sesudah loop
        p = out_path(sesi, label, rep); d = json.load(open(p))
        d["torch_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        json.dump(d, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
