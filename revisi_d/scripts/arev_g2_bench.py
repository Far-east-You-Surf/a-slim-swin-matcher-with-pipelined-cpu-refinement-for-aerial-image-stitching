"""Revisi A IJIES -- G2: pengukuran biaya dengan SATU protokol (butir R1.7), dijalankan sebagai pekerjaan EKSKLUSIF.

Prasyarat (dicek dan dicatat di env.json tiap sesi): profil LACT stok 300 W ('Bench'), GPU tanpa layar (CUDA_VISIBLE_DEVICES
dari penjadwal; NVML dibuka lewat bus PCI perangkat itu, BUKAN indeks 0), beban GPU lain dihentikan, tidak ada pekerjaan GPU lain,
RAPL terbaca (bila tidak: tahap energi dilewati dan dicatat, bukan gagal).

Mode:
  lat   <sesi>          latensi engine (protokol measure.py naskah: 10 pemanasan, 100 execute_async_v3 beruntun, waktu
                        dinding/100; plus 100 iterasi ber-event CUDA -> median/p95/sd), tiap engine proses sendiri,
                        urutan diacak per sesi
  energy <sesi>         energi GPU per inferensi (loop >= 30 s, NVML mJ kumulatif, daya diam 20 s dikurangi)
  thr   <rep>           throughput seluruh pipa (dekode nvJPEG -> engine -> mutual-NN -> penghalus 4 pekerja x 1 utas
                        OpenCV -> MAGSAC akhir), 60 pasangan per urutan + stempel waktu selesai tiap pasangan
                        -> laju total (termasuk isi/kuras) DAN laju tunak (pasangan 10%..90%); energi GPU (NVML) dan CPU
                        (RAPL paket) dalam jendela tunak, kotor dan bersih (dikurangi daya diam terukur sebelum tiap
                        konfigurasi); grayscale penghalus = rumus cvtColor (0,299/0,587/0,114), sama dengan jalur akurasi
Unit = (mode, sesi/rep, konfigurasi) dengan penanda .done; aman diulang.
3 Nov 2026 (instruksi peneliti: GPU tetap di profil hemat daya): throughput diulang lengkap 5 ulangan sebagai himpunan
'thr_powersave' (thr <rep> thr_powersave); thr/1 (Bench, stok) disimpan sebagai rujukan stok. SIFT kini nfeatures=8000 seperti
jalur akurasi, dan RAPL dibaca tiap 1 s dengan koreksi putaran ulang (lihat Power).
Pakai: python revisi_d/scripts/arev_g2_bench.py lat 1 | energy 1 | thr 1 | one-lat <engine> <out.json>
"""
import ctypes
import json
import os
import random
import subprocess
import sys
import threading
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_GPU_TOOLS = _os.environ.get("AERIAL_GPU_TOOLS", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"
OUT = f"{RD}/results/arev/g2"
PY = f"{ROOT}/aerial_fp8_env/bin/python"
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
ENG = {"proposed_b8_s16": f"{RD}/cloud_e14/engines/final/b8_s16.engine",
       "proposed_b18_s16u": f"{RD}/cloud_e14/engines/final/b18_s16u.engine"}
for m in ("matchformer", "lgfctr", "aspanformer", "loftr", "efficientloftr", "topicfm_fast", "jamma"):
    ENG[f"{m}_e2e"] = f"{RD}/results/p2c/engines/{m}_e2e.engine"
    ENG[f"{m}_coarse"] = f"{RD}/results/p2c/engines/{m}_coarse.engine"
    k1 = f"{RD}/results/p2c/engines/{m}_e2e_k1024.engine"
    if os.path.exists(k1):
        ENG[f"{m}_e2e_k1024"] = k1
for tag in ("d8", "d18", "L0", "L4", "s32", "r480", "edm_e2e"):
    p = f"{OUT}/engines/{tag}.engine"
    if os.path.exists(p):
        ENG[f"sweep_{tag}"] = p


# ------------------------------------------------------------------ NVML (perangkat yang benar) + RAPL
class Power:
    def __init__(self):
        self.nv = ctypes.CDLL("libnvidia-ml.so.1")
        self.nv.nvmlInit_v2()
        import torch
        p = torch.cuda.get_device_properties(0)
        bus = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        self.bus = bus
        self.h = ctypes.c_void_p()
        rc = self.nv.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(), ctypes.byref(self.h))
        if rc != 0:
            raise RuntimeError(f"NVML tidak menemukan perangkat {bus} (rc {rc})")
        self.rapl = os.access(RAPL, os.R_OK)
        if self.rapl:
            # energy_uj kembali ke 0 tiap max_energy_range_uj (+-65 kJ = +-13 menit pada 80 W). Selisih awal-akhir saja
            # salah untuk jalan panjang (SIFT Sheffield 2 j 45 m, 3 Nov), jadi pencacah dibaca tiap 1 s di utas latar
            # dan dijumlahkan dengan koreksi putaran ulang; cpu_uj() mengembalikan nilai kumulatif yang tidak berputar.
            self._rmax = int(open(RAPL_MAX).read())
            self._rlock = threading.Lock()
            self._rlast = int(open(RAPL).read())
            self._racc = 0
            threading.Thread(target=self._rapl_loop, daemon=True).start()

    def _rapl_read(self):
        with self._rlock:
            v = int(open(RAPL).read())
            d = v - self._rlast
            if d < 0:
                d += self._rmax
            self._racc += d
            self._rlast = v
            return self._racc

    def _rapl_loop(self):
        while True:
            time.sleep(1.0)
            self._rapl_read()

    def gpu_mj(self):
        v = ctypes.c_ulonglong()
        self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(v))
        return v.value

    def gpu_w(self):
        v = ctypes.c_uint()
        self.nv.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(v))
        return v.value / 1000.0

    def cpu_uj(self):
        if not self.rapl:
            return None
        return self._rapl_read()

    def idle(self, secs=10):
        g0, c0, t0 = self.gpu_mj(), self.cpu_uj(), time.time()
        time.sleep(secs)
        g1, c1, t1 = self.gpu_mj(), self.cpu_uj(), time.time()
        dt = t1 - t0
        return dict(gpu_w=(g1 - g0) / 1e3 / dt, cpu_w=((c1 - c0) / 1e6 / dt) if c0 is not None and c1 >= c0 else None)


def env_snapshot(path):
    import torch
    info = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), boot=open("/proc/sys/kernel/random/boot_id").read().strip(),
                cvd=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0), rapl=os.access(RAPL, os.R_OK))
    try:
        sys.path.insert(0, AERIAL_GPU_TOOLS + "")
        import lact
        info["lact_profile"] = lact.current_profile() or "Default"
    except Exception as e:  # noqa: BLE001
        info["lact_profile"] = f"? {e}"
    for args in (["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE"], ["nvidia-smi"]):
        try:
            info[" ".join(args[1:]) or "smi"] = subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
        except Exception as e:  # noqa: BLE001
            info[" ".join(args)] = str(e)
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except Exception:  # noqa: BLE001
        pass
    info["torch"], info["cuda"] = torch.__version__, torch.version.cuda
    json.dump(info, open(path, "w"), indent=1)
    return info


def one_lat(eng, outp):
    sys.path.insert(0, f"{RD}/cloud_e14/scripts")
    import numpy as np
    from cuda.bindings import runtime as cr
    from trt_common import Runner, ck
    r = Runner(eng)
    r.launch(10)
    r.sync()
    ck(cr.cudaDeviceSynchronize())
    t0 = time.time()
    r.launch(100)
    r.sync()
    primary = (time.time() - t0) / 100 * 1000
    e0, e1 = ck(cr.cudaEventCreate())[0], ck(cr.cudaEventCreate())[0]
    per = []
    for _ in range(100):
        ck(cr.cudaEventRecord(e0, r.stream))
        r.launch(1)
        ck(cr.cudaEventRecord(e1, r.stream))
        ck(cr.cudaEventSynchronize(e1))
        per.append(ck(cr.cudaEventElapsedTime(e0, e1))[0])
    per = np.array(per)
    json.dump(dict(engine=eng, latency_ms=primary, per_iter_median_ms=float(np.median(per)),
                   per_iter_p95_ms=float(np.percentile(per, 95)), per_iter_sd_ms=float(per.std(ddof=1)),
                   device_memory_size_v2_bytes=int(r.engine.device_memory_size_v2),
                   t=time.strftime("%Y-%m-%d %H:%M:%S")), open(outp + ".tmp", "w"), indent=1)
    os.replace(outp + ".tmp", outp)


def one_energy(eng, outp, secs=30.0):
    sys.path.insert(0, f"{RD}/cloud_e14/scripts")
    from trt_common import Runner
    P = Power()
    idle = P.idle(10)
    r = Runner(eng)
    r.launch(10)
    r.sync()
    n, g0, t0 = 0, P.gpu_mj(), time.time()
    samples = []
    while time.time() - t0 < secs:
        r.launch(10)
        r.sync()
        n += 10
        samples.append(P.gpu_w())
    g1, t1 = P.gpu_mj(), time.time()
    dt = t1 - t0
    e = (g1 - g0) / 1e3
    json.dump(dict(engine=eng, n=n, seconds=dt, latency_ms=dt / n * 1e3, energy_j_per_inf=e / n,
                   energy_dyn_j_per_inf=(e - idle["gpu_w"] * dt) / n, power_mean_w=e / dt, power_max_w=max(samples),
                   idle_w=idle["gpu_w"], nvml_bus=P.bus, t=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(outp + ".tmp", "w"), indent=1)
    os.replace(outp + ".tmp", outp)


# ------------------------------------------------------------------ throughput seluruh pipa
def gray_bt601(t):
    """t [3,H,W] uint8 RGB (nvJPEG) -> abu-abu uint8 dengan bobot cvtColor (pembulatan seperti OpenCV)."""
    import torch
    g = (0.299 * t[0].float() + 0.587 * t[1].float() + 0.114 * t[2].float()).round().clamp(0, 255)
    return g.to(torch.uint8).cpu().numpy()


class EDMEngine:
    """EDM, mode deploy repo resmi (results/arev/g2/engines/edm_e2e.engine): masukan [1, 2, 480, 640] abu-abu / 255,
    keluaran [TOPK, 11]; pascaproses = deploy/run_onnx.py (dua arah, mconf > 0,2, tepi 16 px, arah dengan skor sigma
    lebih besar), di GPU. Antarmuka sama dengan p5_throughput.E2E (prep, __call__)."""
    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.e = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.c = self.e.create_execution_context()
        for i in range(self.e.num_io_tensors):
            n = self.e.get_tensor_name(i)
            dt = {trt.float16: torch.float16, trt.float32: torch.float32}[self.e.get_tensor_dtype(n)]
            t = torch.empty(tuple(self.e.get_tensor_shape(n)), dtype=dt, device="cuda")
            self.c.set_tensor_address(n, t.data_ptr())
            if self.e.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp = t
            else:
                self.out = t

    def prep(self, t):
        import torch.nn.functional as F
        g = (0.299 * t[0].float() + 0.587 * t[1].float() + 0.114 * t[2].float())[None, None] / 255
        return F.interpolate(g, size=(480, 640), mode="bilinear", align_corners=False).to(self.inp.dtype)

    def __call__(self, a, b, ow, oh):
        import torch
        self.inp[0, 0].copy_(a[0, 0])
        self.inp[0, 1].copy_(b[0, 0])
        self.c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        m0, m1 = edm_post(self.out)
        sc = torch.tensor([ow / 640, oh / 480], device=m0.device)
        return (m0 * sc).cpu().double().numpy(), (m1 * sc).cpu().double().numpy()


def edm_post(out, W=640, H=480):
    """Pascaproses resmi EDM mode deploy (deploy/run_onnx.py) pada keluaran [TOPK, 11] -> titik citra 0 dan 1 (W x H)."""
    import torch
    o = out.float()
    k0c, k1c, f01, f10 = o[:, 0:2], o[:, 2:4], o[:, 4:6], o[:, 6:8]
    s01, s10, mc = o[:, 8], o[:, 9], o[:, 10]
    m0, m1 = torch.cat([k0c, k0c + f10 * 8]), torch.cat([k1c + f01 * 8, k1c])
    ps, mc2 = torch.cat([s01, s10]), torch.cat([mc, mc])
    br = 16
    keep = (mc2 > 0.2) & (ps > 1e-6) & torch.cat([s01 > s10, s01 <= s10])
    for m in (m0, m1):
        keep &= (m[:, 0] >= br) & (m[:, 0] <= W - br) & (m[:, 1] >= br) & (m[:, 1] <= H - br)
    return m0[keep], m1[keep]


def thr_one(seq, cfg, outp, n_pairs=60):
    import concurrent.futures as cf
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, f"{RD}/scripts")
    os.environ["P5_CVTHREADS"] = "1"
    import p5_throughput as P5
    cv2.setNumThreads(1)
    P5.N_PAIRS = n_pairs
    files = P5.seq_files(seq)[: n_pairs + 1]
    Pw = Power()
    idle = Pw.idle(10)
    done_t = []
    lock = threading.Lock()

    def mark(_f=None):
        with lock:
            done_t.append(time.perf_counter())

    if cfg.startswith("proposed"):
        scale = {"proposed_full": 1.0, "proposed_half": 0.5}.get(cfg)
        eng = P5.Engine()
        pool = cf.ThreadPoolExecutor(4) if scale else None
        for p in files[:2]:
            w = P5.to_input_gpu(P5.decode_gpu(p))
        for _ in range(5):
            P5.mnn(eng(w, w), 640, 480)
        torch.cuda.synchronize()
        g0, c0, t0 = Pw.gpu_mj(), Pw.cpu_uj(), time.perf_counter()
        prev = prev_g = None
        futs = []
        for p in files:
            t = P5.decode_gpu(p)
            x = P5.to_input_gpu(t)
            oh, ow = t.shape[1:]
            g = gray_bt601(t) if pool else None
            torch.cuda.synchronize()
            if prev is not None:
                a, b = P5.mnn(eng(prev, x), ow, oh)
                if pool:
                    fu = pool.submit(P5.head_B, prev_g, g, a, b, scale)
                    fu.add_done_callback(mark)
                    futs.append(fu)
                else:
                    P5.magsac(a, b)
                    mark()
            prev, prev_g = x, g
        ok = sum(f.result() for f in futs) if pool else None
        if pool:
            pool.shutdown()
    elif cfg.endswith("_e2e"):
        eng = EDMEngine(f"{OUT}/engines/edm_e2e.engine") if cfg == "edm_e2e" else P5.E2E(cfg.replace("_e2e", ""))
        if cfg.startswith("jamma"):     # JamMa: RGB ternormalisasi ImageNet (= p2c_common.load_pair_rgb_imagenet)
            import torch.nn.functional as F
            mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
            eng.prep = lambda t: ((F.interpolate(t[None].float() / 255, size=(480, 640), mode="bilinear",
                                                 align_corners=False) - mean) / std).to(eng.ins[0][1]).contiguous()
        for p in files[:2]:
            w = eng.prep(P5.decode_gpu(p))
        for _ in range(5):
            eng(w, w, 640, 480)
        torch.cuda.synchronize()
        g0, c0, t0 = Pw.gpu_mj(), Pw.cpu_uj(), time.perf_counter()
        prev = None
        ok = 0
        for p in files:
            t = P5.decode_gpu(p)
            x = eng.prep(t)
            oh, ow = t.shape[1:]
            torch.cuda.synchronize()
            if prev is not None:
                a, b = eng(prev, x, ow, oh)
                ok += P5.magsac(a, b)
                mark()
            prev = x
    elif cfg in ("sift", "sift_B"):
        sift = cv2.SIFT_create(nfeatures=8000)   # = p26_sift.match, sumber angka akurasi SIFT (P30 -> X4); 3 Nov:
        bf = cv2.BFMatcher(cv2.NORM_L2)          # sebelumnya tanpa batas (192 rb titik di Sheffield, 457 s/pasangan)
        pool = cf.ThreadPoolExecutor(4)

        def feat(p):
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            k, dsc = sift.detectAndCompute(im, None)
            return im, np.float32([kk.pt for kk in k]), dsc

        def pair(fa, fb):
            (ga, ka, da), (gb, kb, db) = fa.result(), fb.result()
            mm = bf.knnMatch(da, db, k=2)
            good = [m for m, n in (x for x in mm if len(x) == 2) if m.distance < 0.8 * n.distance]
            a = np.float64([ka[m.queryIdx] for m in good])
            b = np.float64([kb[m.trainIdx] for m in good])
            if cfg == "sift_B" and len(a) >= 8:
                return P5.head_B(ga, gb, a, b, 1.0)
            return P5.magsac(a, b)

        g0, c0, t0 = Pw.gpu_mj(), Pw.cpu_uj(), time.perf_counter()
        fts = [pool.submit(feat, p) for p in files]
        pp = cf.ThreadPoolExecutor(2)
        futs = []
        for i in range(len(files) - 1):
            fu = pp.submit(pair, fts[i], fts[i + 1])
            fu.add_done_callback(mark)
            futs.append(fu)
        ok = sum(f.result() for f in futs)
        pool.shutdown()
        pp.shutdown()
    else:
        raise ValueError(cfg)
    torch.cuda.synchronize()
    t1, g1, c1 = time.perf_counter(), Pw.gpu_mj(), Pw.cpu_uj()
    n = len(files) - 1
    ts = sorted(done_t)
    lo, hi = int(0.1 * n), int(0.9 * n) - 1
    steady = (hi - lo) / (ts[hi] - ts[lo]) if hi > lo and ts[hi] > ts[lo] else None
    dt = t1 - t0
    res = dict(seq=seq, cfg=cfg, n_pairs=n, seconds=dt, pairs_per_s=n / dt, steady_pairs_per_s=steady,
               first_pair_s=ts[0] - t0 if ts else None, drain_s=t1 - ts[-1] if ts else None,
               registered=ok, gpu_j_per_pair=(g1 - g0) / 1e3 / n,
               cpu_j_per_pair=((c1 - c0) / 1e6 / n) if c0 is not None and c1 is not None and c1 >= c0 else None,
               idle_gpu_w=idle["gpu_w"], idle_cpu_w=idle["cpu_w"], nvml_bus=Pw.bus, rapl=Pw.rapl)
    if res["cpu_j_per_pair"] is not None:
        tot = res["gpu_j_per_pair"] + res["cpu_j_per_pair"]
        res.update(system_j_per_pair_gross=tot,
                   system_j_per_pair_net=tot - (idle["gpu_w"] + idle["cpu_w"]) * dt / n)
    json.dump(res, open(outp + ".tmp", "w"), indent=1)
    os.replace(outp + ".tmp", outp)
    print(json.dumps(res), flush=True)


THR_CFGS = ["proposed_full", "proposed_half", "proposed_none", "matchformer_e2e", "lgfctr_e2e", "aspanformer_e2e",
            "loftr_e2e", "efficientloftr_e2e", "topicfm_fast_e2e", "jamma_e2e", "edm_e2e", "sift", "sift_B"]


def main():
    mode = sys.argv[1]
    if mode == "one-lat":
        return one_lat(sys.argv[2], sys.argv[3])
    if mode == "one-energy":
        return one_energy(sys.argv[2], sys.argv[3])
    if mode == "one-thr":
        return thr_one(sys.argv[2], sys.argv[3], sys.argv[4])
    rep = sys.argv[2]
    tset = sys.argv[3] if len(sys.argv) > 3 else mode   # thr: himpunan keluaran (thr = stok, thr_powersave = profil hemat daya)
    os.makedirs(f"{OUT}/{tset}/{rep}", exist_ok=True)
    os.chdir(ROOT)
    env_snapshot(f"{OUT}/{tset}/{rep}/env.json")
    rnd = random.Random(int(rep) * 7919)
    if mode in ("lat", "energy"):
        items = list(ENG.items())
        rnd.shuffle(items)
        for k, (cfg, eng) in enumerate(items, 1):
            outp = f"{OUT}/{mode}/{rep}/{cfg}.json"
            if not os.path.exists(outp):
                subprocess.run([PY, __file__, f"one-{mode}" if mode == "lat" else "one-energy", eng, outp], check=False)
            print(f"PROGRESS {k}/{len(items)} {mode} sesi {rep} {cfg}", flush=True)
    elif mode == "thr":
        items = [(s, c) for s in ("usegeo1", "sheffield") for c in THR_CFGS]
        rnd.shuffle(items)
        for k, (seq, cfg) in enumerate(items, 1):
            outp = f"{OUT}/{tset}/{rep}/{seq}__{cfg}.json"
            if not os.path.exists(outp):
                subprocess.run([PY, __file__, "one-thr", seq, cfg, outp], check=False)
            print(f"PROGRESS {k}/{len(items)} thr ulangan {rep} {seq} {cfg}", flush=True)
    open(f"{OUT}/{tset}/{rep}.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
