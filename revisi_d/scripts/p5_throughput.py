"""P5 -- throughput stitching end-to-end (pasangan berurutan d=1 sepanjang satu penerbangan).

Tahap per pasangan: dekode JPEG -> ubah ukuran 480x640 -> pencocok -> mutual-NN -> [kepala B] -> MAGSAC.
Setiap bingkai didekode SEKALI (jendela geser), seperti stitching nyata.
Konfigurasi (tiap satu berkas hasil; yang sudah ada dilewati = tahan-putus):
  ours_cpu_serial   dekode CPU (cv2) + engine TRT usulan + mutual-NN GPU + MAGSAC, serial, tanpa B
  ours_gpu_serial   dekode GPU (nvJPEG, torchvision.io.decode_jpeg) + engine + MAGSAC, serial, tanpa B
  ours_gpu_B_pipe   dekode GPU + engine di utas utama; kepala B (pra-lengkung+LK, setengah resolusi) + MAGSAC
                    di kumpulan utas CPU -> dipipakan (GPU mengerjakan pasangan berikutnya)
  <tp>_trt_gpu      pembanding TERKOMPILASI ujung-ke-ujung (engine e2e P2, K=3072, abu-abu 480x640) + dekode GPU,
                    serial -- pembanding yang adil untuk ours_gpu_serial (sama-sama TRT FP16 + nvJPEG)
  <tp>_eager_gpu    pembanding PyTorch eager (match() lengkap termasuk tahap halusnya), dekode GPU
Urutan: UseGeo Dataset-1 (sepanjang-jalur, 1989x1320), sheffield (4000x3000), G Uji (4800x3200).
Pakai: python revisi_d/scripts/p5_throughput.py [konfigurasi,...]
Keluaran: revisi_d/results/p5/<urutan>__<konfigurasi>.json"""
import json, os, sys, time, concurrent.futures as cf
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md
ROOT = AERIAL_ROOT + ""
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/revisi_d/scripts")
sys.path.insert(0, f"{ROOT}/revisi_d/cloud_e14/scripts")
import cv2, numpy as np, torch, torch.nn.functional as F
import stitch_demo as SD
from e13_refine import prewarp_refine
OUT = os.environ.get("P5_OUT", f"{ROOT}/revisi_d/results/p5"); os.makedirs(OUT, exist_ok=True)
ENGINE = f"{ROOT}/revisi_d/cloud_e14/engines/final/b8_s16.engine"
N_PAIRS = int(os.environ.get("P5_N", 60))
cv2.setNumThreads(int(os.environ.get("P5_CVTHREADS", 4)))   # P29: 1 = tiap pekerja HG-LK satu inti


# P20: urutan publik tambahan pengganti survei privat untuk waktu & energi (Eksperimen B)
PUBLIC_EXTRA = {"golf9": AERIAL_DATA + "/Data Uji Coba/DroneMapper_Golf9_May2016"}


def seq_files(seq):
    if seq == "usegeo1":
        import usegeo_gt_eval as UG
        rows = UG.load_poses(1)
        return [UG.img_path(1, r[0]) for r in rows][:N_PAIRS + 1]
    if seq == "usegeo1full":                             # P20: foto UseGeo resolusi penuh (publik, 7953x5279)
        import usegeo_gt_eval as UG
        d = os.path.join(UG.USEGEO, "Dataset-1", "Undistorted_images_full_res")
        rows = [r for r in UG.load_poses(1) if os.path.exists(os.path.join(d, r[0]))]
        return [os.path.join(d, r[0]) for r in rows][:N_PAIRS + 1]
    folder = PUBLIC_EXTRA.get(seq) or SD.FLIGHTS[seq]
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))[:N_PAIRS + 1]


def decode_cpu(p):
    im = SD._imread(p)
    return im


def decode_gpu(p):
    import torchvision.io as tio
    data = tio.read_file(p)
    return tio.decode_jpeg(data, device="cuda")          # [3,H,W] uint8 (orientasi EXIF diabaikan)


def to_input_cpu(im):
    t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1).float() / 255
    return F.interpolate(t[None], size=SD.INPUT_HW, mode="bilinear", antialias=True, align_corners=False).half()


def to_input_gpu(t):
    return F.interpolate(t[None].float() / 255, size=SD.INPUT_HW, mode="bilinear", antialias=True, align_corners=False).half()


class Engine:
    def __init__(self):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.e = self.rt.deserialize_cuda_engine(open(ENGINE, "rb").read())
        self.c = self.e.create_execution_context()
        self.out = torch.empty(tuple(self.e.get_tensor_shape("conf_matrix")), dtype=torch.float16, device="cuda")
        self.c.set_tensor_address("conf_matrix", self.out.data_ptr())
        self.s = torch.cuda.current_stream()

    def __call__(self, a, b):
        a, b = a.contiguous(), b.contiguous()
        self.c.set_tensor_address("img1", a.data_ptr()); self.c.set_tensor_address("img2", b.data_ptr())
        self.c.execute_async_v3(self.s.cuda_stream)
        return self.out


class E2E:
    """Engine e2e pembanding (results/p2c/engines/<m>_e2e.engine): image0/1 abu-abu -> mkpts0_f, mkpts1_f, mconf."""
    def __init__(self, m):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.e = self.rt.deserialize_cuda_engine(open(f"{ROOT}/revisi_d/results/p2c/engines/{m}_e2e.engine", "rb").read())
        self.c = self.e.create_execution_context(); self.ins = []; self.o = {}
        for i in range(self.e.num_io_tensors):
            n = self.e.get_tensor_name(i); shp = tuple(self.e.get_tensor_shape(n))
            dt = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int64: torch.int64, trt.int32: torch.int32}[self.e.get_tensor_dtype(n)]
            if self.e.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ins.append((n, dt))
            else:
                self.o[n] = torch.empty(shp, dtype=dt, device="cuda"); self.c.set_tensor_address(n, self.o[n].data_ptr())

    def prep(self, t):                                   # t: [3,H,W] uint8 GPU -> abu-abu [0,1] 480x640
        g = (0.299 * t[0].float() + 0.587 * t[1].float() + 0.114 * t[2].float())[None, None] / 255
        return F.interpolate(g, size=SD.INPUT_HW, mode="bilinear", align_corners=False).to(self.ins[0][1]).contiguous()

    def __call__(self, a, b, ow, oh):
        self.c.set_tensor_address(self.ins[0][0], a.data_ptr()); self.c.set_tensor_address(self.ins[1][0], b.data_ptr())
        self.c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        ok = self.o["mconf"].float() > 0
        sx, sy = ow / SD.INPUT_HW[1], oh / SD.INPUT_HW[0]
        k0 = self.o["mkpts0_f"][ok].float() * torch.tensor([sx, sy], device="cuda")
        k1 = self.o["mkpts1_f"][ok].float() * torch.tensor([sx, sy], device="cuda")
        return k0.cpu().double().numpy(), k1.cpu().double().numpy()


def mnn(conf, ow, oh, grid=(48, 64)):
    c = conf[0].float()
    bj, bi = c.argmax(1), c.argmax(0)
    rows = torch.arange(c.shape[0], device=c.device)
    keep = bi[bj] == rows
    i1, i2 = rows[keep], bj[keep]
    h, w = grid
    a = torch.stack([(i1 % w + 0.5) * ow / w, (i1 // w + 0.5) * oh / h], 1)
    b = torch.stack([(i2 % w + 0.5) * ow / w, (i2 // w + 0.5) * oh / h], 1)
    return a.cpu().numpy().astype(np.float64), b.cpu().numpy().astype(np.float64)


def magsac(a, b):
    if len(a) < 8:
        return False
    H, m = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 5.0)
    return H is not None


def head_B(g1, g2, a, b, scale=float(os.environ.get("P5_BSCALE", 0.5))):  # P23-A: 1.0 = skala akurasi naskah A
    H0, _ = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 5.0) if len(a) >= 8 else (None, None)
    if H0 is None:
        return False
    s = scale; h1 = cv2.resize(g1, None, fx=s, fy=s); h2 = cv2.resize(g2, None, fx=s, fy=s)
    S = np.diag([s, s, 1.0]); Hs = S @ H0 @ np.linalg.inv(S)
    bb, _ = prewarp_refine(h1, h2, a * s, b * s, Hs, "lk")
    return magsac(a * s, bb)


def run(seq, cfg):
    f = f"{OUT}/{seq}__{cfg}.json"
    if os.path.exists(f):
        print("lewat:", f); return
    files = seq_files(seq)
    t0 = time.perf_counter(); stage = {"decode": 0.0, "match": 0.0, "post": 0.0}; ok = 0
    if cfg.startswith("ours"):
        eng = Engine()
        gpu = "gpu" in cfg
        prev = None; prev_g = None; pend = []
        pool = cf.ThreadPoolExecutor(4) if cfg.endswith("pipe") else None
        # pemanasan: dekoder (nvJPEG/cv2), interpolasi, engine -- tidak dihitung
        for p in files[:2]:
            w = to_input_gpu(decode_gpu(p)) if gpu else to_input_cpu(decode_cpu(p))
        for _ in range(5):
            mnn(eng(w, w), 640, 480)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, p in enumerate(files):
            ts = time.perf_counter()
            if gpu:
                t = decode_gpu(p); x = to_input_gpu(t); oh, ow = t.shape[1:]
                g = (t.float().mean(0).byte().cpu().numpy() if pool else None)
            else:
                im = decode_cpu(p); x = to_input_cpu(im); oh, ow = im.shape[:2]; g = None
            torch.cuda.synchronize(); stage["decode"] += time.perf_counter() - ts
            if prev is not None:
                ts = time.perf_counter()
                conf = eng(prev, x); a, b = mnn(conf, ow, oh)
                stage["match"] += time.perf_counter() - ts
                ts = time.perf_counter()
                if pool:
                    pend.append(pool.submit(head_B, prev_g, g, a, b))
                else:
                    ok += magsac(a, b)
                stage["post"] += time.perf_counter() - ts
            prev, prev_g = x, g
        if pool:
            ts = time.perf_counter(); ok = sum(fu.result() for fu in pend); stage["post"] += time.perf_counter() - ts
            pool.shutdown()
    elif cfg.endswith("_trt_gpu"):
        eng = E2E(cfg.replace("_trt_gpu", ""))
        for p in files[:2]:
            w = eng.prep(decode_gpu(p))
        for _ in range(5):
            eng(w, w, 640, 480)
        torch.cuda.synchronize(); t0 = time.perf_counter(); prev = None
        for i, p in enumerate(files):
            ts = time.perf_counter(); t = decode_gpu(p); x = eng.prep(t); oh, ow = t.shape[1:]
            torch.cuda.synchronize(); stage["decode"] += time.perf_counter() - ts
            if prev is not None:
                ts = time.perf_counter(); a, b = eng(prev, x, ow, oh); stage["match"] += time.perf_counter() - ts
                ts = time.perf_counter(); ok += magsac(a, b); stage["post"] += time.perf_counter() - ts
            prev = x
    else:
        label = cfg.replace("_eager_gpu", "")
        if label in ("loftr", "topicfm_fast", "topicfm_plus", "jamma", "disk_lightglue"):
            m = SD.build_matchers(label)[0]
        else:
            import tp_matchers as TM
            m = TM.build(label)
        for _ in range(2):
            m.match(files[0], files[1])                 # pemanasan
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(len(files) - 1):          # match() mendekode sendiri dua bingkai (tak bisa jendela geser)
            ts = time.perf_counter(); a, b, ow, oh = m.match(files[i], files[i + 1]); torch.cuda.synchronize()
            stage["match"] += time.perf_counter() - ts
            ts = time.perf_counter(); ok += magsac(np.asarray(a, float), np.asarray(b, float)); stage["post"] += time.perf_counter() - ts
    dt = time.perf_counter() - t0
    n = len(files) - 1
    res = dict(seq=seq, cfg=cfg, n_pairs=n, seconds=dt, pairs_per_s=n / dt, ms_per_pair=1000 * dt / n,
               stage_ms_per_pair={k: 1000 * v / n for k, v in stage.items()}, registered=int(ok),
               gpu=torch.cuda.get_device_name(0), image_wh=None)
    json.dump(res, open(f, "w"), indent=1)
    print(f"{seq:9s} {cfg:22s} {res['pairs_per_s']:6.1f} pasangan/s ({res['ms_per_pair']:.1f} ms) "
          f"tahap {{{', '.join(f'{k} {v:.1f}' for k, v in res['stage_ms_per_pair'].items())}}} reg {ok}/{n}", flush=True)


def main():
    cfgs = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "ours_cpu_serial", "ours_gpu_serial", "ours_gpu_B_pipe", "topicfm_fast_eager_gpu", "efficientloftr_eager_gpu",
        "loftr_eager_gpu", "jamma_eager_gpu"]
    for seq in ("usegeo1", "sheffield", "G Uji"):
        for c in cfgs:
            try:
                run(seq, c)
            except Exception as e:
                print("GAGAL", seq, c, type(e).__name__, e, flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
