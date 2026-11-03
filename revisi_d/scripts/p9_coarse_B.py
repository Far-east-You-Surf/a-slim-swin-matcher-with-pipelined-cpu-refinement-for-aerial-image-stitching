"""P9 -- pola "jalur kasar terkompilasi + kepala geometrik B" diwujudkan pada SETIAP pembanding (uji pola).

Untuk tiap metode: engine TensorRT FP16 jalur KASAR (sama dgn engine latensi P2: results/p2c/engines/<m>_coarse.engine;
usulan: cloud_e14/engines/final/b8_s16.engine) -> matriks keyakinan -> mutual-NN (tanpa ambang, = pipa usulan) ->
pusat sel -> varian none / B (MAGSAC -> pra-lengkung -> LK init nol -> petakan balik, e13_refine.prewarp_refine).
Pasangan: UseGeo sepanjang-jalur d = 1-2 (identik P6/P8). Metrik = e13_refine.score (galat GT LiDAR, pose AUC).
Praproses = pembungkus asli: abu-abu [0,1] 480x640 (JamMa: RGB ternormalisasi ImageNet; usulan: RGB [0,1]).
Keluaran (tahan-putus, satu berkas per metode x d): revisi_d/results/p9/<m>__d<d>.jsonl(.done) + summary.json
Pakai: python revisi_d/scripts/p9_coarse_B.py [m1,m2,...] [--limit N] | summary"""
import json, os, sys
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, AERIAL_ROOT + "/revisi_d/scripts")
import e13_refine as E          # chdir ke ROOT
import cv2, numpy as np, torch, torch.nn.functional as F
import tensorrt as trt
UG = E.UG
OUT = "revisi_d/results/p9"; os.makedirs(OUT, exist_ok=True)
ENG = {"usulan": "revisi_d/cloud_e14/engines/final/b8_s16.engine"}
for m in ("efficientloftr", "topicfm_fast", "loftr", "jamma", "aspanformer", "matchformer", "lgfctr"):
    ENG[m] = f"revisi_d/results/p2c/engines/{m}_coarse.engine"
HW = (480, 640)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


class Coarse:
    def __init__(self, m):
        self.m = m
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.e = self.rt.deserialize_cuda_engine(open(ENG[m], "rb").read())
        self.c = self.e.create_execution_context()
        self.ins, self.out = [], None
        for i in range(self.e.num_io_tensors):
            n = self.e.get_tensor_name(i)
            shp, dt = tuple(self.e.get_tensor_shape(n)), self.e.get_tensor_dtype(n)
            tdt = {trt.float16: torch.float16, trt.float32: torch.float32}[dt]
            if self.e.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ins.append((n, shp, tdt))
            elif len(shp) == 3:                       # (1, L, S) matriks keyakinan
                self.out = torch.empty(shp, dtype=tdt, device="cuda"); self.c.set_tensor_address(n, self.out.data_ptr())
            else:
                buf = torch.empty(shp, dtype=tdt, device="cuda"); self.c.set_tensor_address(n, buf.data_ptr())
        L = self.out.shape[1]
        s = int(round((HW[0] * HW[1] / L) ** 0.5))
        self.grid = (HW[0] // s, HW[1] // s)
        assert self.grid[0] * self.grid[1] == L, (m, L, s)

    def prep(self, p):
        im = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        oh, ow = im.shape[:2]
        n, shp, tdt = self.ins[0]
        if shp[1] == 1:                                    # abu-abu [0,1], cv2.resize (= p2c_common)
            g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (HW[1], HW[0]))
            x = torch.from_numpy(g).cuda().float()[None, None] / 255
        else:
            x = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255
            if self.m == "usulan":                         # = stitch_demo.load_pair_tensor (ToTensor + Resize antialias)
                x = F.interpolate(x, size=HW, mode="bilinear", antialias=True, align_corners=False)
            else:                                          # JamMa: cv2.resize lalu ImageNet (= p2c_common)
                r = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255, (HW[1], HW[0]))
                x = (torch.from_numpy(r).cuda().permute(2, 0, 1)[None] - MEAN) / STD
        return x.to(tdt).contiguous(), ow, oh

    def match(self, pA, pB):
        (xa, ow, oh), (xb, _, _) = self.prep(pA), self.prep(pB)
        self.c.set_tensor_address(self.ins[0][0], xa.data_ptr()); self.c.set_tensor_address(self.ins[1][0], xb.data_ptr())
        self.c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        c = self.out[0].float()
        bj, bi = c.argmax(1), c.argmax(0)
        rows = torch.arange(c.shape[0], device="cuda")
        keep = bi[bj] == rows
        i1, i2 = rows[keep], bj[keep]
        h, w = self.grid
        a = torch.stack([(i1 % w + 0.5) * ow / w, (i1 // w + 0.5) * oh / h], 1)
        b = torch.stack([(i2 % w + 0.5) * ow / w, (i2 // w + 0.5) * oh / h], 1)
        return a.cpu().double().numpy(), b.cpu().double().numpy(), ow, oh


def run(m, d, limit=0):
    f = f"{OUT}/{m}__d{d}.jsonl"
    if os.path.exists(f + ".done"):
        print("lewat:", f); return
    C = Coarse(m); cache = UG.DepthCache(); n = 0
    with open(f, "w") as fo:
        for ds, A, B in E.along_pairs(d):
            pA, pB = UG.img_path(ds, A[0]), UG.img_path(ds, B[0])
            a, b, ow, oh = C.match(pA, pB)
            depthA = cache.get(ds, A[0])
            H0, _ = cv2.findHomography(a, b, cv2.USAC_MAGSAC, E.SD.RANSAC_THR) if len(a) >= 8 else (None, None)
            outs = {"none": b, "B": E.prewarp_refine(E.gray(pA), E.gray(pB), a, b, H0, "lk")[0] if H0 is not None else b}
            for v, bb in outs.items():
                s = E.score(a, bb, ow, oh, A, B, depthA)
                s.update(model=m, variant=v, d=d, dataset=ds, frame_a=A[0], frame_b=B[0], grid=list(C.grid))
                fo.write(json.dumps(s) + "\n")
            n += 1
            if limit and n >= limit:
                break
    if not limit:
        open(f + ".done", "w").write(str(n))
    print(m, d, "grid", C.grid, "pasangan", n, flush=True)
    del C; torch.cuda.empty_cache()


def summarize():
    S = {}
    for m in ENG:
        for v in ("none", "B"):
            R = []
            for d in (1, 2):
                f = f"{OUT}/{m}__d{d}.jsonl"
                if os.path.exists(f + ".done"):
                    R += [json.loads(l) for l in open(f) if f'"variant": "{v}"' in l]
            if not R:
                continue
            med = [r["med_err_inb"] for r in R if "med_err_inb" in r]
            S[f"{m}|{v}"] = dict(n=len(R), median_err_inb=float(np.median(med)),
                                 n_matches=float(np.median([r["n"] for r in R])),
                                 success_v2=100 * float(np.mean([r["gate_v2"] for r in R])),
                                 **E.pose_auc([r.get("pose_err", 180.0) or 180.0 for r in R]))
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    for k, s in S.items():
        print(f"{k:22s} n {s['n']} M {s['n_matches']:.0f} med {s['median_err_inb']:.2f} px AUC@5 {s['auc@5']:.1f} "
              f"@10 {s['auc@10']:.1f} v2 {s['success_v2']:.1f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    if args and args[0] == "summary":
        summarize(); sys.exit()
    for m in (args[0].split(",") if args else list(ENG)):
        for d in (1, 2):
            try:
                run(m, d, lim)
            except Exception as e:
                print("GAGAL", m, d, type(e).__name__, e, flush=True)
    summarize()
