"""
FASE 1 + FASE 2 demonstrasi stitching (Artikel D).

KLAIM YANG DIUJI
----------------
"Untuk mosaicking, yang menentukan adalah JUMLAH dan SEBARAN inlier, bukan
presisi per-korespondensi." Rancangan ini harus bisa MEMATAHKAN klaim itu:
model usulan jalur coarse saja (sel /16 = ~10x13 px pada input 480x640),
sedangkan LoFTR punya refinement sub-piksel dan diperkirakan menang pada
akurasi geometrik. Kalau LoFTR menang di SEMUA metrik termasuk panjang rantai
dan drift, justifikasi utama Artikel D gugur dan harus dicabut.

FASE 1 -- kualitas geometrik berpasangan (per pasangan berurutan)
  * n_match, n_inlier, rasio
  * RMSE REPROYEKSI TERTAHAN: korespondensi dibelah dua secara acak, H
    diestimasi dari separuh, galat diukur pada separuh lain. Ini menghindari
    sirkularitas mengukur galat pada titik yang dipakai memfit.
  * SEBARAN: luas convex hull inlier / luas citra, dan okupansi grid 8x8.
  * VALIDITAS H: determinan, bilangan kondisi, kewarasan sudut hasil warp.
  * FLAG KEBERHASILAN -- kegagalan DIHITUNG, tidak dibuang diam-diam.

FASE 2 -- mosaik jalur sekuensial (per jalur)
  * panjang rantai sebelum gagal (metrik utama ketahanan)
  * drift terhadap GPS EXIF sesudah penyelarasan similarity (Umeyama)
  * galat fotometrik di area tumpang tindih sesudah warp
  * kanvas mosaik (opsional, --mosaic)

REFINEMENT (opsional, --refine)
  none | lk (Lucas-Kanade piramidal) | ncc (korelasi silang ternormalisasi +
  fit parabola sub-piksel). Diterapkan di ATAS match coarse, bebas arsitektur,
  sehingga berlaku sama untuk semua model kita.

Pakai:
  python stitch_demo.py --flight "G Uji" --models all --refine none --out gate_arch_results/stitch
"""

import argparse
import json
import math
import os
import sys
import traceback

import cv2
import numpy as np
import torch
from torchvision import transforms
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

# Protokol utama seluruh log: 480x640. Bisa ditimpa utk uji kekokohan --
# pembanding pihak ketiga dirancang di resolusi lebih tinggi (TopicFM diuji
# pada 1200 px, JamMa pada 832), jadi 480x640 berpotensi merugikan mereka dan
# klaim "berpijak sama" perlu diuji, bukan diasumsikan.
INPUT_HW = tuple(int(v) for v in
                 os.environ.get("STITCH_INPUT_HW", "480x640").split("x"))
SAT_K = 3200
RANSAC_THR = 5.0
MIN_MATCHES = 8
SELECT = os.environ.get("STITCH_SELECT", "topk")   # topk | bucket
# CAP jumlah match. Uji seleksi hanya bermakna bila CAP < jumlah kandidat: dgn
# cap 3200 dan kandidat mutual-NN <= 3072, top-K dan ember menyimpan himpunan
# IDENTIK dan cuma urutannya beda -- RANSAC tidak peduli urutan, jadi no-op.
CAP = int(os.environ.get("STITCH_CAP", SAT_K))

FLIGHTS = {
    "G Uji": AERIAL_DATA_UJICOBA + "/G Uji",
    "G6": AERIAL_DATA + "/Data Disertasi/Data Latih Pre-Sticth/G6",
    "sheffield": AERIAL_DATA + "/Data Uji Bersih/"
                 "drone_dataset_sheffield_park_1-master",
}
# G Uji  = LUAR DOMAIN, bersih (tidak ada di indeks pseudo-GT)
# G6     = IN-DOMAIN, TERKONTAMINASI (36 pasang ada di pseudo-GT latih) --
#          dipakai HANYA untuk rantai panjang, kontaminasinya dinyatakan.
# sheffield = LUAR DOMAIN, BERSIH (diaudit dua cara, sesi 67: geografis 7.932 km
#          dari Zieg + pencocokan SIFT/RANSAC ke 38 ortho, inlier tertinggi 6).
#          OpenDroneMap, BSD-2-Clause, boleh diredistribusi. 77 bingkai.
#          BATAS: FC300S @ 37 m = rezim Zieg, BUKAN rezim survei 374 m G Uji.
CONTAMINATED = {"G6"}


def _imread(path):
    """Baca citra dgn orientasi KONSISTEN.

    `cv2.imread` MENERAPKAN rotasi EXIF sedangkan PIL tidak. Pada G6, 2 dari 151
    foto (DSC00089, DSC00121) membawa flag orientasi berbeda sehingga terbaca
    4800x3200 sementara 149 lainnya 3200x4800 -- membuat Lucas-Kanade jatuh
    (ukuran piramida tidak sama) dan merusak geometri setiap pasangan yang
    melibatkannya. Kamera survei terpasang tetap, jadi flag itu praktis pasti
    metadata menyimpang. IGNORE_ORIENTATION memaksa semua berkas seragam.
    """
    return cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


# ----------------------------------------------------------------- pemuat citra
def load_pair_tensor(p1, p2):
    a, b = _imread(p1), _imread(p2)
    if a is None or b is None:
        raise RuntimeError(f"gagal membaca {p1} atau {p2}")
    oh, ow = a.shape[:2]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize(INPUT_HW)])
    t1 = tf(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    t2 = tf(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    return t1, t2, ow, oh


# ------------------------------------------------------------------ model wrapper
class OurModel:
    """model.py (/32) atau model_p16.py (/16), keluaran conf -> match mutual-NN."""

    def __init__(self, label, kind, ckpt, variant=None, grid=(24, 32)):
        # quant_sweep.DEVICE di-hardcode "cuda"; STITCH_DEVICE memungkinkan uji
        # asap di CPU tanpa menyentuh GPU yang sedang dipakai training.
        from quant_sweep import DEVICE
        dev = os.environ.get("STITCH_DEVICE") or DEVICE
        self.label, self.grid, self.device = label, grid, dev
        if kind == "p32":
            from model import AerialStitchingModel
            m = AerialStitchingModel(d_model=128, nhead=8, num_layers=8)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else:
            import model_p16
            m = model_p16.build(variant, d_model=128, nhead=8, num_layers=8,
                                flash=True, pretrained_backbone=False)
            sd = torch.load(ckpt, map_location="cpu")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{label}: checkpoint tidak cocok "
                                   f"(missing={missing[:3]} unexpected={unexpected[:3]})")
        # quant_sweep.DEVICE bisa str ATAU torch.device -- jangan asumsikan .type
        self.half = "cuda" in str(dev)      # CPU tidak mendukung half utk banyak op
        self.model = (m.to(dev).half() if self.half else m.to(dev).float()).eval()

    def match(self, p1, p2):
        from gate_arch_probe2 import topk_points
        t1, t2, ow, oh = load_pair_tensor(p1, p2)
        with torch.no_grad():
            cast = (lambda x: x.half()) if self.half else (lambda x: x.float())
            conf = self.model(cast(t1.to(self.device)),
                              cast(t2.to(self.device))).float().cpu()
        a, b, sc = mutual_nn_with_scores(conf, ow, oh, self.grid)
        if SELECT == "bucket":
            a, b = bucket_select(a, b, sc, ow, oh, cap=CAP)
        elif CAP < len(a):                       # top-K dgn cap yang SAMA
            idx = np.argsort(-sc)[:CAP]
            a, b = a[idx], b[idx]
        return a, b, ow, oh


class LoFTRModel:
    """LoFTR kornia, pipeline PENUH (coarse + fine sub-piksel). Sengaja tidak
    dikompensasi -- keunggulan sub-pikselnya memang nyata dan dinyatakan."""

    def __init__(self, label="loftr"):
        import kornia.feature as KF
        from quant_sweep import DEVICE
        self.label = label
        self.device = os.environ.get("STITCH_DEVICE") or DEVICE
        self.model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def match(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        oh, ow = a.shape[:2]
        g = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            g.append(torch.from_numpy(x).float()[None, None] / 255.0)
        with torch.no_grad():
            out = self.model({"image0": g[0].to(self.device), "image1": g[1].to(self.device)})
        k0 = out["keypoints0"].cpu().numpy()
        k1 = out["keypoints1"].cpu().numpy()
        sx, sy = ow / INPUT_HW[1], oh / INPUT_HW[0]
        a = (k0 * [sx, sy]).astype(np.float64)
        b = (k1 * [sx, sy]).astype(np.float64)
        sc = out.get("confidence")
        sc = sc.cpu().numpy() if sc is not None else np.ones(len(a))
        if SELECT == "bucket":
            a, b = bucket_select(a, b, sc, ow, oh, cap=CAP)
        elif CAP < len(a):
            idx = np.argsort(-sc)[:CAP]
            a, b = a[idx], b[idx]
        return a, b, ow, oh

    def _prep_for_timing(self, p1, p2):
        """Tensor disiapkan sekali; yang diukur hanya forward model."""
        a, b = _imread(p1), _imread(p2)
        g = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            g.append(torch.from_numpy(x).float()[None, None].to(self.device) / 255.0)
        return lambda: self.model({"image0": g[0], "image1": g[1]})


class DiskLightGlueModel:
    """DISK keypoints + LightGlue, the modern SuperGlue successor. Included as a
    SECOND literature baseline: the downstream evaluation originally compared only
    against LoFTR, which left the sparse-pipeline family unrepresented. Pipeline
    and keypoint budget follow `benchmark_lightglue_baseline.py` so the numbers
    stay comparable with the quality sweep already on record."""

    MAX_KP = 2048

    def __init__(self, label="disk_lightglue"):
        import kornia.feature as KF
        from quant_sweep import DEVICE
        self.label = label
        self.device = os.environ.get("STITCH_DEVICE") or DEVICE
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.lg = KF.LightGlue(features="disk").to(self.device).eval()

    def match(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        oh, ow = a.shape[:2]
        t = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            t.append(torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255.0)
        t = [x.to(self.device) for x in t]
        with torch.no_grad():
            f0 = self.disk(t[0], n=self.MAX_KP, pad_if_not_divisible=True)[0]
            f1 = self.disk(t[1], n=self.MAX_KP, pad_if_not_divisible=True)[0]
            data = {"image0": {"keypoints": f0.keypoints[None],
                               "descriptors": f0.descriptors[None],
                               "image_size": torch.tensor([[t[0].shape[3], t[0].shape[2]]],
                                                          device=self.device)},
                    "image1": {"keypoints": f1.keypoints[None],
                               "descriptors": f1.descriptors[None],
                               "image_size": torch.tensor([[t[1].shape[3], t[1].shape[2]]],
                                                          device=self.device)}}
            out = self.lg(data)
        idx = out["matches"][0].cpu().numpy()
        if len(idx) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), ow, oh
        k0 = f0.keypoints.cpu().numpy()[idx[:, 0]]
        k1 = f1.keypoints.cpu().numpy()[idx[:, 1]]
        sx, sy = ow / INPUT_HW[1], oh / INPUT_HW[0]
        aa = (k0 * [sx, sy]).astype(np.float64)
        bb = (k1 * [sx, sy]).astype(np.float64)
        if SELECT == "bucket" or CAP < len(aa):
            sc = out.get("scores")
            sc = sc[0].cpu().numpy() if sc is not None else np.ones(len(aa))
            if SELECT == "bucket":
                aa, bb = bucket_select(aa, bb, sc, ow, oh, cap=CAP)
            elif CAP < len(aa):
                sel = np.argsort(-sc)[:CAP]
                aa, bb = aa[sel], bb[sel]
        return aa, bb, ow, oh


    def _prep_for_timing(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        t = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            t.append(torch.from_numpy(x).permute(2, 0, 1).float()[None].to(self.device) / 255.0)

        def run():
            f0 = self.disk(t[0], n=self.MAX_KP, pad_if_not_divisible=True)[0]
            f1 = self.disk(t[1], n=self.MAX_KP, pad_if_not_divisible=True)[0]
            sz = lambda x: torch.tensor([[x.shape[3], x.shape[2]]], device=self.device)
            return self.lg({"image0": {"keypoints": f0.keypoints[None],
                                       "descriptors": f0.descriptors[None],
                                       "image_size": sz(t[0])},
                            "image1": {"keypoints": f1.keypoints[None],
                                       "descriptors": f1.descriptors[None],
                                       "image_size": sz(t[1])}})
        return run


# ------------------------------------------------- pembanding pihak ketiga (efisien)
# CATATAN PENTING soal proses: TopicFM dan JamMa SAMA-SAMA memakai nama paket
# tingkat atas `src`, jadi keduanya TIDAK BOLEH diimpor dalam satu proses Python.
# Jalankan terpisah (--models topicfm_fast,topicfm_plus lalu --models jamma) ke
# folder --out berbeda, kemudian gabungkan JSON-nya dengan merge_thirdparty.py.
#
# CATATAN soal keadilan pembandingan:
#   * Keduanya memakai bobot MegaDepth publik tanpa penyetelan domain, persis
#     seperti LoFTR dan DISK+LightGlue -- asimetri yang MENGUNTUNGKAN model kami
#     dan sudah dinyatakan di naskah.
#   * Keduanya diberi masukan 480x640 yang SAMA dengan semua model lain, lalu
#     titiknya diskalakan balik ke koordinat foto asli.
#   * Keduanya jalan di PyTorch EAGER, sedangkan angka 10,857 ms model usulan
#     berasal dari engine TensorRT terkompilasi. Waktu keduanya TIDAK sebanding
#     dan tidak dilaporkan sebagai perbandingan latensi.

class TopicFMModel:
    """TopicFM (AAAI 2023) / TopicFM+ -- varian detector-free efisien yang
    membatasi attention ke topik semantik. Pipeline PENUH termasuk fine stage,
    sama perlakuannya dengan LoFTR."""

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "TopicFM")
    CKPT = {"fast": "pretrained/topicfm_fast.ckpt", "plus": "pretrained/topicfm_plus.ckpt"}
    # Konfigurasi coarse per varian, disalin dari configs/megadepth_test_topicfm*.py
    COARSE = {"fast": {"n_samples": 0},
              "plus": {"n_samples": 8, "nhead": 8, "attention": "linear",
                       "n_topic_transformers": 2}}
    THR = 0.25                      # ambang match coarse pada MegaDepth, dari config resmi

    def __init__(self, variant="fast", label=None):
        if self.ROOT not in sys.path:
            sys.path.insert(0, self.ROOT)
        from src.models.topic_fm import TopicFM
        from src import get_model_cfg
        from quant_sweep import DEVICE
        self.label = label or f"topicfm_{variant}"
        self.device = os.environ.get("STITCH_DEVICE") or DEVICE
        cfg = dict(get_model_cfg())
        cfg["match_coarse"]["thr"] = self.THR
        for k, v in self.COARSE[variant].items():
            cfg["coarse"][k] = v
        cfg["loss"]["fine_type"] = "sym_epi"
        m = TopicFM(config=cfg)
        sd = torch.load(os.path.join(self.ROOT, self.CKPT[variant]), map_location="cpu")
        m.load_state_dict(sd["state_dict"])          # strict: cacat bobot harus berisik
        self.model = m.eval().to(self.device)

    def match(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        oh, ow = a.shape[:2]
        g = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            g.append(torch.from_numpy(x).float()[None, None].to(self.device) / 255.0)
        data = {"image0": g[0], "image1": g[1]}
        with torch.no_grad():
            self.model(data)                          # menulis hasil ke dalam `data`
        k0 = data["mkpts0_f"].cpu().numpy()
        k1 = data["mkpts1_f"].cpu().numpy()
        if len(k0) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), ow, oh
        sx, sy = ow / INPUT_HW[1], oh / INPUT_HW[0]
        aa = (k0 * [sx, sy]).astype(np.float64)
        bb = (k1 * [sx, sy]).astype(np.float64)
        sc = data.get("mconf")
        sc = sc.cpu().numpy() if sc is not None else np.ones(len(aa))
        if SELECT == "bucket":
            aa, bb = bucket_select(aa, bb, sc, ow, oh, cap=CAP)
        elif CAP < len(aa):
            sel = np.argsort(-sc)[:CAP]
            aa, bb = aa[sel], bb[sel]
        return aa, bb, ow, oh

    def _prep_for_timing(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        g = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            g.append(torch.from_numpy(x).float()[None, None].to(self.device) / 255.0)
        # dict baru tiap panggilan: model menulis hasilnya ke dalam dict masukan
        return lambda: self.model({"image0": g[0], "image1": g[1]})


class JamMaModel:
    """JamMa (CVPR 2025) -- attention kuadratik diganti pemindaian state-space
    (Mamba). Bobot resmi diunduh dari release v0.1 repo penulis. Pipeline PENUH
    (coarse 1/8 + fine sub-piksel), diperlakukan sama dgn LoFTR dan TopicFM.

    Kernel CUDA `mamba-ssm` HARUS dibangun utk sm_120 dgn shim header di
    `third_party/glibc_shim` -- lihat catatan pemasangan di log riset.
    """

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "JamMa")
    # Normalisasi ImageNet, mengikuti read_megadepth_color repo aslinya.
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, label="jamma"):
        if self.ROOT not in sys.path:
            sys.path.insert(0, self.ROOT)
        from demo.utlis import JamMa as _JamMaFull, cfg
        from quant_sweep import DEVICE
        self.label = label
        self.device = os.environ.get("STITCH_DEVICE") or DEVICE
        self.model = _JamMaFull(config=cfg, pretrained="official").eval().to(self.device)

    def match(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        oh, ow = a.shape[:2]
        t = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            x = (x - self.MEAN) / self.STD
            t.append(torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device))
        data = {"imagec_0": t[0], "imagec_1": t[1]}
        with torch.no_grad():
            self.model(data)                          # menulis hasil ke dalam `data`
        k0 = data["mkpts0_f"].cpu().numpy()
        k1 = data["mkpts1_f"].cpu().numpy()
        if len(k0) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), ow, oh
        sx, sy = ow / INPUT_HW[1], oh / INPUT_HW[0]
        aa = (k0 * [sx, sy]).astype(np.float64)
        bb = (k1 * [sx, sy]).astype(np.float64)
        sc = data.get("mconf_f", data.get("mconf"))
        sc = sc.cpu().numpy() if sc is not None else np.ones(len(aa))
        if len(sc) != len(aa):
            sc = np.ones(len(aa))
        if SELECT == "bucket":
            aa, bb = bucket_select(aa, bb, sc, ow, oh, cap=CAP)
        elif CAP < len(aa):
            sel = np.argsort(-sc)[:CAP]
            aa, bb = aa[sel], bb[sel]
        return aa, bb, ow, oh

    def _prep_for_timing(self, p1, p2):
        a, b = _imread(p1), _imread(p2)
        t = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = cv2.resize(x, (INPUT_HW[1], INPUT_HW[0]))
            x = (x - self.MEAN) / self.STD
            t.append(torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device))
        return lambda: self.model({"imagec_0": t[0], "imagec_1": t[1]})


# -------------------------------------------------------------------- refinement
def refine_lk(img1, img2, pts1, pts2, win=21, levels=3):
    """Lucas-Kanade piramidal, DIINISIALISASI di pts2 -- jadi ia menghaluskan
    tebakan coarse, bukan mencari dari nol."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = pts1.astype(np.float32).reshape(-1, 1, 2)
    p1i = pts2.astype(np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p1o, st, _ = cv2.calcOpticalFlowPyrLK(
        g1, g2, p0, p1i, winSize=(win, win), maxLevel=levels, criteria=crit,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    st = st.reshape(-1).astype(bool)
    out = pts2.copy()
    out[st] = p1o.reshape(-1, 2)[st]
    return pts1, out, int(st.sum())


def refine_ncc(img1, img2, pts1, pts2, half=8, search=4):
    """NCC pada tampalan kecil + fit parabola untuk sub-piksel."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h1, w1 = g1.shape
    h2, w2 = g2.shape
    out = pts2.copy()
    n_ok = 0
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if not (half <= x1i < w1 - half and half <= y1i < h1 - half):
            continue
        if not (half + search <= x2i < w2 - half - search and
                half + search <= y2i < h2 - half - search):
            continue
        tpl = g1[y1i - half:y1i + half + 1, x1i - half:x1i + half + 1]
        win = g2[y2i - half - search:y2i + half + search + 1,
                 x2i - half - search:x2i + half + search + 1]
        try:
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, _, _, mx = cv2.minMaxLoc(res)
        cx, cy = mx
        if not (0 < cx < res.shape[1] - 1 and 0 < cy < res.shape[0] - 1):
            continue
        # fit parabola 1-D pada tiap sumbu
        dx = 0.5 * (res[cy, cx - 1] - res[cy, cx + 1]) / \
             (res[cy, cx - 1] - 2 * res[cy, cx] + res[cy, cx + 1] + 1e-12)
        dy = 0.5 * (res[cy - 1, cx] - res[cy + 1, cx]) / \
             (res[cy - 1, cx] - 2 * res[cy, cx] + res[cy + 1, cx] + 1e-12)
        if abs(dx) > 1 or abs(dy) > 1:
            continue
        out[i] = [x2i + (cx - search) + dx, y2i + (cy - search) + dy]
        n_ok += 1
    return pts1, out, n_ok


REFINERS = {"none": None, "lk": refine_lk, "ncc": refine_ncc}


def bucket_select(pts1, pts2, scores, w, h, n_grid=8, cap=3200):
    """Seleksi TERDIVERSIFIKASI SECARA SPASIAL: ambil match terbaik per ember
    n_grid x n_grid pada citra 1, putaran demi putaran, sampai `cap`.

    KENAPA: top-K global memilih berdasarkan kepercayaan saja, dan kepercayaan
    menumpuk di wilayah bertekstur -- jadi ia BIAS terhadap sebaran. Defisit hull
    model kita (0,29-0,39 vs LoFTR 0,49-0,54) bisa jadi artefak protokol itu,
    bukan sifat arsitektur. Ember memisahkan kedua kemungkinan: kalau dgn ember
    sebarannya menyusul, defisitnya artefak; kalau tetap tertinggal, ia nyata.
    """
    if len(pts1) == 0:
        return pts1, pts2
    gx = np.clip((pts1[:, 0] / w * n_grid).astype(int), 0, n_grid - 1)
    gy = np.clip((pts1[:, 1] / h * n_grid).astype(int), 0, n_grid - 1)
    key = gy * n_grid + gx
    order = np.argsort(-np.asarray(scores))
    buckets = {}
    for i in order:
        buckets.setdefault(int(key[i]), []).append(int(i))
    chosen, r = [], 0
    while len(chosen) < cap:
        added = False
        for b in sorted(buckets):
            if r < len(buckets[b]):
                chosen.append(buckets[b][r]); added = True
                if len(chosen) >= cap:
                    break
        if not added:
            break
        r += 1
    idx = np.array(chosen, dtype=int)
    return pts1[idx], pts2[idx]


def mutual_nn_with_scores(conf, ow, oh, grid):
    """Match mutual-NN + skornya + koordinat piksel asli (konvensi pusat sel)."""
    c = conf[0]
    n = c.shape[0]
    h_feat, w_feat = grid
    best_j = c.argmax(dim=1)
    best_i = c.argmax(dim=0)
    rows = torch.arange(n)
    keep = best_i[best_j[rows]] == rows
    i1 = rows[keep].numpy()
    i2 = best_j[rows[keep]].numpy()
    sc = c[rows[keep], best_j[rows[keep]]].numpy()
    sx, sy = ow / w_feat, oh / h_feat
    p1 = np.stack([(i1 % w_feat + 0.5) * sx, (i1 // w_feat + 0.5) * sy], 1)
    p2 = np.stack([(i2 % w_feat + 0.5) * sx, (i2 // w_feat + 0.5) * sy], 1)
    return p1.astype(np.float64), p2.astype(np.float64), sc


# ------------------------------------------------------------------ metrik fase 1
def hull_area_frac(pts, w, h):
    if len(pts) < 3:
        return 0.0
    try:
        hull = cv2.convexHull(pts.astype(np.float32))
        return float(cv2.contourArea(hull) / (w * h))
    except cv2.error:
        return 0.0


def grid_occupancy(pts, w, h, n=8):
    if len(pts) == 0:
        return 0.0
    gx = np.clip((pts[:, 0] / w * n).astype(int), 0, n - 1)
    gy = np.clip((pts[:, 1] / h * n).astype(int), 0, n - 1)
    return float(len(set(zip(gx.tolist(), gy.tolist()))) / (n * n))


def h_sanity(H, w, h):
    if H is None:
        return dict(valid=False, reason="H None")
    det = float(np.linalg.det(H))
    try:
        cond = float(np.linalg.cond(H))
    except Exception:
        cond = float("inf")
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    try:
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return dict(valid=False, reason="transform gagal", det=det, cond=cond)
    area = float(cv2.contourArea(wc.astype(np.float32)))
    ratio = area / (w * h) if w * h else 0.0
    ok = (abs(det) > 1e-8) and np.isfinite(cond) and (cond < 1e7) and (0.2 < ratio < 5.0)
    return dict(valid=bool(ok), det=det, cond=cond, warp_area_ratio=ratio)


def holdout_rmse(pts1, pts2, seed=0):
    """H difit pada separuh korespondensi, galat diukur pada separuh lain."""
    n = len(pts1)
    if n < 2 * MIN_MATCHES:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    H, _ = cv2.findHomography(pts1[a], pts2[a], cv2.USAC_MAGSAC, RANSAC_THR)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(pts1[b].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - pts2[b], axis=1)
    # median lebih tahan pencilan daripada RMSE mentah pada himpunan bercampur
    return dict(rmse_px=float(np.sqrt(np.mean(err ** 2))),
                median_px=float(np.median(err)),
                p90_px=float(np.percentile(err, 90)), n_eval=int(len(b)))


def eval_pair(matcher, p1, p2, refiner=None):
    r = dict(pair=[os.path.basename(p1), os.path.basename(p2)])
    try:
        a, b, ow, oh = matcher.match(p1, p2)
    except Exception as e:
        r.update(success=False, reason=f"match gagal: {type(e).__name__}: {e}")
        return r, None
    r["n_matches"] = int(len(a))
    if len(a) < MIN_MATCHES:
        r.update(success=False, reason="match < minimum")
        return r, None

    if refiner is not None:
        i1, i2 = _imread(p1), _imread(p2)
        a, b, n_ref = refiner(i1, i2, a, b)
        r["n_refined"] = n_ref

    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, RANSAC_THR)
    if H is None or mask is None:
        r.update(success=False, reason="homografi gagal")
        return r, None
    mask = mask.reshape(-1).astype(bool)
    inl = int(mask.sum())
    r["n_inliers"] = inl
    r["inlier_ratio"] = float(100.0 * inl / len(a))
    r["sanity"] = h_sanity(H, ow, oh)
    r["hull_frac"] = hull_area_frac(a[mask], ow, oh)
    r["grid_occ"] = grid_occupancy(a[mask], ow, oh)
    # PERINGATAN: RMSE reproyeksi tertahan TIDAK SAH untuk matcher ber-grid. Titiknya
    # terkuantisasi ke kisi, jadi homografi yang difit pada separuh memetakan kisi ke
    # kisi secara EKSAK dan sisa pada separuh lain jadi ~0 (baseline /32 terukur 0,0000).
    # Itu konsistensi-diri kisi, bukan akurasi geometrik. Sekelas dgn cacat ambang §58.2.
    r["holdout"] = holdout_rmse(a[mask], b[mask]) if inl >= 2 * MIN_MATCHES else None
    r["holdout_valid_for_grid_matcher"] = False
    r["success"] = bool(r["sanity"]["valid"] and inl >= MIN_MATCHES)
    r["image_wh"] = [ow, oh]
    r["H"] = H.tolist()   # disimpan agar drift bisa dihitung ULANG tanpa inferensi ulang
    return r, (H if r["success"] else None)


# ------------------------------------------------------------------ metrik fase 2
def umeyama_sim(src, dst):
    """Similarity (skala+rotasi+translasi) yang menyelaraskan src ke dst."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    c = float((D * np.diag(S)).sum() / var) if var > 1e-12 else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def photometric_overlap(imgA, imgB, H, max_side=800):
    """Selisih absolut rerata di area tumpang tindih sesudah warp B->A."""
    hA, wA = imgA.shape[:2]
    sc = min(1.0, max_side / max(hA, wA))
    if sc < 1.0:
        S = np.diag([sc, sc, 1.0])
        H = S @ H @ np.linalg.inv(S)
        imgA = cv2.resize(imgA, None, fx=sc, fy=sc)
        imgB = cv2.resize(imgB, None, fx=sc, fy=sc)
        hA, wA = imgA.shape[:2]
    warped = cv2.warpPerspective(imgB, np.linalg.inv(H), (wA, hA))
    valid = cv2.warpPerspective(np.ones(imgB.shape[:2], np.uint8), np.linalg.inv(H), (wA, hA))
    m = valid.astype(bool)
    if m.sum() < 100:
        return None
    ga = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return dict(mad=float(np.abs(ga[m] - gb[m]).mean()), overlap_frac=float(m.mean()))


def read_gps_xy(files):
    """Posisi GPS lokal (meter) tiap bingkai, dari EXIF."""
    from audit_flights_phase0 import read_exif, meters_per_deg
    recs = [read_exif(f) for f in files]
    lat = [r.get("lat") for r in recs]
    lon = [r.get("lon") for r in recs]
    if any(v is None for v in lat) or any(v is None for v in lon):
        return None
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    m_lat, m_lon = meters_per_deg(lat0)
    return np.array([[(lo - lon0) * m_lon, (la - lat0) * m_lat] for la, lo in zip(lat, lon)])


def run_flight(matcher, flight, files, refiner=None, mosaic_path=None, max_frames=None, gap=1):
    """`gap` mencocokkan bingkai i dengan i+gap.

    KENAPA INI ADA: pada bingkai berurutan tumpang tindihnya 88%, sehingga setiap
    matcher memperoleh ratusan inlier padahal homografi hanya butuh empat. Estimator
    jenuh dan metrik hilir kehilangan daya pisah -- itulah sebab drift tidak
    membedakan model mana pun. Menaikkan gap menurunkan tumpang tindih sampai
    mendekati nol dan memindahkan pengukuran ke rezim tempat cakupan dan sebaran
    korespondensi seharusnya menentukan: gap terbesar yang masih teregistrasi
    adalah selubung kemampuan matcher.
    """
    if max_frames:
        files = files[:max_frames]
    pairs, Hs = [], []
    for i in range(len(files) - gap):
        rec, H = eval_pair(matcher, files[i], files[i + gap], refiner)
        rec["index"] = i
        if H is not None:
            ph = photometric_overlap(_imread(files[i]), _imread(files[i + gap]), H)
            rec["photometric"] = ph
        pairs.append(rec)
        Hs.append(H)
        print(f"    [{i:3d}] {'OK ' if rec.get('success') else 'GAGAL'} "
              f"m={rec.get('n_matches','-')} i={rec.get('n_inliers','-')} "
              f"r={rec.get('inlier_ratio',0):.1f}% "
              f"hull={rec.get('hull_frac',0):.2f} "
              f"rmse={(rec.get('holdout') or {}).get('median_px','-')}", flush=True)

    # --- rantai: panjang sampai gagal
    chain_len = 0
    for H in Hs:
        if H is None:
            break
        chain_len += 1
    n_ok = sum(1 for r in pairs if r.get("success"))

    # --- posisi pusat bingkai lewat rantai homografi
    centers, cum = [], np.eye(3)
    ok_idx = []
    for i in range(chain_len + 1):
        if i > 0:
            cum = cum @ np.linalg.inv(Hs[i - 1])
        wh = pairs[min(i, len(pairs) - 1)].get("image_wh")
        if not wh:
            break
        c = np.array([[[wh[0] / 2.0, wh[1] / 2.0]]], dtype=np.float64)
        centers.append(cv2.perspectiveTransform(c, cum).reshape(2))
        ok_idx.append(i)

    # DRIFT BERJENDELA. Rantai penuh rapuh: satu homografi buruk di awal (pasangan
    # DSC00003/4 memang yang tersulit, lihat log) merusak seluruh sisa rantai lewat
    # perkalian matriks, sehingga angka "drift" mengukur satu kegagalan itu saja dan
    # bukan perilaku matcher. Diukur ulang atas sub-rantai geser sepanjang WIN dan
    # dilaporkan mediannya -- tahan terhadap satu pasangan rusak.
    WIN = 10
    gps = read_gps_xy(files)
    drift_windows = None
    if gps is not None and len(Hs) >= WIN:
        vals = []
        for start in range(0, len(Hs) - WIN + 1):
            if any(Hs[j] is None for j in range(start, start + WIN)):
                continue
            cw, cm = [], np.eye(3)
            wh = pairs[start].get("image_wh")
            if not wh:
                continue
            c0 = np.array([[[wh[0] / 2.0, wh[1] / 2.0]]], dtype=np.float64)
            cw.append(cv2.perspectiveTransform(c0, cm).reshape(2))
            for j in range(start, start + WIN):
                cm = cm @ np.linalg.inv(Hs[j])
                cw.append(cv2.perspectiveTransform(c0, cm).reshape(2))
            est = np.array(cw)
            ref = gps[start:start + WIN + 1]
            if len(ref) != len(est):
                continue
            try:
                c, R, t = umeyama_sim(est, ref)
            except np.linalg.LinAlgError:
                continue
            al = (c * (R @ est.T)).T + t
            e = np.linalg.norm(al - ref, axis=1)
            vals.append(dict(start=start, rmse_m=float(np.sqrt(np.mean(e ** 2))),
                             final_m=float(e[-1]), scale_m_per_px=float(c)))
        if vals:
            rm = sorted(v["rmse_m"] for v in vals)
            drift_windows = dict(window=WIN, n_windows=len(vals),
                                 median_rmse_m=rm[len(rm) // 2],
                                 min_rmse_m=rm[0], max_rmse_m=rm[-1],
                                 median_scale=float(np.median([v["scale_m_per_px"] for v in vals])),
                                 per_window=vals)

    drift = None
    if gps is not None and len(centers) >= 3:
        est = np.array(centers)
        ref = gps[np.array(ok_idx)]
        c, R, t = umeyama_sim(est, ref)
        aligned = (c * (R @ est.T)).T + t
        err = np.linalg.norm(aligned - ref, axis=1)
        drift = dict(n_frames=int(len(err)), rmse_m=float(np.sqrt(np.mean(err ** 2))),
                     median_m=float(np.median(err)), final_m=float(err[-1]),
                     max_m=float(err.max()), scale_m_per_px=float(c))

    # MOSAIK PADA JENDELA TERBAIK, bukan rantai penuh. Perantaian naif divergen
    # untuk 3 dari 4 model -- TERMASUK LoFTR -- jadi mosaik rantai penuh bukan
    # pembanding yang adil, ia hanya menandai model mana yang beruntung tidak
    # punya satu pasangan katastrofik. Literatur mosaicking memang memakai
    # optimasi global (bundle adjustment), bukan rantai murni. Di sini dirender
    # jendela 10-bingkai dgn drift terkecil: sebanding antar-model dan terkondisi.
    if mosaic_path:
        best_w = None
        if drift_windows and drift_windows["per_window"]:
            best_w = min(drift_windows["per_window"], key=lambda v: v["rmse_m"])
        if best_w is not None:
            st, wn = best_w["start"], drift_windows["window"]
            try:
                render_mosaic(files[st:st + wn + 1], Hs[st:st + wn], mosaic_path)
                print(f"    (jendela terbaik: bingkai {st}-{st + wn}, "
                      f"drift {best_w['rmse_m']:.2f} m)")
            except Exception as e:
                print(f"    mosaik jendela gagal: {type(e).__name__}: {e}")
        elif chain_len >= 2:
            try:
                render_mosaic(files[:chain_len + 1], Hs[:chain_len], mosaic_path)
            except Exception as e:
                print(f"    mosaik gagal: {type(e).__name__}: {e}")

    return dict(flight=flight, contaminated=flight in CONTAMINATED, gap=gap,
                drift_windowed=drift_windows,
                n_frames=len(files), n_pairs=len(pairs),
                n_pairs_success=n_ok,
                success_rate=float(100.0 * n_ok / max(len(pairs), 1)),
                chain_length=chain_len, drift_vs_gps=drift, pairs=pairs)


def render_mosaic(files, Hs, out_path, scale=0.12):
    cum = [np.eye(3)]
    for H in Hs:
        cum.append(cum[-1] @ np.linalg.inv(H))
    S = np.diag([scale, scale, 1.0])
    cum = [S @ M @ np.linalg.inv(S) for M in cum]
    corners = []
    for f, M in zip(files, cum):
        im = _imread(f)
        h, w = im.shape[:2]
        h, w = int(h * scale), int(w * scale)
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, M).reshape(-1, 2))
    allc = np.vstack(corners)
    mn, mx = allc.min(0), allc.max(0)
    W, H_ = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
    if W * H_ > 60_000_000 or W <= 0 or H_ <= 0:
        raise RuntimeError(f"kanvas tidak wajar {W}x{H_} -- rantai kemungkinan divergen")
    T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]], np.float64)
    canvas = np.zeros((H_, W, 3), np.uint8)
    for f, M in zip(files, cum):
        im = _imread(f)
        im = cv2.resize(im, None, fx=scale, fy=scale)
        w = cv2.warpPerspective(im, T @ M, (W, H_))
        m = (w.sum(2) > 0)
        canvas[m] = w[m]
    cv2.imwrite(out_path, canvas)
    print(f"    mosaik -> {out_path} ({W}x{H_})")


def build_matchers(which):
    out = []
    specs = [
        ("baseline_p32", lambda: OurModel("baseline_p32", "p32",
                                          "aerial_stitching_fp16_best.pth", grid=(24, 32))),
        ("p16_fpn", lambda: OurModel("p16_fpn", "p16",
                                     "aerial_stitching_p16_fpn_qualitybest_val3.7186.pth",
                                     variant="p16_depth8_fpn", grid=(48, 64))),
        ("p16_nofpn", lambda: OurModel("p16_nofpn", "p16",
                                       best_nofpn_ckpt(), variant="p16_depth8", grid=(48, 64))),
        ("p16_depth6", lambda: OurModel("p16_depth6", "p16",
                                        "gate_arch_results/p16_depth6_pruned.pth",
                                        variant="p16_depth6", grid=(48, 64))),
        ("p16_depth4", lambda: OurModel("p16_depth4", "p16",
                                        "gate_arch_results/p16_depth4_pruned.pth",
                                        variant="p16_depth4", grid=(48, 64))),
        ("loftr", LoFTRModel),
        ("disk_lightglue", DiskLightGlueModel),
        # pihak ketiga -- lihat catatan proses di kelasnya (jangan campur dalam
        # satu proses: TopicFM dan JamMa sama-sama memakai paket `src`)
        ("topicfm_fast", lambda: TopicFMModel("fast")),
        ("topicfm_plus", lambda: TopicFMModel("plus")),
        ("jamma", JamMaModel),
    ]
    for name, fn in specs:
        if which != "all" and name not in which.split(","):
            continue
        try:
            out.append(fn())
            print(f"[model] {name} siap")
        except Exception as e:
            print(f"[model] {name} DILEWATI: {type(e).__name__}: {e}")
    return out


def best_nofpn_ckpt():
    """Checkpoint puncak HILIR dari kurva arsip; jatuh ke _best.pth bila
    kurvanya belum ada."""
    p = "gate_arch_results/collect_p16_curve.json"
    if os.path.exists(p):
        d = json.load(open(p))
        cand = [(v["out_saturated_inliers"], v["checkpoint"])
                for k, v in d["results"].items()
                if k.startswith("nofpn_ep") and "error" not in v]
        if cand:
            best = max(cand)[1]
            print(f"[nofpn] puncak hilir dari kurva: {os.path.basename(best)}")
            return best
    return "aerial_stitching_p16_nofpn_best.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="G Uji")
    ap.add_argument("--models", default="all")
    ap.add_argument("--refine", default="none", choices=list(REFINERS))
    ap.add_argument("--out", default="gate_arch_results/stitch")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--mosaic", action="store_true")
    ap.add_argument("--select", default=SELECT, choices=["topk", "bucket"])
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--gap", type=int, default=1, help="cocokkan bingkai i dgn i+gap")
    a = ap.parse_args()
    globals()["SELECT"] = a.select
    globals()["CAP"] = a.cap

    folder = FLIGHTS.get(a.flight, a.flight)
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg")))
    os.makedirs(a.out, exist_ok=True)
    tag = (f"{a.flight.replace(' ', '_')}__{a.refine}__{a.select}"
           f"{a.cap if a.cap != SAT_K else ''}{'__gap%d' % a.gap if a.gap != 1 else ''}")
    print(f"[FASE1+2] jalur={a.flight} ({len(files)} bingkai) refine={a.refine} "
          f"{'[TERKONTAMINASI]' if a.flight in CONTAMINATED else '[bersih]'}")

    refiner = REFINERS[a.refine]
    results = {}
    out_path = os.path.join(a.out, f"stitch__{tag}.json")
    for m in build_matchers(a.models):
        print(f"\n  == {m.label} ==", flush=True)
        mp = os.path.join(a.out, f"mosaic__{tag}__{m.label}.jpg") if a.mosaic else None
        try:
            results[m.label] = run_flight(m, a.flight, files, refiner, mp, a.max_frames, a.gap)
        except Exception as e:
            print(f"  {m.label} GAGAL: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[m.label] = dict(error=f"{type(e).__name__}: {e}")
        json.dump(dict(flight=a.flight, refine=a.refine, select=a.select,
                       cap=a.cap, gap=a.gap,
                       contaminated=a.flight in CONTAMINATED,
                       n_frames=len(files), results=results),
                  open(out_path, "w"), indent=2)   # tulis inkremental
        del m
        torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print(f"{'model':16} {'sukses':>8} {'rantai':>7} {'inlier':>8} {'rasio':>7} "
          f"{'hull':>6} {'rmse med':>9} {'drift w10':>11}")
    for k, v in results.items():
        if "error" in v:
            print(f"{k:16} GAGAL: {v['error'][:60]}")
            continue
        ok = [p for p in v["pairs"] if p.get("success")]
        inl = np.mean([p["n_inliers"] for p in ok]) if ok else 0
        rat = np.mean([p["inlier_ratio"] for p in ok]) if ok else 0
        hull = np.mean([p["hull_frac"] for p in ok]) if ok else 0
        rm = [p["holdout"]["median_px"] for p in ok if p.get("holdout")]
        dw = v.get("drift_windowed")
        dr = dw["median_rmse_m"] if dw else None
        print(f"{k:16} {v['success_rate']:7.1f}% {v['chain_length']:7d} {inl:8.1f} "
              f"{rat:6.1f}% {hull:6.2f} {np.mean(rm) if rm else float('nan'):9.2f} "
              f"{dr if dr is not None else float('nan'):11.2f}")
    print(f"\n[SELESAI] -> {out_path}")


if __name__ == "__main__":
    main()
