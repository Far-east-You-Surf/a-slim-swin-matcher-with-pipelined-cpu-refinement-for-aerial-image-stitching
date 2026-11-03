"""Data publik P39: UseGeo (GT LiDAR), MegaDepth-1500 (protokol letterbox P27), ODM Seneca/Bellus, urutan throughput.
Fungsi UseGeo disalin verbatim dari usegeo_gt_eval.py (modul aslinya chdir + mengimpor stitch_demo/torch)."""
import os
from collections import OrderedDict

import cv2
import numpy as np

from . import USEGEO, MD, LOWTEX

# ------------------------------------------------------------------ UseGeo (verbatim usegeo_gt_eval.py:27-207)
SCALE = 4.0                                  # _res.jpg = 7953x5279 / 4
C = 4640.301 / SCALE
X0P = 3969.320 / SCALE
Y0P = -2600.828 / SCALE
IMG_W, IMG_H = 1989, 1320
DKAPPA_MAX = 15.0
STEP_MAX = 60.0
POSE_FILE = {1: 'Dataset-1/Camera_Inputs/eors.txt',
             2: 'Dataset-2/Image_orientations_dataset2.xyz',
             3: 'Dataset-3/Camera_Inputs/eors.txt'}
K = np.array([[C, 0.0, X0P], [0.0, C, -Y0P], [0.0, 0.0, 1.0]])
S_CV = np.diag([1.0, -1.0, -1.0])


def load_poses(ds):
    rows = []
    for line in open(os.path.join(USEGEO, POSE_FILE[ds])):
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        try:
            rows.append((p[0],) + tuple(map(float, p[1:7])))
        except ValueError:
            continue
    rows.sort(key=lambda r: r[0])
    return rows


def rot(om, ph, ka):
    om, ph, ka = np.deg2rad([om, ph, ka])
    Rx = np.array([[1, 0, 0], [0, np.cos(om), -np.sin(om)], [0, np.sin(om), np.cos(om)]])
    Ry = np.array([[np.cos(ph), 0, np.sin(ph)], [0, 1, 0], [-np.sin(ph), 0, np.cos(ph)]])
    Rz = np.array([[np.cos(ka), -np.sin(ka), 0], [np.sin(ka), np.cos(ka), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def dkappa(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def gt_project(A, B, depthA, u, v):
    h, w = depthA.shape
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    d = depthA[vi, ui]
    dirs = np.stack([u - X0P, -v - Y0P, -C * np.ones_like(u)], 1)
    pc = dirs * (d / np.linalg.norm(dirs, axis=1))[:, None]
    Xw = np.array(A[1:4]) + (rot(*A[4:7]) @ pc.T).T
    pb = (rot(*B[4:7]).T @ (Xw - np.array(B[1:4])).T).T
    infront = pb[:, 2] < 0
    with np.errstate(divide='ignore', invalid='ignore'):
        xb = -C * pb[:, 0] / pb[:, 2]
        yb = -C * pb[:, 1] / pb[:, 2]
    ub = xb + X0P
    vb = -(yb + Y0P)
    valid = infront & np.isfinite(ub) & np.isfinite(vb) & np.isfinite(d) & (d > 0)
    return ub, vb, valid


def gt_relpose(A, B):
    Ra = rot(*A[4:7]) @ S_CV
    Rb = rot(*B[4:7]) @ S_CV
    R_rel = Rb.T @ Ra
    t_rel = Rb.T @ (np.array(A[1:4]) - np.array(B[1:4]))
    return R_rel, t_rel


def pose_error(R_est, t_est, R_gt, t_gt):
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    r_err = float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if n < 1e-12:
        return r_err, 180.0
    c = float(np.clip(np.dot(t_est.ravel(), t_gt.ravel()) / n, -1.0, 1.0))
    t_err = float(np.rad2deg(np.arccos(c)))
    t_err = min(t_err, 180.0 - t_err)
    return r_err, t_err


class DepthCache:
    def __init__(self, n=4):
        self.n, self.d = n, {}

    def get(self, ds, name):
        key = (ds, name)
        if key not in self.d:
            if len(self.d) >= self.n:
                self.d.pop(next(iter(self.d)))
            p = os.path.join(USEGEO, 'Dataset-%d' % ds, 'Depth_resized', 'depth_maps',
                             name.replace('.jpg', '_depth_res.tiff'))
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError('depth tidak terbaca: ' + p)
            self.d[key] = im.astype(np.float64)
        return self.d[key]


def img_path(ds, name):
    return os.path.join(USEGEO, 'Dataset-%d' % ds, 'Depth_resized', 'undistorted_images',
                        name.replace('.jpg', '_res.jpg'))


def along_pairs(d, datasets=(1, 2, 3)):
    """Verbatim e13_refine.along_pairs: (ds, A, B) sepanjang jalur, urutan sama dengan P35."""
    out = []
    for ds in datasets:
        rows = load_poses(ds)
        for i in range(len(rows) - d):
            ok = all(dkappa(rows[k][6], rows[k + 1][6]) < DKAPPA_MAX and
                     np.hypot(rows[k][1] - rows[k + 1][1], rows[k][2] - rows[k + 1][2]) < STEP_MAX
                     for k in range(i, i + d))
            if ok:
                out.append((ds, rows[i], rows[i + d]))
    return out


def turn_pairs(d, datasets=(1, 2, 3)):
    """Pasangan (i, i+d) yang TIDAK sepanjang jalur (evaltool.usegeo_pairs along=False): strata belokan."""
    out = []
    for ds in datasets:
        rows = load_poses(ds)
        for i in range(len(rows) - d):
            ok = all(dkappa(rows[k][6], rows[k + 1][6]) < DKAPPA_MAX and
                     np.hypot(rows[k][1] - rows[k + 1][1], rows[k][2] - rows[k + 1][2]) < STEP_MAX
                     for k in range(i, i + d))
            if not ok:
                out.append((ds, rows[i], rows[i + d]))
    return out


def segments(ds):
    """Nama bingkai -> (id segmen jalur, id blok 10 bingkai dalam segmen). Segmen = deret bingkai berurutan yang tiap
    langkahnya lolos kriteria sepanjang jalur; id unik lintas set: ds*1000 + k."""
    rows = load_poses(ds)
    seg, blk, k, pos = {}, {}, 0, 0
    for i, r in enumerate(rows):
        if i > 0:
            ok = (dkappa(rows[i - 1][6], r[6]) < DKAPPA_MAX and
                  np.hypot(rows[i - 1][1] - r[1], rows[i - 1][2] - r[2]) < STEP_MAX)
            if ok:
                pos += 1
            else:
                k += 1
                pos = 0
        seg[r[0]] = ds * 1000 + k
        blk[r[0]] = (ds * 1000 + k) * 100 + pos // 10
    return seg, blk


# ------------------------------------------------------------------ MegaDepth-1500 (verbatim p15_megadepth, p27 lb)
def md_pairs():
    P = []
    for i, line in enumerate(open(f"{MD}/pairs_calibrated.txt")):
        v = line.split()
        if len(v) < 32:
            continue
        f = np.array(v[2:], float)
        P.append(dict(i=len(P), n0=v[0], n1=v[1], scene=v[0].split('/')[0], K0=f[0:9].reshape(3, 3),
                      K1=f[9:18].reshape(3, 3), R=f[18:27].reshape(3, 3), t=f[27:30]))
    return P


def md_lb(name, W=640, H=480):
    """Citra letterbox 640x480 dari tembolok P27 (baca saja): (path png, skala s, (h, w) asli)."""
    dst = f"{MD}/cache_lb{W}x{H}/{name.replace('/', '__')}.png"
    if not os.path.exists(dst + ".s"):
        raise RuntimeError(f"tembolok letterbox tidak ada: {dst}")
    return dst, float(open(dst + ".s").read()), tuple(map(int, open(dst + ".hw").read().split()))


def md_image(name):
    return f"{MD}/images/{name}"


# ------------------------------------------------------------------ ODM tekstur rendah
def lowtex_files(seq):
    d = LOWTEX[seq]
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg")))


def lowtex_pairs(seq, d):
    """Sama dengan evaltool.flight_pairs: (i, i+d) dalam urutan nama berkas."""
    fs = lowtex_files(seq)
    return [(i, fs[i], fs[i + d]) for i in range(len(fs) - d)]


def gps_xy(path):
    """(timur, utara) meter relatif ke titik acuan tetap (ekuirektangular lokal) + track; dari EXIF GPS."""
    from PIL import Image
    ex = Image.open(path).getexif().get_ifd(0x8825)

    def deg(v, ref):
        x = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -x if ref in ("S", "W") else x
    lat, lon = deg(ex[2], ex[1]), deg(ex[4], ex[3])
    return lat, lon, float(ex[15]) if 15 in ex else np.nan      # tag GPS 15 = GPSTrack


def gps_delta_m(p1, p2):
    la1, lo1, _ = gps_xy(p1)
    la2, lo2, _ = gps_xy(p2)
    R = 6371008.8
    x = np.deg2rad(lo2 - lo1) * np.cos(np.deg2rad(0.5 * (la1 + la2))) * R
    y = np.deg2rad(la2 - la1) * R
    return float(np.hypot(x, y))


# ------------------------------------------------------------------ citra abu-abu (satu jalur untuk semua sel)
def load_gray(path):
    """= e13_refine.gray (stitch_demo._imread + BGR2GRAY)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if im is None:
        raise RuntimeError(f"gagal membaca {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


class GrayCache:
    """LRU kecil: pasangan berurutan berbagi bingkai."""
    def __init__(self, n=6):
        self.n, self.d = n, OrderedDict()

    def get(self, path):
        if path in self.d:
            self.d.move_to_end(path)
            return self.d[path]
        g = load_gray(path)
        self.d[path] = g
        if len(self.d) > self.n:
            self.d.popitem(last=False)
        return g
