"""Sapuan selubung registrasi UseGeo dengan KEBENARAN-DASAR geometris (rencana pra-registrasi Sec. 95.6).

Mengimpor `stitch_demo` TANPA mengubahnya (pola `mosaic_full.py`, Sec. 93): matcher, ambang
RANSAC, gerbang kewarasan homografi, dan cap match semuanya dipakai apa adanya supaya angkanya
sebanding dengan Sec. 61/66/68.

Yang BARU di sini dan tidak ada di jalur mana pun sebelumnya: setiap kecocokan yang dikembalikan
`match()` diadu dengan GT LiDAR+pose UseGeo (rantai diverifikasi ujung-ke-ujung di Sec. 95.3,
median 0,40-0,75 px). Jadi kita tidak lagi bertanya "apakah homografinya waras" (gerbang buatan
sendiri, cacatnya di Sec. 82.2) melainkan "berapa kecocokan yang BENAR-BENAR benar".

Konvensi GT -- DIPUTUSKAN DATA di Sec. 95.3, jangan diubah tanpa menjalankan ulang usegeo_gt_check.py:
  * depth = JARAK Euklidean sepanjang sinar (bukan z-depth; selisihnya 21 px sistematis di v)
  * arah piksel kamera = (u - x0, -v - y0, -c)   [x kanan, y atas, z ke BELAKANG]
  * R = Rx(omega) Ry(phi) Rz(kappa) memetakan KAMERA -> DUNIA, X = X0 + R x_cam
  * c = 4640,301 untuk KETIGA dataset (angka 4738 di PDF Metashape milik bingkai MENTAH)
  * citra _res.jpg = resolusi penuh / 4, jadi intrinsik dibagi 4 juga
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.dirname(_os.path.abspath(__file__))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

sys.path.insert(0, AERIAL_ROOT + '')
os.chdir(AERIAL_ROOT + '')
import stitch_demo as sd

USEGEO = AERIAL_DATA + '/data_tambahan/usegeo'
SCALE = 4.0                                  # _res.jpg = 7953x5279 / 4
C = 4640.301 / SCALE
X0P = 3969.320 / SCALE
Y0P = -2600.828 / SCALE
IMG_W, IMG_H = 1989, 1320

# --- kriteria sepanjang-jalur (Sec. 95.6): SETIAP langkah antara harus memenuhi keduanya
DKAPPA_MAX = 15.0                            # derajat; >=15 = belokan -> rezim antar-jalur
STEP_MAX = 60.0                              # meter

POSE_FILE = {1: 'Dataset-1/Camera_Inputs/eors.txt',
             2: 'Dataset-2/Image_orientations_dataset2.xyz',
             3: 'Dataset-3/Camera_Inputs/eors.txt'}

# Matriks kamera konvensi OpenCV (z ke DEPAN, y ke BAWAH) untuk matriks esensial.
# Turunannya: xn_cv = (u-X0P)/C dan yn_cv = (v+Y0P)/C -- lihat catatan di bawah `gt_relpose`.
K = np.array([[C, 0.0, X0P], [0.0, C, -Y0P], [0.0, 0.0, 1.0]])
S_CV = np.diag([1.0, -1.0, -1.0])            # kamera fotogrametri <-> kamera OpenCV


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
        except ValueError:                   # baris header non-numerik
            continue
    rows.sort(key=lambda r: r[0])            # nama = stempel waktu -> urutan abjad = urutan terbang
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
    """Unproject piksel (u,v) foto A lewat depth+pose A, proyeksikan ke piksel foto B.

    Mengembalikan (ub, vb, valid) -- valid=False bila titiknya jatuh di BELAKANG kamera B
    (pb_z >= 0 dalam konvensi fotogrametri z-ke-belakang) atau depth-nya tidak masuk akal.
    """
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
    """Pose relatif A->B dalam konvensi OpenCV: x_b = R x_a + t (t skala metrik).

    R_cv = R_fotogrametri @ diag(1,-1,-1) memetakan kamera-OpenCV -> dunia, karena
    pc = diag(1,-1,-1) p_cv. Maka R_rel = R_cv_B^T R_cv_A dan t = R_cv_B^T (X0_A - X0_B).
    """
    Ra = rot(*A[4:7]) @ S_CV
    Rb = rot(*B[4:7]) @ S_CV
    R_rel = Rb.T @ Ra
    t_rel = Rb.T @ (np.array(A[1:4]) - np.array(B[1:4]))
    return R_rel, t_rel


def pose_error(R_est, t_est, R_gt, t_gt):
    """Galat sudut rotasi & translasi (derajat), konvensi identik LoFTR/SuperGlue."""
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    r_err = float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if n < 1e-12:
        return r_err, 180.0
    c = float(np.clip(np.dot(t_est.ravel(), t_gt.ravel()) / n, -1.0, 1.0))
    t_err = float(np.rad2deg(np.arccos(c)))
    t_err = min(t_err, 180.0 - t_err)        # E punya ambiguitas tanda
    return r_err, t_err


def score_pair(a, b, ow, oh, A, B, depthA, want_pose):
    """Semua metrik satu pasangan satu matcher. `a`,`b` di bingkai piksel citra ASLI."""
    r = {'n_matches': int(len(a))}
    if len(a) < sd.MIN_MATCHES:
        r.update(gate=False, reason='match < minimum')
        return r

    # --- (a) GERBANG LAMA Sec. 61, persis seperti stitch_demo.eval_pair
    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, sd.RANSAC_THR)
    if H is None or mask is None:
        r.update(gate=False, reason='homografi gagal', n_inliers=0)
    else:
        mask = mask.reshape(-1).astype(bool)
        inl = int(mask.sum())
        san = sd.h_sanity(H, ow, oh)
        r['n_inliers'] = inl
        r['inlier_ratio'] = float(100.0 * inl / len(a))
        r['gate'] = bool(san['valid'] and inl >= sd.MIN_MATCHES)
        r['warp_area_ratio'] = san.get('warp_area_ratio')

    # --- (b) GALAT REPROYEKSI GT tiap kecocokan
    sx, sy = IMG_W / float(ow), IMG_H / float(oh)
    ua, va = a[:, 0] * sx, a[:, 1] * sy
    ub_gt, vb_gt, valid = gt_project(A, B, depthA, ua, va)
    err = np.hypot(ub_gt - b[:, 0] * sx, vb_gt - b[:, 1] * sy)
    err_v = err[valid]
    r['n_gt_valid'] = int(valid.sum())
    if len(err_v) == 0:
        r.update(med_err=None, gate_gt=False, n_correct2=0)
        return r
    thr = {'1': 0.01 * IMG_W, '2': 0.02 * IMG_W, '5': 0.05 * IMG_W}   # 19,89 / 39,78 / 99,45 px
    r['med_err'] = float(np.median(err_v))
    r['p90_err'] = float(np.percentile(err_v, 90))
    for k, t in thr.items():
        r['frac' + k] = float((err_v < t).mean())          # penyebut = kecocokan ber-GT sah
        r['n_correct' + k] = int((err_v < t).sum())
    # cakupan yang TIDAK bergantung homografi (medan berelief -> H bukan model GT yang sah)
    r['gate_gt'] = bool(r['n_correct2'] >= sd.MIN_MATCHES)
    # apakah GT-nya jatuh di dalam bingkai B sama sekali (memisahkan "salah lokasi" dari
    # "titiknya memang tidak terlihat di B" -- keduanya kesalahan, tapi kesalahan BEDA)
    inb = valid & (ub_gt >= 0) & (ub_gt < IMG_W) & (vb_gt >= 0) & (vb_gt < IMG_H)
    r['frac_inbounds'] = float(inb.sum() / max(1, valid.sum()))
    if inb.sum() >= 1:
        r['med_err_inbounds'] = float(np.median(err[inb]))

    # --- (c) AUC POSE RELATIF (standar literatur, intrinsik diketahui)
    if want_pose and len(a) >= 6:
        pa = np.stack([ua, va], 1)
        pb = np.stack([b[:, 0] * sx, b[:, 1] * sy], 1)
        E, em = cv2.findEssentialMat(pa, pb, K, method=cv2.USAC_MAGSAC,
                                     prob=0.9999, threshold=1.0)
        if E is not None and E.shape == (3, 3):
            _, R_e, t_e, _ = cv2.recoverPose(E, pa, pb, K, mask=em)
            R_g, t_g = gt_relpose(A, B)
            re, te = pose_error(R_e, t_e, R_g, t_g)
            r['rot_err'] = re
            r['t_err'] = te
            r['pose_err'] = max(re, te)
    return r


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matchers', default='baseline_p32,p16_nofpn,loftr,disk_lightglue')
    ap.add_argument('--datasets', default='1,2,3')
    ap.add_argument('--deltas', default='1,2,3,4,5')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--limit', type=int, default=0, help='batasi pasangan per (dataset,delta) utk uji cepat')
    ap.add_argument('--no-pose', action='store_true')
    a = ap.parse_args()

    datasets = [int(x) for x in a.datasets.split(',')]
    deltas = [int(x) for x in a.deltas.split(',')]
    out = a.out or 'gate_arch_results/usegeo_gt_eval_%s.json' % a.tag
    jsonl = out.replace('.json', '.jsonl')
    os.makedirs('gate_arch_results', exist_ok=True)

    matchers = sd.build_matchers(a.matchers)
    if not matchers:
        print('tidak ada matcher terbangun'); return
    cache = DepthCache()
    recs = []
    fj = open(jsonl, 'a')
    t_start = time.time()

    for ds in datasets:
        rows = load_poses(ds)
        print('\n=== Dataset-%d: %d bingkai ===' % (ds, len(rows)), flush=True)
        for delta in deltas:
            todo = []
            for i in range(len(rows) - delta):
                steps = [(dkappa(rows[k][6], rows[k + 1][6]),
                          float(np.hypot(rows[k][1] - rows[k + 1][1], rows[k][2] - rows[k + 1][2])))
                         for k in range(i, i + delta)]
                along = all(dk < DKAPPA_MAX and st < STEP_MAX for dk, st in steps)
                todo.append((i, i + delta, along,
                             max(dk for dk, _ in steps), sum(st for _, st in steps)))
            if a.limit:
                todo = todo[:a.limit]
            n_al = sum(1 for t in todo if t[2])
            print('  D=%d: %d pasangan (%d sepanjang-jalur, %d belokan)'
                  % (delta, len(todo), n_al, len(todo) - n_al), flush=True)
            for pi, (i, j, along, dkmax, dist) in enumerate(todo):
                A, B = rows[i], rows[j]
                try:
                    depthA = cache.get(ds, A[0])
                except RuntimeError as e:
                    print('    LEWAT %s: %s' % (A[0], e)); continue
                pA, pB = img_path(ds, A[0]), img_path(ds, B[0])
                for m in matchers:
                    t0 = time.perf_counter()
                    try:
                        mk = m.match(pA, pB)
                    except Exception as e:
                        recs.append(dict(dataset=ds, delta=delta, i=i, matcher=m.label,
                                         along=along, error='%s: %s' % (type(e).__name__, e)))
                        continue
                    ma, mb, ow, oh = mk
                    dt = (time.perf_counter() - t0) * 1e3
                    r = score_pair(np.asarray(ma, float), np.asarray(mb, float), ow, oh,
                                   A, B, depthA, not a.no_pose)
                    r.update(dataset=ds, delta=delta, i=i, matcher=m.label, along=along,
                             frame_a=A[0], frame_b=B[0], dkappa_max=round(dkmax, 2),
                             path_len=round(dist, 2), ms=round(dt, 1))
                    recs.append(r)
                    fj.write(json.dumps(r) + '\n')
                if pi % 25 == 0:
                    fj.flush()
                    el = time.time() - t_start
                    print('    [%3d/%3d] %s  %s' % (pi + 1, len(todo), A[0],
                          '  '.join('%s g=%d n2=%s' % (r['matcher'], r.get('gate', False),
                                                       r.get('n_correct2', '-'))
                                    for r in recs[-len(matchers):])), flush=True)
                    print('        (elapsed %.1f mnt)' % (el / 60.0), flush=True)
            json.dump(recs, open(out, 'w'))
    fj.close()
    json.dump(recs, open(out, 'w'))
    print('\nselesai: %d catatan -> %s  (%.1f mnt)'
          % (len(recs), out, (time.time() - t_start) / 60.0))


if __name__ == '__main__':
    main()
