"""Geometri dan metrik P39: MAGSAC++, gerbang v2, pose UseGeo/MegaDepth, AUC, galat kisi-H."""
import cv2
import numpy as np

from . import data as D

RANSAC_THR = 5.0
MIN_INL = 8
AREA_LO, AREA_HI = 0.2, 5.0


def magsac(a, b):
    """= tt_refiner.magsac, ditambah topeng inlier. Kembali (H, inl_mask) atau (None, None)."""
    if len(a) < 8:
        return None, None
    H, m = cv2.findHomography(a, b, cv2.USAC_MAGSAC, RANSAC_THR)
    if H is None:
        return None, None
    return H, (m.reshape(-1).astype(bool) if m is not None else np.zeros(len(a), bool))


def apply_h(H, a):
    return cv2.perspectiveTransform(np.asarray(a, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


# --- verbatim recompute_sanity.corners_ok / sanity
def corners_ok(H, w, h):
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    try:
        p = cv2.perspectiveTransform(c, np.asarray(H, np.float64)).reshape(-1, 2)
    except cv2.error:
        return False, None, None
    if not np.all(np.isfinite(p)):
        return False, None, None
    cross = []
    for i in range(4):
        a, b, cc = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append(np.cross(b - a, cc - b))
    signs = np.sign(cross)
    convex = bool(np.all(signs > 0) or np.all(signs < 0))
    area = float(abs(cv2.contourArea(p.astype(np.float32))))
    ratio = area / (w * h) if w * h else 0.0
    return convex, ratio, p


def sanity(H, w, h):
    if H is None:
        return dict(valid=False, reason="tanpa H")
    Hm = np.asarray(H, np.float64)
    det = float(np.linalg.det(Hm))
    if abs(det) < 1e-8:
        return dict(valid=False, reason="det~0", det=det)
    convex, ratio, _ = corners_ok(Hm, w, h)
    if ratio is None:
        return dict(valid=False, reason="warp gagal", det=det)
    ok = convex and (AREA_LO < ratio < AREA_HI)
    return dict(valid=bool(ok), det=det, convex=convex, warp_area_ratio=ratio)


def gates(a, b, ow, oh):
    """H akhir = MAGSAC++ atas titik keluaran. Kembali (H, n_inl, gate_sanity, gate_v2)."""
    H, inl = magsac(a, b)
    if H is None:
        return None, 0, False, False
    n_inl = int(inl.sum())
    s = bool(sanity(H, ow, oh)["valid"])
    return H, n_inl, s, bool(s and n_inl >= MIN_INL)


# ------------------------------------------------------------------ pose
def pose_usegeo(a, b, A, B):
    """Persis P35: findEssentialMat USAC_MAGSAC (prob 0,9999, 1 px) + recoverPose; gagal = 180."""
    pe = 180.0
    if len(a) >= 6:
        E, em = cv2.findEssentialMat(a, b, D.K, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=1.0)
        if E is not None and E.shape == (3, 3):
            _, R_e, t_e, _ = cv2.recoverPose(E, a, b, D.K, mask=em)
            re, te = D.pose_error(R_e, t_e, *D.gt_relpose(A, B))
            pe = float(max(re, te))
    return pe


def md_angle_err(R, t, R_gt, t_gt):
    """Verbatim p15_megadepth.angle_err."""
    cos_r = np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1)
    e_r = np.degrees(np.arccos(cos_r))
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    e_t = np.degrees(np.arccos(np.clip(np.dot(t.ravel(), t_gt.ravel()) / max(n, 1e-12), -1, 1)))
    e_t = min(e_t, 180 - e_t)
    return e_r, e_t


def md_estimate_pose(k0, k1, K0, K1, thresh=0.5, conf=0.99999):
    """Verbatim p15_megadepth.estimate_pose."""
    if len(k0) < 5:
        return None
    f = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    n0 = (k0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    n1 = (k1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), threshold=thresh / f, prob=conf, method=cv2.RANSAC)
    if E is None:
        return None
    best = None
    for e in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(e, n0, n1, np.eye(3), 1e9, mask=mask)
        if best is None or n > best[0]:
            best = (n, R, t[:, 0], mask.ravel() > 0)
    return best


def pose_md(a, b, p):
    """Seperti P38 run_md: setRNGSeed(0) sebelum tiap estimasi; gagal = 180."""
    cv2.setRNGSeed(0)
    est = md_estimate_pose(a, b, p["K0"], p["K1"]) if len(a) >= 5 else None
    if est is None:
        return 180.0
    er, et = md_angle_err(est[1], est[2], p["R"], p["t"])
    return float(max(er, et))


# ------------------------------------------------------------------ AUC
def auc_step(errs, t):
    """AUC tangga P39 = 100 * rerata max(0, 1 - e/t)."""
    e = np.asarray(errs, float)
    return float(100.0 * np.mean(np.clip(1.0 - e / t, 0.0, None))) if len(e) else float("nan")


def auc_trap(errs, t=5.0):
    """Verbatim p35_tt_prereg.auc (trapesium P35)."""
    e = np.sort(np.asarray(errs, float)); n = len(e)
    rec = np.arange(1, n + 1) / n
    e = np.r_[0.0, e]; rec = np.r_[0.0, rec]
    k = np.searchsorted(e, t)
    x = np.r_[e[:k], t]; y = np.r_[rec[:k], rec[k - 1]]
    return float(np.trapezoid(y, x) / t * 100)


# ------------------------------------------------------------------ galat kisi-H (UseGeo)
def hgrid_points(w=D.IMG_W, h=D.IMG_H, nx=32, ny=24):
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * w / nx, (np.arange(ny) + 0.5) * h / ny)
    return np.stack([gx.ravel(), gy.ravel()], 1)


class HGridGT:
    """GT LiDAR pada titik kisi untuk satu pasangan; galat median |H(x) - GT(x)| atas titik sah di dalam citra 2."""
    def __init__(self, A, B, depthA):
        self.x = hgrid_points()
        ub, vb, valid = D.gt_project(A, B, depthA, self.x[:, 0], self.x[:, 1])
        inb = valid & (ub >= 0) & (ub < D.IMG_W) & (vb >= 0) & (vb < D.IMG_H)
        self.x, self.gt = self.x[inb], np.stack([ub[inb], vb[inb]], 1)

    def err(self, H):
        if H is None or len(self.x) == 0:
            return float("inf")
        p = apply_h(H, self.x)
        e = np.hypot(*(p - self.gt).T)
        return float(np.median(e)) if np.all(np.isfinite(e)) else float("inf")

    def oracle(self):
        if len(self.x) < 8:
            return None
        H, _ = cv2.findHomography(self.x, self.gt, 0)
        return H
