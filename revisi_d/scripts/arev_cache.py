"""Revisi A IJIES (T0) -- cache dekode sekali untuk pelatihan: ortho -> memmap .npy, foto pseudo-GT -> .npy 1024x768.

Pelatih lama mendekode SELURUH ortho JPEG (hingga 1,15 GP) pada setiap sampel, sehingga satu langkah latih ±0,73 s
dan dua run paralel saling berebut CPU. Cache ini menyimpan larik yang PERSIS sama dengan yang dihasilkan pelatih:
  ortho  : cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)                         (dataset_homography._load_ortho)
  pseudo : cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (1024, 768)) (dataset_grid_p16.PseudoGTSparseDataset)
Tiap berkas ditulis ke .tmp lalu rename; penanda .done memuat bentuk + sha256 berkas. Berkas yang penandanya sah
dilewati, jadi pekerjaan ini aman diulang sesudah putus daya.

Pakai: python revisi_d/scripts/arev_cache.py orthos|pseudo|all [--workers 4]
"""
import os
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))   # sama dengan dataset_homography.py
import argparse
import glob
import hashlib
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = AERIAL_ROOT + ""
CACHE = f"{ROOT}/cache_arev"
TRAIN = AERIAL_DATA + "/Data Disertasi/Data Latih Orthomosaic"
VAL = AERIAL_DATA + "/Data Disertasi/Data Validasi Orthomosaic"
PSEUDO_INDEX = f"{ROOT}/pseudo_gt_cache_p16/pseudo_gt_index.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def key_for(path):
    """Nama cache unik dan stabil dari jalur sumber (folder + basename)."""
    d = os.path.basename(os.path.dirname(path)).replace(" ", "_")
    return f"{d}__{os.path.basename(path)}.npy"


def ortho_files(dirs=None):
    out = []
    for d in (dirs or (TRAIN, VAL)):
        for pat in ("*.JPG", "*.jpeg", "*.jpg"):
            out += glob.glob(os.path.join(d, pat))
    return sorted(set(out))


def done_ok(dst):
    dn = dst + ".done"
    if not (os.path.exists(dst) and os.path.exists(dn)):
        return False
    try:
        info = json.load(open(dn))
        return os.path.getsize(dst) == info["bytes"]
    except Exception:  # noqa: BLE001
        return False


def write_npy(arr, dst, src):
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    info = dict(src=src, shape=list(arr.shape), dtype=str(arr.dtype), bytes=os.path.getsize(dst),
                sha256=sha256(dst), t=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(dst + ".done.tmp", "w") as f:
        json.dump(info, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(dst + ".done.tmp", dst + ".done")
    return info


def do_ortho(src):
    dst = f"{CACHE}/orthos/{key_for(src)}"
    if done_ok(dst):
        return src, "sudah"
    img = cv2.imread(src)
    if img is None:
        raise RuntimeError(f"gagal baca {src}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    info = write_npy(np.ascontiguousarray(img), dst, src)
    return src, f"{info['shape']}"


def do_pseudo(src):
    dst = f"{CACHE}/pseudo/{key_for(src)}"
    if done_ok(dst):
        return src, "sudah"
    img = cv2.imread(src)
    if img is None:
        raise RuntimeError(f"gagal baca {src}")
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1024, 768))
    info = write_npy(np.ascontiguousarray(img), dst, src)
    return src, f"{info['shape']}"


def run(fn, items, workers, label):
    n = len(items)
    with Pool(workers) as pool:
        for k, (src, msg) in enumerate(pool.imap_unordered(fn, items), 1):
            print(f"PROGRESS {k}/{n} {label} {os.path.basename(src)} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["orthos", "pseudo", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dirs", default=None, help="folder ortho (dipisah koma); bawaan = data latih/validasi privat")
    ap.add_argument("--index", default=PSEUDO_INDEX, help="indeks pseudo-GT yang fotonya di-cache")
    ap.add_argument("--tag", default=None, help="nama penanda selesai (bawaan = argumen what)")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    os.makedirs(f"{CACHE}/orthos", exist_ok=True)
    os.makedirs(f"{CACHE}/pseudo", exist_ok=True)
    if a.what in ("pseudo", "all"):
        ents = json.load(open(a.index))
        imgs = sorted({e["img1_path"] for e in ents} | {e["img2_path"] for e in ents})
        run(do_pseudo, imgs, max(a.workers, 6), "pseudo")
    if a.what in ("orthos", "all"):
        files = ortho_files(a.dirs.split(",") if a.dirs else None)
        # yang terbesar dulu supaya puncak RAM tidak menumpuk di akhir
        files.sort(key=lambda p: -os.path.getsize(p))
        run(do_ortho, files, a.workers, "ortho")
    with open(f"{CACHE}/{a.tag or a.what}.alldone", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    sys.exit(main())
