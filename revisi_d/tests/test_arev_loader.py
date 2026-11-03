"""Uji kesetaraan loader ber-cache (revisi A IJIES, T0).

1. Isi: setiap memmap ortho == dekode JPEG segar (cv2.imread + BGR2RGB), setiap foto pseudo-GT ber-cache ==
   cv2.resize(BGR2RGB(imread), (1024, 768)).
2. Sampel: untuk RNG yang sama (np.random + torch), dataset asli dan dataset ber-cache menghasilkan tensor identik
   (torch.equal) -- K sampel sintetis dan K sampel pseudo-GT.
Keluaran: revisi_d/results/arev/t0/loader_equiv.json; penanda .done hanya bila semua lolos.
Pakai: python revisi_d/tests/test_arev_loader.py [K]
"""
import os
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))
import json
import sys
import time

ROOT = AERIAL_ROOT + ""
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/revisi_d/scripts")
import cv2
import numpy as np
import torch
import yaml

import arev_data as AD
from arev_cache import ortho_files, PSEUDO_INDEX

OUT = f"{ROOT}/revisi_d/results/arev/t0/loader_equiv.json"


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cv2.setNumThreads(4)
    res = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), K=K)
    # 1. isi cache
    bad = []
    files = ortho_files()
    for k, p in enumerate(files, 1):
        mm = np.load(f"{AD.CACHE}/orthos/{AD.cache_key(p)}", mmap_mode="r")
        ref = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        if mm.shape != ref.shape or not np.array_equal(mm, ref):
            bad.append(p)
        print(f"PROGRESS {k}/{len(files) + 2 * K} isi ortho {os.path.basename(p)} {'OK' if p not in bad else 'BEDA'}",
              flush=True)
        del ref
    ents = json.load(open(PSEUDO_INDEX))
    imgs = sorted({e["img1_path"] for e in ents} | {e["img2_path"] for e in ents})
    bad_p = []
    for p in imgs:
        c = np.load(f"{AD.CACHE}/pseudo/{AD.cache_key(p)}")
        ref = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (1024, 768))
        if c.shape != ref.shape or not np.array_equal(c, ref):
            bad_p.append(p)
    res.update(ortho_files=len(files), ortho_mismatch=bad, pseudo_files=len(imgs), pseudo_mismatch=bad_p)
    # 2. sampel
    cfg = yaml.safe_load(open("config.yaml"))
    d = cfg["dataset"]
    kw = dict(data_path=d["train_path"], epoch_size=K,
              extra_files=[os.path.join(d["val_path"], n) for n in d["val_exclude"]], max_cache_images=1,
              grid_hw=(48, 64))
    old, new = AD.OrigOrtho(**kw), AD.MemmapOrtho(**kw)
    assert old.image_files == new.image_files
    syn_bad = 0
    for k in range(K):
        outs = []
        for ds in (old, new):
            np.random.seed(7000 + k)
            torch.manual_seed(7000 + k)
            outs.append(ds[k])
        same = all(torch.equal(x, y) for x, y in zip(*outs))
        syn_bad += 0 if same else 1
        print(f"PROGRESS {len(files) + k + 1}/{len(files) + 2 * K} sampel sintetis {k} {'OK' if same else 'BEDA'}",
              flush=True)
    po, pn = AD.OrigPseudo(PSEUDO_INDEX, epoch_size=K), AD.CachedPseudo(PSEUDO_INDEX, epoch_size=K)
    ps_bad = 0
    for k in range(K):
        outs = []
        for ds in (po, pn):
            np.random.seed(9000 + k)
            torch.manual_seed(9000 + k)
            outs.append(ds[k])
        same = all(torch.equal(x, y) for x, y in zip(*outs))
        ps_bad += 0 if same else 1
        print(f"PROGRESS {len(files) + K + k + 1}/{len(files) + 2 * K} sampel pseudo {k} {'OK' if same else 'BEDA'}",
              flush=True)
    res.update(synthetic_mismatch=syn_bad, pseudo_sample_mismatch=ps_bad)
    res["pass"] = not bad and not bad_p and syn_bad == 0 and ps_bad == 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res)[:600], flush=True)
    if res["pass"]:
        with open(OUT + ".done", "w") as f:
            f.write(res["t"])
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
