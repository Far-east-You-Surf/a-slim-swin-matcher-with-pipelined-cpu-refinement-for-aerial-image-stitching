"""Revisi A IJIES (paket T) -- pembungkus pelatihan untuk sapuan terkendali, tanpa menyunting pelatih asli.

Menjalankan `train_fp16_p16.py` APA ADANYA lewat runpy (pola `revisi_d/scripts/d1_train.py`); yang disuntikkan:
  * varian arsitektur: blok tahap 3 {8, 18}, lapis badan {0, 4, 8}, kisi {/16, /32 (avg-pool 2x2 sesudah tahap 3,
    tanpa parameter baru)}, resolusi internal {768x1024, 480x640 (antialias saat mengecilkan potongan latih)};
  * inisialisasi: 'pred' = warisan pendahulu /32 (seperti usulan), 'clean' = Swin-B ImageNet + pencocok acak;
  * benih (random/numpy/torch); saat lanjut dari epoch e, benih = benih*1000 + e (dicatat di log);
  * loader memmap (cache_arev/, isi identik bit demi bit dengan dekode JPEG lama; dicek tests/test_arev_loader.py);
  * titik simpan atomik: torch.save -> .tmp + fsync + rename; _last.pt lama disimpan sebagai _last.pt.prev, dan
    _last.pt yang rusak (putus daya) diganti .prev sebelum pelatih membacanya;
  * pseudo-GT: indeks 48x64 asli, atau indeks kisi lain hasil `arev_pseudo_kp.py` (--pseudo <index.json>), atau none.
Resep lain identik: AdamW 1e-4 konstan, batch 2 x akumulasi 8, 12 epoch x 2667 langkah, arsip tiap epoch.
Selesai = semua 12 arsip epoch ada -> penanda revisi_d/results/arev/train/<tag>.train.done (sha256 tiap arsip).

Pakai: python revisi_d/scripts/arev_train.py --tag d8_s2 --seed 2 [--blocks 8] [--layers 8] [--stride 16]
           [--res 768x1024] [--init pred] [--pseudo default|none|<index>] [--train-root DIR --val-root DIR]
           [--epochs 12] [--smoke N]
"""
import argparse
import hashlib
import json
import os
import random
import runpy
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
CACHE = f"{ROOT}/cache_arev"
DONE_DIR = f"{ROOT}/revisi_d/results/arev/train"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16, choices=[16, 32])
    ap.add_argument("--res", default="768x1024", choices=["768x1024", "480x640"])
    ap.add_argument("--init", default="pred", choices=["pred", "clean"])
    ap.add_argument("--pseudo", default="default")
    ap.add_argument("--train-root", default=None)
    ap.add_argument("--val-root", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    tag = f"arev_{a.tag}"
    os.environ.update(P16_TAG=tag, P16_ARCHIVE="1", P16_MAX_EPOCHS=str(a.epochs), P16_BATCH="2",
                      P16_WORKERS=os.environ.get("P16_WORKERS", "8"), P16_RESUME="auto", P16_VARIANT="arev")
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    cv2.setNumThreads(1)

    last = f"aerial_stitching_p16_{tag}_last.pt"
    # ---- titik simpan atomik + pemulihan _last.pt yang rusak -----------------------------------------------------
    _orig_save = torch.save

    def atomic_save(obj, f, *args, **kw):
        if not isinstance(f, (str, os.PathLike)):
            return _orig_save(obj, f, *args, **kw)
        f = os.fspath(f)
        tmp = f + ".tmp"
        _orig_save(obj, tmp, *args, **kw)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        if f.endswith("_last.pt") and os.path.exists(f):
            os.replace(f, f + ".prev")
        os.replace(tmp, f)
        dfd = os.open(os.path.dirname(os.path.abspath(f)), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    torch.save = atomic_save

    start_epoch = 0
    for cand in (last, last + ".prev"):
        if os.path.exists(cand):
            try:
                blob = torch.load(cand, map_location="cpu", weights_only=False)
                start_epoch = int(blob.get("epoch", 0))
                if cand != last:
                    os.replace(cand, last)
                    print(f"[arev] {last} rusak/hilang -> dipulihkan dari .prev (epoch {start_epoch})", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[arev] gagal memuat {cand}: {e!r}", flush=True)
                if cand == last:
                    os.replace(last, last + ".rusak")
    eff_seed = a.seed if start_epoch == 0 else a.seed * 1000 + start_epoch
    random.seed(eff_seed)
    np.random.seed(eff_seed % (2 ** 32))
    torch.manual_seed(eff_seed)
    torch.cuda.manual_seed_all(eff_seed)
    # Aliran sampel = fungsi benih SAJA (bukan konsumsi RNG inisialisasi model yang berbeda antar varian):
    # DataLoader latih (shuffle=True) diberi generator bertenaga benih efektif -> urutan, base_seed worker, dan semua
    # augmentasi (np.random/torch di worker) identik untuk semua varian dengan benih yang sama (desain berpasangan).
    import torch.utils.data as TUD
    _orig_dl_init = TUD.DataLoader.__init__

    def dl_init(self, *args, **kw):
        if kw.get("shuffle") and kw.get("generator") is None:
            kw["generator"] = torch.Generator().manual_seed(eff_seed)
        return _orig_dl_init(self, *args, **kw)
    TUD.DataLoader.__init__ = dl_init
    print(f"[arev] tag={tag} seed={a.seed} benih_efektif={eff_seed} mulai_epoch={start_epoch} "
          f"blocks={a.blocks} layers={a.layers} stride={a.stride} res={a.res} init={a.init} pseudo={a.pseudo}",
          flush=True)

    # ---- konfigurasi: lapis badan, akar data --------------------------------------------------------------------
    _orig_yaml = yaml.safe_load

    def safe_load(stream):
        cfg = _orig_yaml(stream)
        if isinstance(cfg, dict) and "model" in cfg and "dataset" in cfg:
            cfg["model"]["num_layers"] = a.layers
            if a.train_root:
                cfg["dataset"]["train_path"] = a.train_root
            if a.val_root:
                cfg["dataset"]["val_path"] = a.val_root
                cfg["dataset"]["val_exclude"] = []
            if a.pseudo == "none":
                cfg.setdefault("pseudo_gt", {})["enabled"] = False
        return cfg
    yaml.safe_load = safe_load

    # ---- kisi dan dataset memmap --------------------------------------------------------------------------------
    ih, iw = (int(x) for x in a.res.split("x"))
    grid = (ih // a.stride, iw // a.stride)
    import dataset_grid_p16 as DG
    DG.GRID_HW = grid

    sys.path.insert(0, f"{ROOT}/revisi_d/scripts")
    import arev_data
    arev_data.install()
    if a.pseudo not in ("default", "none"):
        _orig_mixed = DG.build_mixed_sparse

        def build_mixed_sparse(synthetic_dataset, index_path, mix_ratio, enabled=True):
            return _orig_mixed(synthetic_dataset, a.pseudo, mix_ratio, enabled)
        DG.build_mixed_sparse = build_mixed_sparse
    elif grid != (48, 64) and a.pseudo == "default":
        raise SystemExit(f"kisi {grid} butuh indeks pseudo-GT kisi itu (--pseudo <index>) atau --pseudo none")

    # ---- model varian --------------------------------------------------------------------------------------------
    import model_p16 as MP

    from arev_models import ArevModel

    if a.init == "clean":
        def no_warm(self, ckpt_path, verbose=True):
            own = self.state_dict()
            print(f"[warm-start] DILEWATI (init bersih): tulang punggung = Swin-B ImageNet, badan/kepala/proj acak "
                  f"(benih {a.seed}); {len(own)} tensor", flush=True)
            return dict(loaded=0, total=len(own), fresh_modules=["layers", "matching_head", "proj"],
                        init="clean_imagenet_backbone")
        ArevModel.load_pretrained_from = no_warm

    def build(variant="arev", **kw):
        kw.pop("use_fpn", None)
        return ArevModel(stage3_blocks=a.blocks, use_fpn=False, stride=a.stride, target_size=(ih, iw), **kw)
    MP.build = build

    # ---- jalankan pelatih asli -----------------------------------------------------------------------------------
    sys.argv = ["train_fp16_p16.py"] + (["--smoke", str(a.smoke)] if a.smoke else [])
    t0 = time.time()
    runpy.run_path(f"{ROOT}/train_fp16_p16.py", run_name="__main__")
    if a.smoke:
        os.makedirs(DONE_DIR, exist_ok=True)
        with open(f"{DONE_DIR}/{a.tag}.smoke.done", "w") as f:
            json.dump(dict(args=vars(a), grid=list(grid), seconds=round(time.time() - t0)), f)
        return
    arch = f"ckpt_archive_p16_{tag}"
    files = sorted(f for f in os.listdir(arch) if f.startswith("ep") and f.endswith(".pth")) if os.path.isdir(arch) else []
    got = {int(f[2:5]) for f in files}
    if not all(e in got for e in range(1, a.epochs + 1)):
        raise SystemExit(f"[arev] arsip belum lengkap: {sorted(got)}")
    os.makedirs(DONE_DIR, exist_ok=True)
    info = dict(tag=tag, args=vars(a), grid=list(grid), archive=arch,
                files={f: sha256(os.path.join(arch, f)) for f in files},
                meta=json.load(open(f"aerial_stitching_p16_{tag}_meta.json")),
                seconds_this_session=round(time.time() - t0), t=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = f"{DONE_DIR}/{a.tag}.train.done.tmp"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, f"{DONE_DIR}/{a.tag}.train.done")
    print(f"[arev] SELESAI {tag}: {len(files)} arsip", flush=True)


if __name__ == "__main__":
    main()
