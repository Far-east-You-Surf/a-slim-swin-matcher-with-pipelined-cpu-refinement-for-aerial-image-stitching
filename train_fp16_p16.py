"""
Pelatihan arsitektur Artikel D: pencocokan grid /16 + fusi konteks /32 (model_p16.py).

Beda dari `train_fp16_dataonly.py` (Model A), semuanya WAJIB bukan pilihan:
  1. Model `AerialStitchingModelP16` (grid 3072 token, stage 3 dipangkas 8 blok, fusi FPN),
     dihangatkan dari `aerial_stitching_fp16_best.pth` -- hanya `proj`/`lateral` yang acak.
  2. Label JARANG + `compute_loss_sparse` (matriks GT padat = 37.7 MB/sampel pada grid ini).
  3. Attention matcher lewat SDPA (`flash=True`): attention 3072 token yang dimaterialisasi
     butuh ~9.7 GB hanya untuk backward, tidak muat di 16 GB. Graf ekspornya tetap identik
     (§20), jadi ini tidak mengubah engine yang nanti dibangun.
  4. Pseudo-GT dibaca dari `pseudo_gt_cache_p16/` (regen_pseudo_gt_p16.py), bukan cache lama
     yang indeksnya dipanggang pada grid 24x32.

Pakai:  python train_fp16_p16.py [--smoke N]
"""

import json
import os
import sys
import time

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

import model_p16
from dataset_grid_p16 import (GRID_HW, HomographyOrthoDatasetP16, build_mixed_sparse,
                              compute_loss_sparse)

# Varian & penamaan keluaran diparameterkan supaya ABLASI memakai JALUR KODE YANG SAMA PERSIS
# dgn run utama -- satu-satunya yang berbeda hanyalah preset arsitekturnya.
VARIANT = os.environ.get("P16_VARIANT", "p16_depth8_fpn")
TAG = os.environ.get("P16_TAG", "fpn" if VARIANT.endswith("_fpn") else "nofpn")
CKPT_WARM = "aerial_stitching_fp16_best.pth"
CKPT_OUT = f"aerial_stitching_p16_{TAG}_best.pth"
CKPT_LAST = f"aerial_stitching_p16_{TAG}_last.pt"   # state lengkap utk melanjutkan
META_OUT = f"aerial_stitching_p16_{TAG}_meta.json"
ARCHIVE_DIR = f"ckpt_archive_p16_{TAG}"             # P16_ARCHIVE=1 -> simpan tiap epoch
PSEUDO_INDEX = "pseudo_gt_cache_p16/pseudo_gt_index.json"
REPORT = f"training_report_p16_{TAG}.md"


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for img1, img2, gt in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss = compute_loss_sparse(model(img1.to(device), img2.to(device)), gt.to(device))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def save_report(stats):
    with open(REPORT, "w") as f:
        f.write(f"""# Laporan Pelatihan Artikel D -- grid /16 + fusi FPN ({VARIANT})

## 1. Ringkasan Waktu
- **Mulai:** {stats['start']}
- **Selesai:** {stats['end']}
- **Durasi:** {stats['duration']}
- **Status Berhenti:** {stats['stop_reason']}

## 2. Epoch & Batch
- **Epoch dilalui:** {stats['epochs_done']} / {stats['epochs_total']}
- **Batch per epoch:** {stats['batches']}
- **Sampel training per epoch:** {stats['samples']}

## 3. Data
- **Training:** {stats['data_path']}
- **Validasi:** {stats['val_path']}
- **Pseudo-GT /16:** {stats['pseudo']}

## 4. Memori
- **Rata-rata puncak VRAM:** {stats['avg_vram']:.2f} GB
- **Puncak VRAM tertinggi:** {stats['max_vram']:.2f} GB

## 5. Loss
- **Val loss terbaik:** {stats['best_val']:.4f}
- **Train loss terakhir:** {stats['final_train']:.4f}
- **Val loss terakhir:** {stats['final_val']:.4f}
""")


def main():
    smoke = 0
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    else:
        from training_logging import setup_tee_logging
        setup_tee_logging(f"training_run_p16_{TAG}.log")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True      # shape statis -> autotune kernel sekali di awal
    cfg = yaml.safe_load(open("config.yaml"))
    tcfg, dcfg, pcfg = cfg["training"], cfg["dataset"], cfg.get("pseudo_gt", {})

    frac = tcfg.get("gpu_memory_fraction")
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac, 0)

    model = model_p16.build(VARIANT, d_model=cfg["model"]["d_model"],
                            nhead=cfg["model"]["nhead"],
                            num_layers=cfg["model"]["num_layers"], flash=True).to(device)

    # Melanjutkan run yang terputus. `P16_RESUME=auto` memakai state lengkap terakhir
    # (bobot + optimizer + epoch) bila ada; kalau tidak, jatuh ke checkpoint bobot saja.
    resume_state, start_epoch, resumed_best = None, 0, float('inf')
    want_resume = os.environ.get("P16_RESUME", "")
    if want_resume:
        path = CKPT_LAST if want_resume == "auto" else want_resume
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "model" in blob:
                model.load_state_dict({k: v.float() for k, v in blob["model"].items()})
                resume_state, start_epoch = blob, blob.get("epoch", 0)
                resumed_best = blob.get("best_val", float('inf'))
                print(f"[resume] state lengkap dari {path}: lanjut dari epoch {start_epoch}, "
                      f"best_val={resumed_best:.4f}")
            else:
                model.load_state_dict({k: v.float() for k, v in blob.items()})
                print(f"[resume] BOBOT SAJA dari {path} (optimizer di-reset; "
                      f"epoch dihitung ulang dari 0)")
            warm = dict(resumed_from=path)
        elif want_resume == "auto":
            print(f"[resume] {path} belum ada -- mulai dari warm start biasa.")
            warm = model.load_pretrained_from(CKPT_WARM)
        else:
            raise FileNotFoundError(path)
    else:
        warm = model.load_pretrained_from(CKPT_WARM)
    json.dump(dict(variant=VARIANT, grid=list(GRID_HW), d_model=cfg["model"]["d_model"],
                   num_layers=cfg["model"]["num_layers"],
                   stage3_blocks=len(model.kept_block_indices), use_fpn=model.use_fpn,
                   kept_block_indices=model.kept_block_indices, warm_start=warm),
              open(META_OUT, "w"), indent=2)

    # RAM. OOM 2026-08-18 (log OOM kernel 06:42): 8 worker train + 2 worker val, tiap
    # slot memegang LRU 2 ortho terdekode. Ortho terbesar 33215x34607 = 3.21 GiB RGB,
    # jadi puncak per slot ~6.4 GiB mantap dan ~9.6 GiB saat mendekode gambar ke-3
    # (worker yg dibunuh: 9.87 GB) -> 10 slot x 9.6 = ~96 GiB > 91 GiB RAM + 8 GiB swap.
    # Cache diturunkan ke 1: `__getitem__` memilih ortho ACAK dari 36 file tiap panggilan,
    # jadi hit rate cuma turun 5.6% -> 2.8% (cache-nya memang nyaris tak berguna) sementara
    # puncak per slot terpotong separuh. TIDAK mengubah distribusi sampling, jadi run ini
    # tetap sebanding langsung dgn run FPN yg sudah selesai.
    CACHE_IMGS = int(os.environ.get("P16_CACHE", 1))
    val_exclude = dcfg.get("val_exclude", [])
    train_ds = HomographyOrthoDatasetP16(
        data_path=dcfg["train_path"], epoch_size=(smoke * tcfg["batch_size"] if smoke else 4000),
        extra_files=[os.path.join(dcfg["val_path"], n) for n in val_exclude],
        max_cache_images=CACHE_IMGS, grid_hw=GRID_HW)
    val_ds = HomographyOrthoDatasetP16(
        data_path=dcfg["val_path"], epoch_size=(2 * tcfg["batch_size"] if smoke else dcfg.get("val_epoch_size", 400)),
        exclude_names=val_exclude, max_cache_images=CACHE_IMGS, grid_hw=GRID_HW)

    if not smoke:
        train_ds = build_mixed_sparse(train_ds, PSEUDO_INDEX, pcfg.get("mix_ratio", 0.25),
                                      enabled=pcfg.get("enabled", True))

    # Bisa ditimpa dari lingkungan supaya throughput bisa DISAPU (P16_BATCH/P16_WORKERS)
    # tanpa mengedit config.yaml yang dipakai skrip lain. Catatan config "batch_size=2 sudah
    # 97% util" diukur 2026-07-10 pada ARSITEKTUR LAMA di RTX 5090 -- tidak otomatis berlaku
    # di sini (grid 3072 token, SDPA, kartu berbeda), jadi diukur ulang.
    bs = int(os.environ.get("P16_BATCH", tcfg["batch_size"]))
    nw = int(os.environ.get("P16_WORKERS", 2 if smoke else tcfg.get("num_workers", 12)))
    dl_kw = dict(pin_memory=True)
    if nw > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=2)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, **dl_kw)
    # Val hanya 400 crop/epoch tapi 2 worker persisten-nya residen SEPANJANG training dan
    # memegang cache ortho sendiri -- ikut penyebab OOM 2026-08-18. Dijalankan di proses
    # utama saja; biayanya sedetik-dua per epoch, bukan 2 slot RAM permanen.
    nw_val = int(os.environ.get("P16_VAL_WORKERS", 0))
    val_kw = dict(pin_memory=True)
    if nw_val > 0:
        val_kw.update(persistent_workers=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw_val, **val_kw)

    optimizer = optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda')
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        print("[resume] optimizer + scaler dipulihkan")
    accum = max(1, tcfg.get("target_effective_batch_size", 16) // bs)
    epochs = 1 if smoke else int(os.environ.get("P16_MAX_EPOCHS", tcfg["epochs"]))
    patience = tcfg.get("early_stopping_patience", 25)
    print(f"[cfg] batch={bs} x accum={accum} | workers={nw} | grid={GRID_HW} | "
          f"batches/epoch={len(train_dl)}")

    model.train()
    t_start = time.time()
    best_val, final_train, final_val = resumed_best, float('inf'), float('inf')
    bad, epochs_done, stop_reason = 0, 0, "Selesai (Full Epoch)"
    vram = []

    try:
        for epoch in range(0 if smoke else start_epoch, epochs):
            ep_loss, t_ep = 0.0, time.time()
            t_warm = t_ep
            for i, (img1, img2, gt) in enumerate(train_dl):
                if i == 0:
                    optimizer.zero_grad()
                if i == 3:
                    t_warm = time.time()   # patokan waktu steady-state (lihat blok [SMOKE])
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    conf = model(img1.to(device), img2.to(device))
                    loss = compute_loss_sparse(conf, gt.to(device)) / accum
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                ep_loss += loss.item() * accum
                vram.append(torch.cuda.max_memory_allocated() / 1024 ** 3)
                torch.cuda.reset_peak_memory_stats()
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_dl)}] "
                          f"Loss: {loss.item()*accum:.4f} | VRAM: {vram[-1]:.2f}GB | "
                          f"{(time.time()-t_ep)/(i+1):.2f}s/step", flush=True)

            avg_train = ep_loss / len(train_dl)
            val_loss = validate(model, val_dl, device)
            print(f"Epoch [{epoch+1}/{epochs}] Train {avg_train:.4f} | Val {val_loss:.4f} | "
                  f"{(time.time()-t_ep)/60:.1f} menit", flush=True)

            if smoke:
                # Buang 3 step pertama: cudnn.benchmark autotune + pengisian cache ortho
                # membuat step awal jauh lebih lambat dan menyesatkan kalau ikut dirata-rata.
                warm = min(3, len(train_dl) - 1)
                s_per_step = (time.time() - t_warm) / max(len(train_dl) - warm, 1)
                print(f"[SMOKE] batch={bs} workers={nw} | {len(train_dl)} step OK | "
                      f"VRAM puncak {max(vram):.2f} GB | {s_per_step:.2f} s/step | "
                      f"{s_per_step/bs:.3f} s/sampel | "
                      f"4000 sampel/epoch = {4000*s_per_step/bs/3600:.2f} jam", flush=True)
                return

            # State lengkap tiap epoch: run ini pernah hilang karena proses dibunuh,
            # dan tanpa optimizer state melanjutkan berarti mengulang pemanasan Adam.
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch + 1, best_val=best_val),
                       CKPT_LAST)

            # Arsip per-epoch. Diperlukan karena val loss terbukti BERLAWANAN ARAH dgn
            # kualitas pencocokan (log §59.3): checkpoint harus dipilih memakai metrik
            # hilir sesudahnya, jadi kandidatnya tidak boleh ditimpa.
            if os.environ.get("P16_ARCHIVE") == "1":
                os.makedirs(ARCHIVE_DIR, exist_ok=True)
                torch.save({k: v.half() for k, v in model.state_dict().items()},
                           f"{ARCHIVE_DIR}/ep{epoch + 1:03d}_val{val_loss:.4f}.pth")

            if val_loss < best_val:
                best_val, bad = val_loss, 0
                torch.save({k: v.half() for k, v in model.state_dict().items()}, CKPT_OUT)
                print(f"--> [Model Terbaik Disimpan] Val loss turun ke {best_val:.4f}", flush=True)
            else:
                bad += 1
                if bad >= patience:
                    stop_reason = "Terpicu Early Stopping"
                    print(f"==== Early Stopping pada epoch {epoch+1} ====")
                    break
            epochs_done, final_train, final_val = epoch + 1, avg_train, val_loss

    except KeyboardInterrupt:
        stop_reason = "Dihentikan Manual"
    finally:
        dur = int(time.time() - t_start)
        save_report(dict(
            start=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start)),
            end=time.strftime('%Y-%m-%d %H:%M:%S'),
            duration=f"{dur//3600}h {(dur%3600)//60}m {dur%60}s", stop_reason=stop_reason,
            epochs_done=epochs_done, epochs_total=epochs, batches=len(train_dl),
            samples=len(train_dl.dataset), data_path=dcfg["train_path"], val_path=dcfg["val_path"],
            pseudo=PSEUDO_INDEX, avg_vram=sum(vram)/max(len(vram), 1), max_vram=max(vram or [0]),
            best_val=best_val, final_train=final_train, final_val=final_val))


if __name__ == "__main__":
    main()
