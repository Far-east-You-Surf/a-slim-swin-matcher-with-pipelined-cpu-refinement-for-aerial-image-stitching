"""Revisi A IJIES (T3, butir R1.1) -- latih ulang pembanding (LGFCTR, MatchFormer) pada aliran data udara YANG SAMA
dengan usulan, dari bobot outdoor resminya.

Sama dengan usulan: aliran sampel (4000 potongan ortho sintetis + 1333 pasangan nyata pseudo-GT per epoch, ColorJitter
yang sama; DataLoader batch 2, 8 worker, generator = benih -> urutan sampel IDENTIK dengan run usulan berbenih sama),
12 epoch x 2667 langkah, batch efektif 16 (akumulasi 8), AdamW 1e-4 konstan, wd 1e-4, seleksi epoch dengan aturan q15.
Milik pembanding: arsitektur, loss resmi keluarga LoFTR (fokal kasar jarang + L2 berbobot std halus, lgfctr_loss.py),
ukuran masukan 480x640 abu-abu (= pembungkus evaluasi), clip gradien 0,5 (bawaan pelatih LoFTR).
Supervisi dari homografi (bukan kedalaman+pose): kisi /8 citra 0 dilengkungkan H, citra 1 dengan H^-1, sel tujuan =
pembulatan terdekat, hanya pasangan mutual di dalam citra (pola spvs_coarse repo); target halus = sisa di jendela /2.
H sintetis = homografi acak sampel itu sendiri; H pasangan nyata = homografi RANSAC guru pseudo-label
(pseudo_gt_cache_arev/index_48x64.json, guru dan pasangan sama dengan label usulan).
Presisi bf16 dengan perhatian linear dihitung FP32 (FP16 murni menghasilkan NaN, catatan P2c).
Titik simpan atomik per epoch (_last.pt + arsip bobot), lanjut otomatis; benih lanjut = benih*1000+epoch.

Pakai: python revisi_d/scripts/arev_tp_train.py --model lgfctr|matchformer --seed 1 [--epochs 12] [--smoke N]
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
RD = f"{ROOT}/revisi_d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lgfctr", "matchformer"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--smoke", type=int, default=0)
    # MatchFormer large-LA kehabisan memori pada batch 2 di 16 GB (uji asap 2 Nov 02:37): batch 2 tetap dimuat dari
    # DataLoader yang sama (aliran sampel identik dgn run usulan), lalu dipecah jadi mikro-batch 1 pasangan; batch
    # efektif tetap 16 (8 batch x 2). Bobot loss menjadi rerata per pasangan, bukan per batch (dicatat di naskah).
    ap.add_argument("--micro", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{RD}/scripts")
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from torch.utils.data import ConcatDataset, DataLoader
    cv2.setNumThreads(1)
    import arev_data as AD
    import tp_matchers as TM

    tag = f"tp_{a.model}_ft_s{a.seed}" + ("_smoke" if a.smoke else "")
    arch = f"{ROOT}/ckpt_archive_{tag}"
    last = f"{ROOT}/{tag}_last.pt"
    done_f = f"{RD}/results/arev/train/{tag}.train.done"
    os.makedirs(arch, exist_ok=True)
    os.makedirs(os.path.dirname(done_f), exist_ok=True)

    def atomic_save(obj, path):
        torch.save(obj, path + ".tmp")
        with open(path + ".tmp", "rb") as fh:
            os.fsync(fh.fileno())
        if os.path.exists(path) and path.endswith("_last.pt"):
            os.replace(path, path + ".prev")
        os.replace(path + ".tmp", path)

    # ---- model + bobot resmi ---------------------------------------------------------------------------------
    m, cfg = TM.BUILDERS[a.model]()
    dev = torch.device("cuda")
    m = m.to(dev)
    if a.model == "lgfctr":
        lcfg = cfg
        n_attn = 0
        for mod in m.modules():
            if type(mod).__name__ == "LinearAttention":
                of = mod.forward

                def f(q, k, v, q_mask=None, kv_mask=None, of=of):
                    with torch.autocast("cuda", enabled=False):
                        return of(q.float(), k.float(), v.float(), q_mask, kv_mask)
                mod.forward = f
                n_attn += 1
        res, win = tuple(cfg["LGFCTR"]["RESOLUTION"]), cfg["LGFCTR"]["MATCH_FINE"]["FINE_WINDOW_SIZE"]
    else:
        import types
        from p2c_matchformer import attn_fp32_forward
        n_attn = 0
        for mod in m.modules():
            if type(mod).__name__ == "Attention" and hasattr(mod, "feature_map"):
                def f(self, x, _f=attn_fp32_forward):
                    with torch.autocast("cuda", enabled=False):
                        return _f(self, x)
                mod.forward = types.MethodType(f, mod)
                n_attn += 1
        res, win = (8, 2), 5
        mc = cfg["match_coarse"]
        # loss keluarga LoFTR dengan nilai bawaan yang sama dengan LGFCTR/LoFTR outdoor
        lcfg = {"LGFCTR": {"LOSS": dict(COARSE_TYPE="focal", COARSE_WEIGHT=1.0, FOCAL_ALPHA=0.25, FOCAL_GAMMA=2.0,
                                        POS_WEIGHT=1.0, NEG_WEIGHT=1.0, FINE_TYPE="l2_with_std", FINE_WEIGHT=1.0,
                                        FINE_WEIGHT_CLS=1.0, FINE_CORRECT_THR=1.0, POS_WEIGHT_CLS=1.0,
                                        NEG_WEIGHT_CLS=1.0),
                           "MATCH_COARSE": dict(MATCH_TYPE=mc.get("match_type", "dual_softmax"),
                                                SPARSE_SPVS=bool(mc.get("sparse_spvs", True)))}}
    spec = importlib.util.spec_from_file_location("lgfctr_loss", f"{RD}/thirdparty/LGFCTR/src/losses/lgfctr_loss.py")
    LM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LM)
    loss_fn = LM.LGFCTRLoss(lcfg)
    print(f"[tp] {a.model}: perhatian FP32 dipatch {n_attn} modul; resolusi {res}, jendela halus {win}; "
          f"loss {lcfg['LGFCTR']['LOSS']['COARSE_TYPE'] if isinstance(lcfg, dict) else lcfg.LGFCTR.LOSS.COARSE_TYPE}",
          flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    start_epoch = 0
    for cand in (last, last + ".prev"):
        if os.path.exists(cand):
            try:
                blob = torch.load(cand, map_location="cpu", weights_only=False)
                m.load_state_dict(blob["model"])
                opt.load_state_dict(blob["optimizer"])
                start_epoch = int(blob["epoch"])
                print(f"[tp] lanjut dari {cand}: epoch {start_epoch}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[tp] gagal memuat {cand}: {e!r}", flush=True)
    eff_seed = a.seed if start_epoch == 0 else a.seed * 1000 + start_epoch
    random.seed(eff_seed)
    np.random.seed(eff_seed % 2 ** 32)
    torch.manual_seed(eff_seed)

    # ---- data: aliran yang sama dengan usulan + H tiap sampel ----------------------------------------------------
    OrthoH, PseudoH, tmat = AD.OrthoH, AD.PseudoH, AD._tmat

    dcfg = yaml.safe_load(open("config.yaml"))["dataset"]
    syn = OrthoH(data_path=dcfg["train_path"], epoch_size=(a.smoke * 2 if a.smoke else 4000),
                 extra_files=[os.path.join(dcfg["val_path"], n) for n in dcfg["val_exclude"]], max_cache_images=1,
                 grid_hw=(48, 64))
    ds = syn if a.smoke else ConcatDataset([syn, PseudoH(f"{ROOT}/pseudo_gt_cache_p16/pseudo_gt_index.json", epoch_size=1333)])
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True,
                    prefetch_factor=2, generator=torch.Generator().manual_seed(eff_seed))
    accum = 8
    Tin = torch.tensor(tmat((640 / 1024, 480 / 768)), dtype=torch.float64, device=dev)
    Tin_inv = torch.linalg.inv(Tin)
    gw = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
    scale_c, scale_f, radius = res[0], res[1], win // 2
    h0, w0 = 480 // scale_c, 640 // scale_c
    gy, gx = torch.meshgrid(torch.arange(h0, device=dev), torch.arange(w0, device=dev), indexing="ij")
    grid_c = torch.stack([gx, gy], -1).reshape(1, -1, 2).double()          # = kornia create_meshgrid(normalized=False)
    grid_i = grid_c * scale_c

    def warp(H, p):  # H (N,3,3), p (N,M,2)
        ph = torch.cat([p, torch.ones_like(p[..., :1])], -1) @ H.transpose(1, 2)
        return ph[..., :2] / ph[..., 2:3]

    def supervise(data, H):
        N = H.shape[0]
        g0 = grid_i.repeat(N, 1, 1)
        w0i = warp(H, g0)
        w1i = warp(torch.linalg.inv(H), g0)
        w0c, w1c = w0i / scale_c, w1i / scale_c
        r0, r1 = w0c.round().long(), w1c.round().long()
        idx1 = r0[..., 0] + r0[..., 1] * w0
        idx0 = r1[..., 0] + r1[..., 1] * w0

        def oob(p):
            return (p[..., 0] < 0) | (p[..., 0] >= w0) | (p[..., 1] < 0) | (p[..., 1] >= h0)
        bad = ~torch.isfinite(w0i).all(-1)
        idx1[oob(r0) | bad] = 0
        idx0[oob(r1) | ~torch.isfinite(w1i).all(-1)] = 0
        loop = torch.gather(idx0, 1, idx1)
        ok = loop == torch.arange(h0 * w0, device=dev)[None]
        ok[:, 0] = False
        ok &= ~bad
        cm = torch.zeros(N, h0 * w0, h0 * w0, device=dev)
        b_ids, i_ids = torch.where(ok)
        j_ids = idx1[b_ids, i_ids]
        cm[b_ids, i_ids, j_ids] = 1
        if len(b_ids) == 0:
            b_ids = i_ids = j_ids = torch.zeros(1, dtype=torch.long, device=dev)
        data.update(conf_matrix_gt=cm, spv_b_ids=b_ids, spv_i_ids=i_ids, spv_j_ids=j_ids,
                    spv_w_pt0_i=w0i.float(), spv_pt1_i=g0.float(), dataset_name=["homography"] * N)
        return int(ok.sum())

    def supervise_fine(data):
        b, i, j = data["b_ids"], data["i_ids"], data["j_ids"]
        data["expec_f_gt"] = (data["spv_w_pt0_i"][b, i] - data["spv_pt1_i"][b, j]) / scale_f / radius

    m.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.precision == "bf16"))
    n_ep = 1 if a.smoke else a.epochs
    for epoch in range(start_epoch, n_ep):
        t_ep, ep_loss, n_ok = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        nb = len(dl)
        for it, (i1, i2, H, kind) in enumerate(dl):
            i1, i2 = i1.to(dev, non_blocking=True), i2.to(dev, non_blocking=True)
            g1 = F.interpolate((i1 * gw).sum(1, keepdim=True), size=(480, 640), mode="bilinear", align_corners=False)
            g2 = F.interpolate((i2 * gw).sum(1, keepdim=True), size=(480, 640), mode="bilinear", align_corners=False)
            Hin = Tin @ H.to(dev) @ Tin_inv
            nb_in = g1.shape[0]
            parts = [slice(0, nb_in)] if a.micro == 1 else [slice(k, k + 1) for k in range(nb_in)]
            b_loss, n_pos = 0.0, 0
            for sl in parts:
                data = {"image0": g1[sl], "image1": g2[sl]}
                n_pos += supervise(data, Hin[sl])
                with amp:
                    m(data)
                supervise_fine(data)
                data["conf_matrix"] = data["conf_matrix"].float()
                data["expec_f"] = data["expec_f"].float()
                loss_fn(data)
                loss = data["loss"] / (accum * len(parts))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss tidak hingga pada epoch {epoch + 1} langkah {it}")
                loss.backward()
                b_loss += float(data["loss"]) / len(parts)
                del data, loss
            data = {"loss": torch.tensor(b_loss)}
            if (it + 1) % accum == 0 or (it + 1) == nb:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
                opt.step()
                opt.zero_grad(set_to_none=True)
            ep_loss += float(data["loss"])
            n_ok += n_pos
            if it % 50 == 0:
                print(f"Epoch [{epoch + 1}/{n_ep}] Batch [{it}/{nb}] Loss: {float(data['loss']):.4f} | "
                      f"gt+ {n_pos} | VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f}GB | "
                      f"{(time.time() - t_ep) / (it + 1):.2f}s/step", flush=True)
        print(f"Epoch [{epoch + 1}/{n_ep}] Train {ep_loss / nb:.4f} | {(time.time() - t_ep) / 60:.1f} menit", flush=True)
        if a.smoke:
            return
        atomic_save({k: v.detach().float().cpu() for k, v in m.state_dict().items()}, f"{arch}/ep{epoch + 1:03d}.pth")
        atomic_save(dict(model=m.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1), last)
    files = sorted(f for f in os.listdir(arch) if f.endswith(".pth"))
    with open(done_f + ".tmp", "w") as f:
        json.dump(dict(tag=tag, args=vars(a), archive=arch, files=files, t=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    os.replace(done_f + ".tmp", done_f)
    print(f"[tp] SELESAI {tag}", flush=True)


if __name__ == "__main__":
    main()
