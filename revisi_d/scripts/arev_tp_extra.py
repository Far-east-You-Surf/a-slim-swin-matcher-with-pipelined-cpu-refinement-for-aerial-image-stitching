"""Revisi A IJIES (G3, butir R2.2) -- pencocok efisien tambahan yang didaftarkan ke tp_matchers TANPA menyunting
berkas itu: EDM (Li dkk., ICCV 2025; github.com/chicleee/EDM commit 8a31d81, Apache-2.0; bobot outdoor
weights/edm_outdoor.ckpt sha256 f686bebd...).

Protokol sama dengan pembanding lain: abu-abu [0,1] 480x640 (cv2.resize), konfigurasi outdoor resmi
(configs/edm/outdoor/edm_base.py: MCONF_THR 0.05, BORDER_RM 0), TOPK = 0,35 x jumlah sel kisi /8 sesuai catatan
config repo ("Top-K should not exceed grid_size ... approximately grid_size * 0.35") -> 1680 pada 480x640;
NPE = [832, 832, 480, 640] (resolusi latih outdoor, resolusi uji = masukan).
Pakai: import arev_tp_extra; arev_tp_extra.install()  (lalu evaltool.build_matchers("edm") / tp_matchers.build("edm"))
"""
import os

import torch

import tp_matchers as TM

EDM_ROOT = os.path.join(TM.TP, "EDM")
EDM_W = os.path.join(EDM_ROOT, "weights", "edm_outdoor.ckpt")


def _lightning_shim():
    """EDM mengimpor `lightning.pytorch.utilities.rank_zero_only`; venv punya pytorch_lightning 2.6.5 (API sama)."""
    import sys
    import types
    try:
        import lightning  # noqa: F401
        return
    except ImportError:
        pass
    import pytorch_lightning as pl
    import pytorch_lightning.utilities as plu
    lt = types.ModuleType("lightning")
    lt.pytorch = pl
    sys.modules.update({"lightning": lt, "lightning.pytorch": pl, "lightning.pytorch.utilities": plu})


def _build_edm():
    TM.REPOS.setdefault("edm", EDM_ROOT)
    TM._enter_repo("edm")
    _lightning_shim()
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config
    c = get_cfg_defaults()
    c.merge_from_file(os.path.join(EDM_ROOT, "configs", "edm", "outdoor", "edm_base.py"))
    h, w = 480, 640
    c.EDM.COARSE.TOPK = int(h / 8 * w / 8 * 0.35)
    # pengodean posisi ternormalisasi: posisi uji diskalakan ke resolusi latih outdoor (832x832), seperti
    # configs/data/megadepth_test_1500.py tetapi dengan resolusi uji = masukan protokol kita (480x640)
    c.EDM.TRAIN_RES_H, c.EDM.TRAIN_RES_W, c.EDM.TEST_RES_H, c.EDM.TEST_RES_W = 832, 832, h, w
    c.EDM.NECK.NPE = [832, 832, h, w]
    cfg = lower_config(c)["edm"]
    m = EDM(config=cfg)
    sd = torch.load(TM.WEIGHTS["edm"], map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("state_dict", sd))          # strict
    return m.eval(), cfg


def install():
    TM.REPOS["edm"] = EDM_ROOT
    TM.WEIGHTS.setdefault("edm", EDM_W)
    TM.BUILDERS["edm"] = _build_edm
    if "edm" not in TM.LABELS:
        TM.LABELS.append("edm")
    try:
        import evaltool as ET
        orig = ET.build_matchers
        if not getattr(orig, "_arev_edm", False):
            def build_matchers(which):
                out = []
                for n in which.split(","):
                    if n == "edm":
                        out.append(ET._Capped(TM.build("edm")))
                        print("[model] edm siap", flush=True)
                    else:
                        out += orig(n)
                return out
            build_matchers._arev_edm = True
            ET.build_matchers = build_matchers
    except ImportError:
        pass
