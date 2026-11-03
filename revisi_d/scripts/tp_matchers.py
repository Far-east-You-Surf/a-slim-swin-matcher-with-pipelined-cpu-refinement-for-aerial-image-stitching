"""
E1.2 -- pembungkus matcher pihak ketiga dgn antarmuka `stitch_demo` (Revisi Artikel D).

Antarmuka (sama dgn stitch_demo.TopicFMModel/JamMaModel):
    m = build("efficientloftr")          # label: efficientloftr | lgfctr | swinmatcher_tgrs |
                                         #        matchformer | aspanformer
    pts1, pts2, orig_w, orig_h = m.match(p1, p2)   # titik di piksel FOTO ASLI
    run = m._prep_for_timing(p1, p2); run()        # forward saja, utk latensi eager
Masukan: `stitch_demo._imread` (IMREAD_IGNORE_ORIENTATION) lalu resize ke stitch_demo.INPUT_HW
(480x640). Ambang = ambang bawaan config outdoor/test repo. SEMUA kecocokan hasil model
dikembalikan -- tanpa top-K / ember / CAP (beda dgn kelas stitch_demo yang memotong ke CAP).

PENTING -- satu metode per proses: EfficientLoFTR, LGFCTR, ASpanFormer, MatchFormer(`model`)
dan SwinMatcher semuanya memakai paket tingkat-atas `src` (dan TopicFM/JamMa di third_party/
juga). `build()` menolak bila paket `src` milik repo lain sudah termuat.

Bobot & lisensi: revisi_d/results/e12/provenance.json. Torch tidak di-pin ulang; Lightning
dilewati (state_dict dimuat langsung, torch.load weights_only=False utk ckpt Lightning).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
TP = os.path.join(PROJ, "revisi_d", "thirdparty")
PYDEPS = os.path.join(TP, "pydeps")
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
if os.path.isdir(PYDEPS) and PYDEPS not in sys.path:
    sys.path.append(PYDEPS)          # append: tidak pernah membayangi paket venv

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPOS = {
    "efficientloftr": os.path.join(TP, "EfficientLoFTR"),
    "lgfctr": os.path.join(TP, "LGFCTR"),
    "swinmatcher_tgrs": os.path.join(TP, "SwinMatcher"),
    "matchformer": os.path.join(TP, "MatchFormer"),
    "aspanformer": os.path.join(TP, "ml-aspanformer"),
}
WEIGHTS = {
    "efficientloftr": os.path.join(TP, "weights", "eloftr", "weights", "eloftr_outdoor.ckpt"),
    "lgfctr": os.path.join(TP, "weights", "lgfctr", "LGFCTR_outdoor.ckpt"),
    "swinmatcher_tgrs": os.path.join(TP, "weights", "swinmatcher", "swinmatcher_512.ckpt"),
    "matchformer": os.path.join(TP, "weights", "matchformer", "outdoor-large-LA.ckpt"),
    "aspanformer": os.path.join(TP, "weights", "aspanformer", "weights", "outdoor.ckpt"),
}
LABELS = list(REPOS)


TOP_PKG = {"matchformer": "model"}   # MatchFormer: paket `model` + `config` (bentrok model.py proyek!)


def _enter_repo(name):
    root = REPOS[name]
    pkg = TOP_PKG.get(name, "src")
    mod = sys.modules.get(pkg)
    if name == "matchformer":
        mod = None          # bentrok `model` ditangani di _build_matchformer
    if mod is not None:
        f = os.path.abspath(getattr(mod, "__file__", "") or list(getattr(mod, "__path__", [""]))[0])
        if not f.startswith(root):
            raise RuntimeError(f"paket `src` dari repo lain sudah termuat ({f}); "
                               f"jalankan {name} di proses terpisah")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("state_dict", ck) if isinstance(ck, dict) else ck


# ------------------------------------------------------------------ pembangun mentah
def _build_efficientloftr():
    _enter_repo("efficientloftr")
    from copy import deepcopy
    from src.loftr import LoFTR, full_default_cfg, reparameter
    cfg = deepcopy(full_default_cfg)          # full, fp32, thr 0.2 (config resmi model full)
    m = LoFTR(config=cfg)
    m.load_state_dict(_load_ckpt(WEIGHTS["efficientloftr"]))   # strict
    m = reparameter(m)                        # "Essential for good performance" (README)
    return m.eval(), cfg


def _build_lgfctr():
    root = _enter_repo("lgfctr")
    from src.config.default import get_cfg_defaults
    from src.lgfctr import LGFCTR
    config = get_cfg_defaults()
    # config evaluasi resmi (vis/vis_single_pair.py & scripts/reproduce_test): thr coarse 0.2
    config.merge_from_file(os.path.join(root, "configs", "lgfctr", "outdoor", "lgfctr_ds_eval.py"))
    m = LGFCTR(config["LGFCTR"])
    m.load_state_dict(_load_ckpt(WEIGHTS["lgfctr"]), strict=True)
    return m.eval(), config


def _build_swinmatcher_tgrs():
    """SwinMatcher-TGRS (LotrL). Backbone `build_backbone()` MENGUNCI img_size=512 (PatchEmbed meng-
    assert H==W==512; mask SW-MSA & input_resolution dihitung di __init__; window 16). 480x640 TIDAK
    dapat dipakai: pada /16 lebar 40 tidak habis dibagi window 16, dan di /32 window menyusut ke
    min(15,20)=15 yang tidak membagi 20. Ukuran sah = kelipatan 512 (persegi). Jadi pembungkus
    memakai 512x512 (protokol demo repo) lalu titik diskalakan ke piksel foto asli."""
    _enter_repo("swinmatcher_tgrs")
    from copy import deepcopy
    from src.swinmatcher import SwinMatcher, default_cfg
    cfg = deepcopy(default_cfg)
    cfg["match_coarse"]["thr"] = 0.3     # nilai demo.py ("default: 0.3")
    cfg["match_coarse"]["border_rm"] = 2
    cfg["fine"]["thr"] = 0.1
    cfg["img_size"] = 512
    m = SwinMatcher(config=cfg)
    m.load_state_dict(_load_ckpt(WEIGHTS["swinmatcher_tgrs"]))   # strict
    return m.eval(), cfg


def _build_matchformer():
    """MatchFormer outdoor large-LA, config README bagian 'Outdoor' (largela, (8,2), d_model 256),
    thr coarse 0.2 (defaultmf.py), border_rm 0. Bobot = state_dict polos berprefiks `matcher.`
    (dimuat persis seperti lightning_loftr.PL_LoFTR: k.replace('matcher.', ''))."""
    root = _enter_repo("matchformer")
    # `model/` & `config/` repo adalah namespace package (tanpa __init__.py): model.py proyek
    # di sys.path akan MENANG. Saat impor, sys.path dibatasi sementara agar paket repo yang terambil.
    # Bila model.py proyek SUDAH termuat (mis. lewat collect_pilar3_multipair), ia disingkirkan
    # sementara dari sys.modules lalu dikembalikan; modul MatchFormer tetap terdaftar sbg
    # `model.matchformer`, `model.backbone...` (tidak ada impor malas di forward-nya).
    saved = list(sys.path)
    stash = {k: sys.modules.pop(k) for k in ("model", "config") if k in sys.modules}
    sys.path[:] = [root] + [p for p in saved if os.path.abspath(p or os.getcwd()) not in (PROJ, HERE)]
    try:
        from config.defaultmf import get_cfg_defaults
        from model.matchformer import Matchformer
        from model.utils.misc import lower_config
    finally:
        sys.path[:] = saved
        for k in ("model", "config"):
            sys.modules.pop(k, None)       # paket namespace MatchFormer tidak dibiarkan membayangi
        sys.modules.update(stash)
    c = get_cfg_defaults()
    c.MATCHFORMER.BACKBONE_TYPE = "largela"
    c.MATCHFORMER.SCENS = "outdoor"
    c.MATCHFORMER.RESOLUTION = (8, 2)
    c.MATCHFORMER.COARSE.D_MODEL = 256
    c.MATCHFORMER.COARSE.D_FFN = 256
    cfg = lower_config(c)["matchformer"]
    m = Matchformer(config=cfg)
    sd = torch.load(WEIGHTS["matchformer"], map_location="cpu", weights_only=False)
    m.load_state_dict({k.replace("matcher.", ""): v for k, v in sd.items()})   # strict
    return m.eval(), cfg


def _build_aspanformer():
    """ASpanFormer outdoor, configs/aspan/outdoor/aspan_test.py (COARSEST_LEVEL 36, TRAIN_RES 832,
    TEST_RES 1152, dual-softmax, thr 0.2). Bobot dimuat seperti demo/demo.py (strict=False; kunci
    `sample_offset` dibuang oleh load_state_dict repo). Inferensi memakai jalur demo
    `online_resize=True` (pos-encoding diskalakan ke TRAIN_RES, ds=[4,4]) -- lihat TPMatcher."""
    root = _enter_repo("aspanformer")
    from src.ASpanFormer.aspanformer import ASpanFormer
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config
    c = get_cfg_defaults()
    c.merge_from_file(os.path.join(root, "configs", "aspan", "outdoor", "aspan_test.py"))
    cfg = lower_config(c)["aspan"]
    m = ASpanFormer(config=cfg)
    res = m.load_state_dict(_load_ckpt(WEIGHTS["aspanformer"]), strict=False)
    m._load_report = dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
    return m.eval(), cfg


ONLINE_RESIZE = {"aspanformer"}     # forward(data, online_resize=True) spt demo.py

INPUT_OVERRIDE = {"swinmatcher_tgrs": (512, 512)}

BUILDERS = {
    "aspanformer": _build_aspanformer,
    "matchformer": _build_matchformer,
    "swinmatcher_tgrs": _build_swinmatcher_tgrs,
    "efficientloftr": _build_efficientloftr,
    "lgfctr": _build_lgfctr,
}


def build_raw(name, device="cpu"):
    m, cfg = BUILDERS[name]()
    return m.to(device).eval()


# ------------------------------------------------------------------------ pembungkus
class TPMatcher:
    """Pembungkus seragam. Semua kandidat E1.2 memakai masukan abu-abu [0,1] (N,1,H,W) dan
    menulis `mkpts0_f`, `mkpts1_f`, `mconf` ke dict masukan (konvensi LoFTR)."""

    def __init__(self, label, device=None):
        from stitch_demo import INPUT_HW
        self.label = label
        self.input_hw = INPUT_OVERRIDE.get(label, INPUT_HW)
        self.device = device or os.environ.get("STITCH_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = BUILDERS[label]()
        self.model = self.model.to(self.device).eval()

    # -- praproses
    def _tensors(self, p1, p2):
        from stitch_demo import _imread
        a, b = _imread(p1), _imread(p2)
        if a is None or b is None:
            raise RuntimeError(f"gagal membaca {p1} atau {p2}")
        oh, ow = a.shape[:2]
        h, w = self.input_hw
        g = []
        for im in (a, b):
            x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (w, h))
            g.append(torch.from_numpy(x).float()[None, None].to(self.device) / 255.0)
        return g, ow, oh

    def _batch(self, g):
        return {"image0": g[0], "image1": g[1]}

    def match(self, p1, p2):
        g, ow, oh = self._tensors(p1, p2)
        data = self._batch(g)
        with torch.no_grad():
            self._fwd(data)
        k0 = data["mkpts0_f"].detach().float().cpu().numpy()
        k1 = data["mkpts1_f"].detach().float().cpu().numpy()
        self.last_conf = data.get("mconf")
        if len(k0) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), ow, oh
        h, w = self.input_hw
        s = np.array([ow / w, oh / h])
        return (k0 * s).astype(np.float64), (k1 * s).astype(np.float64), ow, oh

    def _prep_for_timing(self, p1, p2):
        g, _, _ = self._tensors(p1, p2)
        # dict baru tiap panggilan: model menulis hasilnya ke dalam dict masukan
        return lambda: self._fwd(self._batch(g))

    def _fwd(self, data):
        if self.label in ONLINE_RESIZE:
            return self.model(data, online_resize=True)
        return self.model(data)


def build(name, device=None):
    if name not in BUILDERS:
        raise KeyError(f"tidak dikenal / tidak dapat dijalankan: {name}")
    return TPMatcher(name, device)
