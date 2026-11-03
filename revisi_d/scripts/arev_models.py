"""Revisi A IJIES -- varian arsitektur sapuan terkendali + pembungkus pencocok untuk evaluasi.

ArevModel = AerialStitchingModelP16 (tanpa fusi) dengan dua opsi tambahan yang tidak menambah parameter:
  stride 32   : avg-pool 2x2 pada keluaran tahap 3 sebelum proyeksi (kisi 24x32 pada 768x1024);
  target_size : resolusi internal (768x1024 usulan, atau 480x640 tanpa pembesaran; antialias saat mengecilkan).
Lapis badan dan blok tahap 3 memakai argumen model_p16 yang sudah ada.

Spesifikasi pencocok untuk evaltool: 'arev=<tag>=<checkpoint>=<label>' -- arsitektur dibaca dari penanda
revisi_d/results/arev/train/<tag>.train.done (argumen run), jadi checkpoint selalu dimuat ke arsitektur yang benar
(muat ketat: kunci hilang/lebih = galat).
"""
import json
import os

import torch
import torch.nn.functional as F

import model_p16 as MP
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

ROOT = AERIAL_ROOT + ""
DONE_DIR = f"{ROOT}/revisi_d/results/arev/train"


class ArevModel(MP.AerialStitchingModelP16):
    def __init__(self, *args, stride=16, target_size=(768, 1024), **kw):
        super().__init__(*args, **kw)
        self.target_size = tuple(target_size)
        self.arev_stride = stride

    def extract_features(self, img):
        down = img.shape[-2] > self.target_size[0] or img.shape[-1] > self.target_size[1]
        x = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False, antialias=down)
        c3 = self.trunk(x)                                     # [B, H, W, 512]
        if self.arev_stride == 32:
            c3 = F.avg_pool2d(c3.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        b, h, w, c = c3.shape
        return self.proj(c3.reshape(b, h * w, c))


def run_args(tag):
    """Argumen arsitektur run dari penanda selesai (atau berkas meta bila run belum selesai)."""
    p = f"{DONE_DIR}/{tag}.train.done"
    if os.path.exists(p):
        return json.load(open(p))["args"]
    raise FileNotFoundError(f"penanda run {p} belum ada")


def grid_of(args):
    ih, iw = (int(x) for x in args["res"].split("x"))
    return (ih // args["stride"], iw // args["stride"]), (ih, iw)


def build_net(args, pretrained_backbone=False):
    grid, (ih, iw) = grid_of(args)
    net = ArevModel(d_model=128, nhead=8, num_layers=args["layers"], stage3_blocks=args["blocks"], use_fpn=False,
                    flash=True, pretrained_backbone=pretrained_backbone, stride=args["stride"], target_size=(ih, iw))
    return net, grid


def build_matcher(spec):
    """'arev=<tag>=<ckpt>=<label>' -> objek SD.OurModel (mutual-NN pada kisi varian)."""
    import stitch_demo as SD
    from quant_sweep import DEVICE
    _, tag, path, label = spec.split("=")
    args = run_args(tag)
    net, grid = build_net(args)
    sd = torch.load(path, map_location="cpu")
    miss, unexp = net.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    if miss or unexp:
        raise RuntimeError(f"{label}: checkpoint tidak cocok missing={miss[:3]} unexpected={unexp[:3]}")
    m = SD.OurModel.__new__(SD.OurModel)
    m.label, m.grid, m.device, m.half = label, grid, DEVICE, True
    m.model = net.to(DEVICE).half().eval()
    m.arev_args = args
    return m


def install_into_evaltool():
    """evaltool.build_matchers menerima spesifikasi 'arev=...' (pola _p16_ckpt/_bb_ckpt)."""
    import evaltool as ET
    orig = ET.build_matchers

    def build_matchers(which):
        out = []
        for n in which.split(","):
            if n.startswith("arev="):
                out.append(build_matcher(n))
                print(f"[model] {n} siap (kisi {out[-1].grid})", flush=True)
            else:
                out += orig(n)
        return out
    ET.build_matchers = build_matchers
    return ET
