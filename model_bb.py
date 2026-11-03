"""
Varian tulang punggung untuk uji "apakah Swin-B masih yang terbaik?" (Artikel D tahap 3, butir 7).

Kerangka IDENTIK dengan `model_p16.AerialStitchingModelP16` tanpa FPN (usulan): masukan diubah ke 768x1024,
fitur /16 (48x64 = 3072 token) -> proj -> 8 blok perhatian-diri/silang LoFTR (d=128, flash) -> matching_head ->
dual-softmax. Yang diganti HANYA ekstraktor fitur sampai /16; semuanya diambil dari timm (bobot ImageNet publik).

Preset (`build(name)`):
  swinb_d8_in      Swin-B torchvision, tahap 3 dipangkas 18->8, bobot ImageNet (KONTROL adil: sama dgn usulan
                   tetapi TANPA penghangatan aerial tulang punggung)
  swinv2s_w8      SwinV2-S jendela 8 (192x256..24x32 habis dibagi 8 -> tanpa padding), tahap 3 utuh (18 blok)
  swinv2s_w8_d8   idem, tahap 3 dipangkas 18->8 (pasangan shift dijaga, indeks = kept_stage3_indices)
  swinv2b_w8_d8    SwinV2-B jendela 8, tahap 3 dipangkas 18->8 (lebar sama dgn Swin-B usulan)
  convnext_t       ConvNeXt-T (kontrol CNN murni), tahap 3 (/16, 384 kanal)
  convnext_s       ConvNeXt-S
  effvit_l1        EfficientViT-MIT L1 (perhatian linear ReLU multi-skala), tahap /16
  effvit_l2        EfficientViT-MIT L2
  pvtv2_b2         PVTv2-B2 (perhatian reduksi-spasial), tahap /16

Bobot matcher (layers + matching_head) dapat DIHANGATKAN dari checkpoint usulan (`load_matcher_from`); tulang
punggung dan `proj` selalu dari ImageNet / acak -- satu-satunya perbedaan antar-preset adalah ekstraktornya.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_flash import LoFTRAttentionBlockFlash
from model_p16 import kept_stage3_indices

TIMM = {
    "swinv2s_w8": ("swinv2_small_window8_256.ms_in1k", 2),
    "swinv2s_w8_d8": ("swinv2_small_window8_256.ms_in1k", 2),
    "swinv2b_w8_d8": ("swinv2_base_window8_256.ms_in1k", 2),
    "convnext_t": ("convnext_tiny.fb_in22k_ft_in1k", 2),
    "convnext_s": ("convnext_small.fb_in22k_ft_in1k", 2),
    "effvit_l1": ("efficientvit_l1.r224_in1k", 2),
    "effvit_l2": ("efficientvit_l2.r384_in1k", 2),
    "pvtv2_b2": ("pvt_v2_b2.in1k", 2),
    "effvit_l1_att16": ("efficientvit_l1.r224_in1k", 3),
    "effvit_l1_att16_d2": ("efficientvit_l1.r224_in1k", 3),
    "effvit_l1_att16_d4": ("efficientvit_l1.r224_in1k", 3),
}
# EfficientViT-L: tahap /16 (stages_2) hanya MBConv; LiteMLA baru ada di stages_3 (/32). Varian *_att16 membawa
# stages_3 ke /16 dengan stride 1 pada depthwise blok pertamanya, sehingga 6 blok LiteMLA bekerja pada kisi 48x64.
STRIDE1 = {"effvit_l1_att16", "effvit_l1_att16_d2", "effvit_l1_att16_d4"}
ATT_KEEP = {"effvit_l1_att16_d2": 2, "effvit_l1_att16_d4": 4}   # blok LiteMLA yang disimpan (dari 6)
PRUNE = {"swinv2s_w8_d8": 8, "swinv2b_w8_d8": 8}


class TimmTrunk(nn.Module):
    """Ekstraktor timm features_only -> keluaran satu tahap /16 dalam format [B, H, W, C]."""

    def __init__(self, name, out_idx, hw, pretrained=True, prune_stage3=None, stride1=False):
        super().__init__()
        import timm
        kw = dict(features_only=True, out_indices=(out_idx,), pretrained=pretrained)
        if name.startswith("swinv2"):
            kw.update(img_size=hw)
        self.m = timm.create_model(name, **kw)
        if prune_stage3:
            st = self.m.layers_2 if hasattr(self.m, "layers_2") else self.m.layers[2]
            keep = kept_stage3_indices(len(st.blocks), prune_stage3)
            st.blocks = nn.Sequential(*[st.blocks[i] for i in keep])
            self.kept = keep
        if name.startswith("efficientvit") and out_idx == 3 and stride1:
            self.m.stages_3.blocks[0].main.depth_conv.conv.stride = (1, 1)
            if stride1 is not True:
                b = self.m.stages_3.blocks
                self.m.stages_3.blocks = nn.Sequential(*[b[i] for i in range(1 + stride1)])
        self.channels = self.m.feature_info.channels()[-1]
        self.nhwc = getattr(self.m, "output_fmt", "NCHW") == "NHWC" or name.startswith("swin")

    def forward(self, x):
        y = self.m(x)[-1]
        if y.shape[-1] == self.channels and y.shape[1] != self.channels:
            return y                                  # sudah NHWC
        return y.permute(0, 2, 3, 1)


class SwinBTrunk(nn.Module):
    def __init__(self, blocks=8, pretrained=True):
        super().__init__()
        from torchvision.models import swin_b, Swin_B_Weights
        feats = list(swin_b(weights=Swin_B_Weights.DEFAULT if pretrained else None).features)
        st = feats[5]
        keep = kept_stage3_indices(len(st), blocks)
        feats[5] = nn.Sequential(*[st[i] for i in keep])
        self.m = nn.Sequential(*feats[:6])
        self.channels = 512

    def forward(self, x):
        return self.m(x)


class SelfOnlyBlockFlash(nn.Module):
    """Ablasi badan perhatian: blok LoFTR TANPA perhatian-silang (nama modul sama -> hangat dari usulan)."""
    def __init__(self, d_model, nhead):
        super().__init__()
        from model_flash import FlashMHA
        self.self_attn = FlashMHA(d_model, nhead)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):
        feat1 = self.norm1(feat1 + self.self_attn(feat1, feat1, feat1)[0])
        feat2 = self.norm1(feat2 + self.self_attn(feat2, feat2, feat2)[0])
        return self.norm3(feat1 + self.ffn(feat1)), self.norm3(feat2 + self.ffn(feat2))


def head_blocks(net):
    """Blok residual tulang punggung yang dapat dibuang (shortcut identitas) -> [(modul_stage, indeks_blok)]."""
    t, out = net.trunk.m, []
    for sn in ("stages_0", "stages_1", "stages_2", "stages_3"):
        st = getattr(t, sn, None)
        if st is None:
            continue
        for i, b in enumerate(st.blocks):
            if type(getattr(b, "shortcut", None)).__name__ == "Identity":
                out.append((st, i))
    return out


def prune_spec(preset):
    """'__dh1.3' buang blok kepala (indeks global head_blocks), '__db0.7' buang blok badan perhatian."""
    dh, db = [], []
    for m in preset.split("__")[1:]:
        if m.startswith("dh"):
            dh = [int(x) for x in m[2:].split(".") if x]
        elif m.startswith("db"):
            db = [int(x) for x in m[2:].split(".") if x]
    return dh, db


def prune_after_load(net, preset):
    """Pemangkasan lapis TANPA latih ulang: dipanggil SESUDAH bobot penuh dimuat (P14)."""
    dh, db = prune_spec(preset)
    if dh:
        H = head_blocks(net)
        drop = {}
        for g in dh:
            st, i = H[g]
            drop.setdefault(id(st), (st, set()))[1].add(i)
        for st, idx in drop.values():
            st.blocks = nn.Sequential(*[b for i, b in enumerate(st.blocks) if i not in idx])
    if db:
        net.layers = nn.ModuleList([l for i, l in enumerate(net.layers) if i not in set(db)])
    return net


def parse_preset(preset):
    """'<dasar>[__L<n>][__self][__s<benih>]' -> (dasar, num_layers|None, self_only, benih|None)."""
    base, *mods = preset.split("__")
    nl, so, seed = None, False, None
    for m in mods:
        if m.startswith("L"):
            nl = int(m[1:])
        elif m == "self":
            so = True
        elif m.startswith("s"):
            seed = int(m[1:])
    return base, nl, so, seed


class BBMatcher(nn.Module):
    def __init__(self, preset, d_model=128, nhead=8, num_layers=8, pretrained=True):
        super().__init__()
        self.target_size = (768, 1024)
        self.preset = preset
        preset, nl, self_only, _ = parse_preset(preset)
        num_layers = num_layers if nl is None else nl
        if preset == "swinb_d8_in":
            self.trunk = SwinBTrunk(8, pretrained)
        else:
            name, idx = TIMM[preset]
            self.trunk = TimmTrunk(name, idx, self.target_size, pretrained, PRUNE.get(preset), ATT_KEEP.get(preset, True) if preset in STRIDE1 else False)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
        self.proj = nn.Linear(self.trunk.channels, d_model)
        blk = SelfOnlyBlockFlash if self_only else LoFTRAttentionBlockFlash
        self.layers = nn.ModuleList([blk(d_model, nhead) for _ in range(num_layers)])
        self.matching_head = nn.Linear(d_model, d_model)

    def extract_features(self, img):
        x = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)      # semua preset: normalisasi ImageNet
        c3 = self.trunk(x)
        b, h, w, c = c3.shape
        assert (h, w) == (48, 64), (self.preset, c3.shape)
        return self.proj(c3.reshape(b, h * w, c))

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2)
        f1 = F.normalize(self.matching_head(feat1), p=2, dim=-1)
        f2 = F.normalize(self.matching_head(feat2), p=2, dim=-1)
        conf = torch.einsum('b n d, b m d -> b n m', f1, f2) / 0.1
        return F.softmax(conf, dim=1) * F.softmax(conf, dim=2)

    def load_matcher_from(self, ckpt_path, verbose=True):
        """Hangatkan HANYA layers.* dan matching_head.* dari checkpoint usulan (tidak bergantung tulang punggung)."""
        sd = torch.load(ckpt_path, map_location="cpu")
        sub = {k: v for k, v in sd.items() if k.startswith(("layers.", "matching_head."))}
        miss, unexp = self.load_state_dict(sub, strict=False)
        if verbose:
            print(f"[warm-start matcher] {len(sub)} tensor dari {ckpt_path}; tak terduga {len(unexp)}")
        return len(sub)


def build(preset, **kw):
    return BBMatcher(preset, **kw)
