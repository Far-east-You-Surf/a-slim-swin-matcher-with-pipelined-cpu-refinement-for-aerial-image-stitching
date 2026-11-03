"""
Arsitektur usulan Artikel D: pencocokan pada grid /16 dgn realokasi komputasi.

Turunan langsung dari pengukuran kita sendiri (log §58):
  - Kedalaman stage 3 terbukti 56% redundan (§24.5: 18->8 blok gratis) -> ambil anggarannya.
  - Jumlah token TIDAK redundan (§24.5: B3 gagal) -> belanjakan anggarannya di resolusi grid.
  - Grid 768 JENUH pada 270.9 inlier out-of-domain (§58.3) sedangkan grid 3072 mencapai 542.7.
  - Sensitivitas berjalan BERLAWANAN dgn porsi FLOPs (§34) -> matcher itu leher botol, bukan backbone.

Perubahan struktural vs `model.py`:
  1. Stage 3 Swin dipangkas 18 -> `stage3_blocks` blok (per-PASANGAN, menjaga selang-seling shift).
  2. Pencocokan pindah dari /32 (24x32 = 768 token) ke /16 (48x64 = 3072 token).
  3. `proj` menjadi 512->d_model (bukan 1024->d_model).
  4. OPSIONAL `use_fpn`: stage 4 dipertahankan sbg jalur konteks, hasilnya diproyeksikan
     1024->512 lalu di-upsample dan DIJUMLAHKAN ke fitur /16. Ini satu-satunya bagian
     usulan yang benar-benar baru: grid korespondensi rapat yang tetap membawa konteks
     semantik stage 4. Tanpa ini, membuang stage 4 juga membuang 2 blok penalaran global.
  5. `flash=True` memakai FlashMHA (F.scaled_dot_product_attention) di matcher. Ini BUKAN
     perubahan matematika -- parameter identik nn.MultiheadAttention -- melainkan syarat
     agar training muat: attention 3072 token yang dimaterialisasi butuh ~9.7 GB hanya
     untuk backward. Catatan §20 justru memastikan ini aman untuk deployment: eksporter
     legacy mendekomposisi SDPA menjadi MatMul+Softmax+MatMul, jadi graf ONNX-nya identik
     dgn versi nn.MultiheadAttention. Kegagalan pilar #2 menjadi pemungkin pilar ini.

Bobot: backbone + matcher + matching_head DIHANGATKAN dari checkpoint aerial yang ada
(`load_pretrained_from`), bukan dari ImageNet -- hanya `proj` (dan `lateral` bila FPN
dipakai) yang berinisialisasi acak.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights

from model import LoFTRAttentionBlock
from model_flash import LoFTRAttentionBlockFlash

STAGE3_IDX = 5          # features[5] = stage 3 (18 blok, 512 kanal, /16)
STAGE3_CHANNELS = 512
STAGE4_CHANNELS = 1024


def kept_stage3_indices(n_blocks_total, keep_blocks):
    """Indeks blok stage 3 yang dipertahankan -- SAMA PERSIS dgn gate_pilar3._prune_stage3
    supaya varian yang dilatih di sini identik dgn yang sudah lolos gerbang §58.4."""
    assert keep_blocks % 2 == 0, "keep_blocks harus genap (jaga pasangan shifted/non-shifted)"
    n_pairs_avail = n_blocks_total // 2
    pair_idx = np.linspace(0, n_pairs_avail - 1, keep_blocks // 2).round().astype(int)
    out = []
    for p in pair_idx:
        out += [2 * int(p), 2 * int(p) + 1]
    return out


class AerialStitchingModelP16(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=8, stage3_blocks=8,
                 use_fpn=True, flash=True, pretrained_backbone=True):
        super().__init__()
        self.target_size = (768, 1024)
        self.use_fpn = use_fpn
        self.d_model = d_model

        weights = Swin_B_Weights.DEFAULT if pretrained_backbone else None
        feats = list(swin_b(weights=weights).features)

        stage3 = feats[STAGE3_IDX]
        self.stage3_blocks_before = len(stage3)
        if stage3_blocks is not None and stage3_blocks != len(stage3):
            self.kept_block_indices = kept_stage3_indices(len(stage3), stage3_blocks)
            feats[STAGE3_IDX] = nn.Sequential(*[stage3[i] for i in self.kept_block_indices])
        else:
            self.kept_block_indices = list(range(len(stage3)))

        # jalur utama berhenti tepat sesudah stage 3 -> [B, 48, 64, 512]
        self.trunk = nn.Sequential(*feats[:STAGE3_IDX + 1])

        if use_fpn:
            # jalur konteks: merge /32 + stage 4 -> [B, 24, 32, 1024]
            self.down = nn.Sequential(feats[STAGE3_IDX + 1], feats[STAGE3_IDX + 2])
            self.lateral = nn.Linear(STAGE4_CHANNELS, STAGE3_CHANNELS)

        self.proj = nn.Linear(STAGE3_CHANNELS, d_model)

        block_cls = LoFTRAttentionBlockFlash if flash else LoFTRAttentionBlock
        self.layers = nn.ModuleList([block_cls(d_model, nhead) for _ in range(num_layers)])
        self.matching_head = nn.Linear(d_model, d_model)

    def extract_features(self, img):
        x = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False)
        c3 = self.trunk(x)                                  # [B, H, W, 512]
        if self.use_fpn:
            c4 = self.lateral(self.down(c3))                # [B, H/2, W/2, 512]
            c4 = F.interpolate(c4.permute(0, 3, 1, 2), size=c3.shape[1:3],
                               mode="bilinear", align_corners=False)
            c3 = c3 + c4.permute(0, 2, 3, 1)
        b, h, w, c = c3.shape
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

    # ------------------------------------------------------------------
    def load_pretrained_from(self, ckpt_path, verbose=True):
        """Hangatkan dari checkpoint aerial yang ada. Blok stage 3 dipetakan lewat
        INDEKS ASLI-nya (bukan berurutan) -- pemangkasan menomori ulang blok, jadi
        memuat berurutan akan memasang bobot blok yang salah."""
        sd = torch.load(ckpt_path, map_location="cpu")
        own = self.state_dict()
        remap = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                rest = k[len("backbone."):]
                head, _, tail = rest.partition(".")
                idx = int(head)
                if idx < STAGE3_IDX:
                    nk = f"trunk.{idx}.{tail}"
                elif idx == STAGE3_IDX:
                    blk, _, sub = tail.partition(".")
                    if int(blk) not in self.kept_block_indices:
                        continue                     # blok ini dipangkas
                    nk = f"trunk.{STAGE3_IDX}.{self.kept_block_indices.index(int(blk))}.{sub}"
                elif self.use_fpn:
                    nk = f"down.{idx - (STAGE3_IDX + 1)}.{tail}"
                else:
                    continue                         # stage 4 dibuang
            else:
                nk = k
            if nk in own and own[nk].shape == v.shape:
                remap[nk] = v
        self.load_state_dict(remap, strict=False)
        fresh = sorted({k.split(".")[0] for k in own if k not in remap})
        if verbose:
            print(f"[warm-start] {len(remap)}/{len(own)} tensor dimuat dari {ckpt_path}; "
                  f"modul berinisialisasi acak: {fresh}")
        return dict(loaded=len(remap), total=len(own), fresh_modules=fresh)


def build(variant="p16_depth8_fpn", **kw):
    presets = {
        "p16_depth8_fpn":   dict(stage3_blocks=8, use_fpn=True),
        "p16_depth8":       dict(stage3_blocks=8, use_fpn=False),
        "p16_depth12_fpn":  dict(stage3_blocks=12, use_fpn=True),
        # varian gerbang pemangkasan lanjutan (§66): dipangkas ZERO-SHOT dari
        # checkpoint depth8 yang sudah dilatih, bukan dilatih dari awal.
        "p16_depth6":       dict(stage3_blocks=6, use_fpn=False),
        "p16_depth4":       dict(stage3_blocks=4, use_fpn=False),
    }
    cfg = dict(presets[variant])
    cfg.update(kw)
    return AerialStitchingModelP16(**cfg)
