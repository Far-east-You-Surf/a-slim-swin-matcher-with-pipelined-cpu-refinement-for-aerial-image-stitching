"""
Varian model.py dengan attention block memakai flash attention eksplisit
(F.scaled_dot_product_attention), BUKAN nn.MultiheadAttention.

Kenapa perlu modul terpisah (bukan cukup pasang need_weights=False di
nn.MultiheadAttention): dibuktikan empiris (2026-07-12) bahwa jalur cepat
internal nn.MultiheadAttention (need_weights=False) memilih backend cuDNN
(~0.039ms/panggilan utk shape head_dim=16, seq=768), BUKAN flash attention
(~0.0185ms/panggilan, ~2.1x lebih cepat lagi) -- meski keduanya lebih cepat
drpd jalur lambat need_weights=True (~0.074ms) yang dipakai model.py
ASLI (bug ringan: attention weights dihitung & dibuang via `_`, padahal
tidak pernah dipakai). F.scaled_dot_product_attention() dipanggil LANGSUNG
(tanpa lewat nn.MultiheadAttention) memilih flash attention scr otomatis
utk shape ini di RTX 5090 -- diverifikasi via torch.nn.attention.sdpa_kernel
forcing tiap backend satu per satu.

FlashMHA di bawah SENGAJA memakai nama parameter identik dgn
nn.MultiheadAttention (in_proj_weight/in_proj_bias/out_proj.weight/bias)
supaya checkpoint FP16 yang SUDAH dilatih (aerial_stitching_fp16_best.pth,
dst.) bisa langsung di-load_state_dict() TANPA retraining -- ini murni
penggantian KERNEL komputasi attention, bukan perubahan matematika/arsitektur,
jadi bobot yang sudah dipelajari tetap valid. Numerical equivalence thd
nn.MultiheadAttention asli diverifikasi terpisah (lihat verify_flash_equivalence.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights


class FlashMHA(nn.Module):
    """Drop-in pengganti nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
    -- parameter sama persis, forward pakai F.scaled_dot_product_attention."""
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value):
        B, Nq, E = query.shape
        Nk = key.shape[1]

        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        q = q.view(B, Nq, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, Nq, E)
        return self.out_proj(attn_out), None


class LoFTRAttentionBlockFlash(nn.Module):
    """Identik dgn LoFTRAttentionBlock (model.py), self_attn/cross_attn diganti FlashMHA."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = FlashMHA(d_model, nhead)
        self.cross_attn = FlashMHA(d_model, nhead)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):
        feat1_sa, _ = self.self_attn(feat1, feat1, feat1)
        feat1 = self.norm1(feat1 + feat1_sa)

        feat2_sa, _ = self.self_attn(feat2, feat2, feat2)
        feat2 = self.norm1(feat2 + feat2_sa)

        feat1_ca, _ = self.cross_attn(feat1, feat2, feat2)
        feat1 = self.norm2(feat1 + feat1_ca)

        feat2_ca, _ = self.cross_attn(feat2, feat1, feat1)
        feat2 = self.norm2(feat2 + feat2_ca)

        feat1 = self.norm3(feat1 + self.ffn(feat1))
        feat2 = self.norm3(feat2 + self.ffn(feat2))

        return feat1, feat2


class AerialStitchingModelFlash(nn.Module):
    """Identik dgn AerialStitchingModel (model.py) scr struktur/state_dict-key
    (backbone/proj/layers/matching_head) -- HANYA blok attention internal
    (LoFTRAttentionBlock -> LoFTRAttentionBlockFlash) yg beda kernel komputasi.
    load_state_dict() dari checkpoint model.py asli harus 1:1 cocok tanpa remap."""
    def __init__(self, d_model=128, nhead=8, num_layers=4, use_fp8=False):
        super().__init__()
        self.target_size = (768, 1024)

        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.backbone = swin.features

        swin_out_dim = 1024
        self.proj = nn.Linear(swin_out_dim, d_model)

        self.layers = nn.ModuleList([
            LoFTRAttentionBlockFlash(d_model, nhead) for _ in range(num_layers)
        ])

        self.matching_head = nn.Linear(d_model, d_model)
        self.use_fp8 = use_fp8

    def extract_features(self, img):
        img_resized = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
        features = self.backbone(img_resized)
        if features.dim() == 4:
            B, H, W, C = features.shape
            features = features.view(B, H * W, C)
        features = self.proj(features)
        return features

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2)

        feat1_match = F.normalize(self.matching_head(feat1), p=2, dim=-1)
        feat2_match = F.normalize(self.matching_head(feat2), p=2, dim=-1)

        conf_matrix = torch.einsum('b n d, b m d -> b n m', feat1_match, feat2_match)
        conf_matrix = conf_matrix / 0.1
        conf_matrix = F.softmax(conf_matrix, dim=1) * F.softmax(conf_matrix, dim=2)

        return conf_matrix
