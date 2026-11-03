"""
§56 -- ABLASI TERBALIK: model hibrida custom dgn attention LINEAR, bobot identik.

Kenapa ini ada. §26 menjalankan ablasi variabel tunggal pada LoFTR asli dgn arah
LINEAR -> SOFTMAX, dan kerapuhan terhadap kuantisasi aktivasi MUNCUL. Arah
sebaliknya belum pernah diuji: model hibrida proyek ini (Swin-B + 4 blok
LoFTR-style ber-nn.MultiheadAttention, 82 node Softmax) ditukar SOFTMAX -> LINEAR.
Kalau kepadatan softmax memang yang menentukan, kerapuhan harus HILANG.

Nilainya: dua ablasi kausal, dua arah berlawanan, dua arsitektur berbeda, keduanya
pada bobot byte-identik. Itu jauh lebih kuat daripada menambah titik korelasional
ketiga, dan menutup bantahan "efeknya artefak implementasi kornia".

Cara kerjanya sama persis dgn model_flash.py (§20), yang sudah membuktikan pola
ini di proyek ini: LinearMHA memakai NAMA DAN BENTUK PARAMETER yang identik dgn
nn.MultiheadAttention (in_proj_weight / in_proj_bias / out_proj.weight / bias),
jadi checkpoint FP16 yang sudah dilatih dimuat lewat load_state_dict(strict=True)
TANPA remap dan TANPA retraining. Yang berubah hanya CARA menghitung attention --
dan operasi itu tak berparameter di kedua varian, persis seperti LinearAttention
vs FullAttention di kornia.

Rumusan linear attention sengaja disamakan dgn kornia (Katharopoulos et al., ICML
2020): peta fitur elu(x)+1, lalu (phi(Q) (phi(K)^T V)) dinormalkan oleh
phi(Q) . sum(phi(K)). Tidak ada eksponensiasi pada matriks attention, jadi tidak
ada penguatan galat ber-gain tinggi. Penskalaan 1/sqrt(d) ikut hilang karena
memang tidak ada dalam rumusan linear attention.

Yang TIDAK diubah, dan penting untuk kesahihan ablasi: dual-softmax di ujung
(`conf_matrix`) TETAP ADA. Varian linear karena itu menyisakan 2 node Softmax,
persis sama dgn varian linear LoFTR di §25/§26 -- jadi kedua sisi perbandingan
dalam kedua ablasi punya struktur ujung yang sama.

BATAS YANG WAJIB DINYATAKAN DI PAPER: bobot ini dilatih DENGAN softmax attention,
jadi varian linear pasti lebih buruk secara absolut. Ini bayangan cermin dari batas
§26 (varian full LoFTR memakai bobot yang dilatih untuk linear). Sah untuk
pertanyaan SENSITIVITAS -- tiap varian diukur terhadap referensi FP16-nya sendiri --
TIDAK sah sebagai klaim kualitas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights


def elu_feature_map(x):
    return F.elu(x) + 1.0


class LinearMHA(nn.Module):
    """Drop-in pengganti nn.MultiheadAttention(embed_dim, nhead, batch_first=True):
    parameter identik, attention dihitung tanpa softmax (Katharopoulos et al.)."""

    def __init__(self, embed_dim, nhead, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim
        self.eps = eps

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

        q = F.linear(query, w_q, b_q).view(B, Nq, self.nhead, self.head_dim)
        k = F.linear(key, w_k, b_k).view(B, Nk, self.nhead, self.head_dim)
        v = F.linear(value, w_v, b_v).view(B, Nk, self.nhead, self.head_dim)

        Q = elu_feature_map(q)
        K = elu_feature_map(k)

        # normalisasi panjang seperti kornia, supaya KV tidak meledak di fp16
        v = v / Nk
        KV = torch.einsum("nshd,nshv->nhdv", K, v)
        Z = 1.0 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        out = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * Nk

        return self.out_proj(out.reshape(B, Nq, E)), None


class LoFTRAttentionBlockLinear(nn.Module):
    """Identik dgn LoFTRAttentionBlock (model.py); self_attn/cross_attn -> LinearMHA."""

    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = LinearMHA(d_model, nhead)
        self.cross_attn = LinearMHA(d_model, nhead)

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


class AerialStitchingModelLinear(nn.Module):
    """Kunci state_dict identik dgn AerialStitchingModel (model.py).
    load_state_dict(strict=True) dari checkpoint asli harus cocok 1:1."""

    def __init__(self, d_model=128, nhead=8, num_layers=4, use_fp8=False):
        super().__init__()
        self.target_size = (768, 1024)

        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.backbone = swin.features

        self.proj = nn.Linear(1024, d_model)
        self.layers = nn.ModuleList([
            LoFTRAttentionBlockLinear(d_model, nhead) for _ in range(num_layers)
        ])
        self.matching_head = nn.Linear(d_model, d_model)
        self.use_fp8 = use_fp8

    def extract_features(self, img):
        img_resized = F.interpolate(img, size=self.target_size, mode='bilinear',
                                    align_corners=False)
        features = self.backbone(img_resized)
        if features.dim() == 4:
            B, H, W, C = features.shape
            features = features.view(B, H * W, C)
        return self.proj(features)

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2)

        feat1_match = F.normalize(self.matching_head(feat1), p=2, dim=-1)
        feat2_match = F.normalize(self.matching_head(feat2), p=2, dim=-1)

        conf_matrix = torch.einsum('b n d, b m d -> b n m', feat1_match, feat2_match)
        conf_matrix = conf_matrix / 0.1
        # dual-softmax ujung SENGAJA dipertahankan: varian linear menyisakan 2 node
        # Softmax, sama persis dgn varian linear LoFTR di §25/§26.
        conf_matrix = F.softmax(conf_matrix, dim=1) * F.softmax(conf_matrix, dim=2)
        return conf_matrix
