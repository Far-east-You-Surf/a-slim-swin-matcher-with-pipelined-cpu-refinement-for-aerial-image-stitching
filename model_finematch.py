import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights
from model import LoFTRAttentionBlock


class FineMatchingHead(nn.Module):
    """
    Modul refinement ringan gaya LoFTR: untuk tiap match coarse, ambil satu
    vektor fitur pusat dari img1 (level fine) dan window kecil di sekitar
    lokasi coarse-nya di img2 (level fine), lalu korelasi + soft-argmax
    untuk regresi offset sub-pixel.

    window = ratio + 1 (bukan angka sembarang): offset sub-pixel target di
    grid fine, relatif thd pusat sel fine yang berpadanan dengan sel koarse
    GT, secara matematis SELALU jatuh di [-ratio/2, ratio/2) -- turunan dari
    hubungan floor(pts2/scale_coarse) vs pts2/scale_fine. Window ini
    menjamin cakupan penuh tanpa pemborosan.
    """
    def __init__(self, fine_dim, ratio=4, temperature=0.1):
        super().__init__()
        self.ratio = ratio
        self.window = ratio + 1
        self.r = ratio // 2
        self.temperature = temperature

    def forward(self, fine_map1, fine_map2, coarse_hw, gt_idx2):
        # fine_map1/2: [B, Hf, Wf, Cf] (channels-last, hasil proyeksi fine_proj)
        # coarse_hw: (Hc, Wc) ukuran grid coarse
        # gt_idx2: [B, N] indeks flat koarse target di img2 utk tiap sel koarse img1 (arange(N))
        B, Hf, Wf, Cf = fine_map1.shape
        Hc, Wc = coarse_hw
        N = Hc * Wc
        device = fine_map1.device
        w = self.window

        idx1 = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        y1c, x1c = idx1 // Wc, idx1 % Wc
        y2c, x2c = gt_idx2 // Wc, gt_idx2 % Wc

        fy1 = y1c * self.ratio + self.ratio // 2
        fx1 = x1c * self.ratio + self.ratio // 2
        fy2 = y2c * self.ratio + self.ratio // 2
        fx2 = x2c * self.ratio + self.ratio // 2

        flat1 = fine_map1.reshape(B, Hf * Wf, Cf)
        center_idx = fy1 * Wf + fx1
        center_feat = torch.gather(flat1, 1, center_idx.unsqueeze(-1).expand(-1, -1, Cf))  # [B,N,Cf]

        pad = self.r
        fine_map2_p = F.pad(fine_map2.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='replicate')
        Wf_p = Wf + 2 * pad
        flat2_p = fine_map2_p.permute(0, 2, 3, 1).reshape(B, -1, Cf)

        dyx = torch.arange(w, device=device)
        ys = fy2.unsqueeze(-1) + dyx.view(1, 1, w)  # [B,N,w] indeks baris di peta ter-padding
        xs = fx2.unsqueeze(-1) + dyx.view(1, 1, w)  # [B,N,w] indeks kolom di peta ter-padding
        window_idx = ys.unsqueeze(-1) * Wf_p + xs.unsqueeze(-2)  # [B,N,w,w]
        window_idx = window_idx.reshape(B, N * w * w)
        window_feat = torch.gather(flat2_p, 1, window_idx.unsqueeze(-1).expand(-1, -1, Cf))
        window_feat = window_feat.reshape(B, N, w, w, Cf)

        corr = torch.einsum('bnc,bnijc->bnij', center_feat, window_feat) / self.temperature
        heat = F.softmax(corr.reshape(B, N, w * w), dim=-1).reshape(B, N, w, w)

        coords = torch.arange(-self.r, self.r + 1, device=device, dtype=heat.dtype)
        exp_dy = torch.einsum('bnij,i->bn', heat, coords)
        exp_dx = torch.einsum('bnij,j->bn', heat, coords)
        return torch.stack([exp_dx, exp_dy], dim=-1)  # [B,N,2] offset sub-pixel (satuan grid fine)


class AerialStitchingModelFineMatch(nn.Module):
    """
    Varian AerialStitchingModel (model.py) + tahap fine-matching refinement.
    Coarse path (Swin-B stride-32 -> LoFTR blocks -> dual-softmax) sengaja
    TIDAK diubah dari model.py, supaya perilaku coarse tetap bisa
    dibandingkan apel-ke-apel dengan baseline murni (lihat train_fp16_dataonly.py).
    Fine path menyadap keluaran Swin di stride-8 (akhir stage2) sebagai
    fitur beresolusi lebih tinggi untuk window matching lokal.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=8, fine_dim=64, fine_loss_weight=1.0):
        super().__init__()
        self.target_size = (768, 1024)
        self.fine_loss_weight = fine_loss_weight

        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        # features[0:4] = patch_embed + stage1 + patch_merge + stage2 -> stride 8, dim 256
        # features[4:8] = patch_merge + stage3 + patch_merge + stage4 -> stride 32, dim 1024
        self.backbone_fine = swin.features[:4]
        self.backbone_coarse = swin.features[4:]

        coarse_dim = 1024
        fine_src_dim = 256
        self.coarse_stride = 32
        self.fine_stride = 8
        self.ratio = self.coarse_stride // self.fine_stride  # 4

        self.coarse_proj = nn.Linear(coarse_dim, d_model)
        self.fine_proj = nn.Linear(fine_src_dim, fine_dim)

        self.layers = nn.ModuleList([
            LoFTRAttentionBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.matching_head = nn.Linear(d_model, d_model)
        self.fine_head = FineMatchingHead(fine_dim, ratio=self.ratio)

    def extract_features(self, img):
        img_resized = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
        fine_map = self.backbone_fine(img_resized)        # [B, Hf, Wf, 256]
        coarse_map = self.backbone_coarse(fine_map)        # [B, Hc, Wc, 1024]

        B, Hc, Wc, C = coarse_map.shape
        coarse_flat = self.coarse_proj(coarse_map.reshape(B, Hc * Wc, C))
        fine_proj_map = self.fine_proj(fine_map)            # [B, Hf, Wf, fine_dim]
        return coarse_flat, fine_proj_map, (Hc, Wc)

    def _coarse_forward(self, img1, img2):
        coarse1, fine1, coarse_hw = self.extract_features(img1)
        coarse2, fine2, _ = self.extract_features(img2)

        for layer in self.layers:
            coarse1, coarse2 = layer(coarse1, coarse2)

        feat1_match = F.normalize(self.matching_head(coarse1), p=2, dim=-1)
        feat2_match = F.normalize(self.matching_head(coarse2), p=2, dim=-1)
        conf_matrix = torch.einsum('bnd,bmd->bnm', feat1_match, feat2_match)
        conf_matrix = conf_matrix / 0.1
        conf_matrix = F.softmax(conf_matrix, dim=1) * F.softmax(conf_matrix, dim=2)
        return conf_matrix, fine1, fine2, coarse_hw

    def forward(self, img1, img2):
        # Forward coarse-only -- tanda tangan IDENTIK dgn model.py, supaya
        # tetap kompatibel dengan pipeline export TensorRT/kuantisasi yang
        # sudah ada apabila varian ini nanti juga mau dikuantisasi.
        conf_matrix, _, _, _ = self._coarse_forward(img1, img2)
        return conf_matrix

    def forward_with_fine(self, img1, img2, gt_idx2):
        """Dipakai saat training: forward coarse + regresi fine-matching,
        disupervisi via indeks GT (teacher forcing ala LoFTR asli -- gradien
        tidak perlu lewat argmax/threshold yang non-differentiable)."""
        conf_matrix, fine1, fine2, coarse_hw = self._coarse_forward(img1, img2)
        fine_offset = self.fine_head(fine1, fine2, coarse_hw, gt_idx2)
        return conf_matrix, fine_offset

    @torch.no_grad()
    def refine_inference(self, img1, img2, conf_threshold=0.1):
        """Inference end-to-end: ambil match koarse (thresholding, sama
        seperti extract_matches() di skrip eval), lalu perhalus tiap match
        dgn fine_head. Mengembalikan idx koarse img1, idx koarse img2, dan
        offset sub-pixel (satuan grid fine, kalikan fine_stride=8 utk piksel)
        per match. Saat ini hanya mendukung batch=1."""
        conf_matrix, fine1, fine2, coarse_hw = self._coarse_forward(img1, img2)
        B, N, _ = conf_matrix.shape
        assert B == 1, "refine_inference saat ini hanya mendukung batch=1"

        mask = conf_matrix[0] > conf_threshold
        idx = torch.nonzero(mask)  # [K,2] -> (idx1, idx2)
        if idx.numel() == 0:
            return idx[:, 0], idx[:, 1], torch.zeros((0, 2), device=img1.device)

        idx1_sel, idx2_sel = idx[:, 0], idx[:, 1]
        gt_idx2_full = torch.zeros((1, N), dtype=torch.long, device=img1.device)
        gt_idx2_full[0, idx1_sel] = idx2_sel
        fine_offset_full = self.fine_head(fine1, fine2, coarse_hw, gt_idx2_full)  # [1,N,2]
        fine_offset = fine_offset_full[0, idx1_sel]  # [K,2]
        return idx1_sel, idx2_sel, fine_offset


def compute_fine_loss(fine_offset, fine_gt_offset, valid_mask):
    diff = (fine_offset - fine_gt_offset).pow(2).sum(dim=-1)  # [B,N]
    diff = diff * valid_mask
    denom = valid_mask.sum().clamp_min(1e-6)
    return diff.sum() / denom
