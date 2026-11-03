import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights

class LoFTRAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # Q, K, V Activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Feed Forward Network (4D Linear equivalent)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):
        # Self Attention
        feat1_sa, _ = self.self_attn(feat1, feat1, feat1)
        feat1 = self.norm1(feat1 + feat1_sa)
        
        feat2_sa, _ = self.self_attn(feat2, feat2, feat2)
        feat2 = self.norm1(feat2 + feat2_sa)
        
        # Cross Attention
        feat1_ca, _ = self.cross_attn(feat1, feat2, feat2) # Q=feat1, K=feat2, V=feat2
        feat1 = self.norm2(feat1 + feat1_ca)
        
        feat2_ca, _ = self.cross_attn(feat2, feat1, feat1) # Q=feat2, K=feat1, V=feat1
        feat2 = self.norm2(feat2 + feat2_ca)
        
        # FFN
        feat1 = self.norm3(feat1 + self.ffn(feat1))
        feat2 = self.norm3(feat2 + self.ffn(feat2))
        
        return feat1, feat2

class AerialStitchingModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, use_fp8=False):
        super().__init__()
        self.target_size = (768, 1024) # H, W
        
        # Feature Extraction: Swin-Base
        # In practice, we extract features from an intermediate layer of Swin.
        # For simplicity in this skeleton, we use a pretrained Swin and project it.
        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.backbone = swin.features # Output shape is typically [B, C, H', W']
        
        # Swin-Base output dimension is 1024
        swin_out_dim = 1024
        self.proj = nn.Linear(swin_out_dim, d_model)
        
        # Transformer Blocks (Lx)
        self.layers = nn.ModuleList([
            LoFTRAttentionBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Matching Head (Linear)
        self.matching_head = nn.Linear(d_model, d_model)
        
        self.use_fp8 = use_fp8
        
    def extract_features(self, img):
        # Internal Downscaling to 1024x768
        img_resized = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Swin-Base forward
        # Expected input shape [B, 3, 768, 1024]
        # Swin outputs [B, 1024, H_feat, W_feat] where H_feat, W_feat = 768/32, 1024/32 = 24, 32
        features = self.backbone(img_resized) 
        
        # Flatten and project to d_model
        # Swin feature output shape in torchvision is [B, H, W, C]
        if features.dim() == 4:
             B, H, W, C = features.shape
             features = features.view(B, H * W, C)
        
        features = self.proj(features)
        return features

    def forward(self, img1, img2):
        # 1. Feature Extraction (fp16 default, fp8 targeted)
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        
        # 2. Transformer Blocks
        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2)
            
        # 3. Matching Head -> Confidence Matrix
        feat1_match = F.normalize(self.matching_head(feat1), p=2, dim=-1)
        feat2_match = F.normalize(self.matching_head(feat2), p=2, dim=-1)
        
        # Compute confidence matrix (Inner product)
        # feat1: [B, N, D], feat2: [B, N, D]
        # conf_matrix: [B, N, N]
        conf_matrix = torch.einsum('b n d, b m d -> b n m', feat1_match, feat2_match)
        
        # Dual-softmax matching (from LoFTR)
        conf_matrix = conf_matrix / 0.1 # temperature
        conf_matrix = F.softmax(conf_matrix, dim=1) * F.softmax(conf_matrix, dim=2)
        
        return conf_matrix

# Note on Computational Reduction (FP8 / FP4):
# For Target 1 on RTX 5090, you can leverage native PyTorch FP8 support or TorchAO.
# Example integration with TorchAO will be shown in the training script.
