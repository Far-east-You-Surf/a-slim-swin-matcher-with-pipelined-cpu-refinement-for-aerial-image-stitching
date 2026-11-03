"""P2c -- LoFTR (kornia, pretrained='outdoor', = stitch_demo.LoFTRModel): (a) coarse -> conf_matrix, (b) e2e statis.
Pakai: python revisi_d/scripts/p2c_loftr.py [K] [e2e]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F

METHOD = "loftr"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


class CoarseConf(nn.Module):
    """backbone ResNet-FPN -> PE -> transformer kasar (linear attn) -> dual-softmax (rumus kornia CoarseMatching)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, img0, img1):
        m = self.m
        feats_c, _ = m.backbone(torch.cat([img0, img1], 0))
        c0, c1 = feats_c.split(1)
        c0 = m.pos_encoding(c0).flatten(2).transpose(1, 2)
        c1 = m.pos_encoding(c1).flatten(2).transpose(1, 2)
        c0, c1 = m.loftr_coarse(c0, c1, None, None)
        c0, c1 = c0 / c0.shape[-1] ** .5, c1 / c1.shape[-1] ** .5
        sim = torch.einsum("nlc,nsc->nls", c0, c1) / m.coarse_matching.temperature
        return F.softmax(sim, 1) * F.softmax(sim, 2)


class E2E(nn.Module):
    def __init__(self, m, thr, border, hw_c, scale, K):
        super().__init__(); self.m = m
        import kornia.feature.loftr.loftr_module.fine_preprocess as FP
        C.shim_F(FP)
        C.wrap_int_hw(m.fine_preprocess); C.wrap_int_hw(m.fine_matching)
        m.coarse_matching.get_coarse_match = lambda conf, data: C.static_coarse_match(
            conf, thr, border, hw_c, hw_c, K, scale)

    def forward(self, img0, img1):
        out = self.m({"image0": img0, "image1": img1})
        return out["keypoints0"], out["keypoints1"], out["confidence"]


def main():
    import kornia.feature as KF
    m0 = KF.LoFTR(pretrained="outdoor").cuda().eval()
    g = C.load_pair_gray()
    with torch.no_grad():
        o = m0({"image0": g[0], "image1": g[1]})
    ref_full = (o["keypoints0"].float().cpu().numpy(), o["keypoints1"].float().cpu().numpy())
    cm = m0.coarse_matching
    hw_c = (C.H // 8, C.W // 8); scale = 8.0
    with torch.no_grad():
        conf32 = CoarseConf(m0)(*g)
    print("M asli:", len(ref_full[0]), "thr", cm.thr, "border", cm.border_rm, "match_type", cm.match_type)
    g16 = [x.half() for x in g]
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse", CoarseConf(copy.deepcopy(m0)).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=conf32,
                   changes="tidak ada (ResNet-FPN + PE + 4x(self,cross) linear attention + dual-softmax)")
        torch.cuda.empty_cache()
    args = (cm.thr, cm.border_rm, hw_c, scale, K)
    sv = C.static_vs_original(E2E(copy.deepcopy(m0), *args).eval(), g, ref_full)
    print("static fp32 vs asli:", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(m0), *args).half().eval(), g16,
               ["image0", "image1"], ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=f"get_coarse_match -> top-K statis (K={K}; mutual-NN+thr {cm.thr}+border {cm.border_rm}); "
                       "entri tak valid tetap melalui tahap fine (mconf=0); stride F.unfold & hw dijadikan int; "
                       "koordinat kasar FP32",
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
