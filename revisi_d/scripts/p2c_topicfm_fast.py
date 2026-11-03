"""P2c -- TopicFM-fast (stitch_demo.TopicFMModel('fast'), thr 0.25): (a) coarse -> conf_matrix, (b) e2e statis.
Pakai: python revisi_d/scripts/p2c_topicfm_fast.py [K] [e2e]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F

METHOD = "topicfm_fast"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


class CoarseConf(nn.Module):
    """backbone FPN -> TopicFormer -> conf = dual-softmax(einsum/sqrt(C)) PERSIS rumus CoarseMatching repo."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, img0, img1):
        feats_c, _ = self.m.backbone(torch.cat([img0, img1], 0))
        c0, c1 = feats_c.split(1)
        c0 = c0.flatten(2).transpose(1, 2); c1 = c1.flatten(2).transpose(1, 2)
        c0, c1, _ = self.m.coarse_net(c0, c1, None, None)
        sim = torch.einsum("nlc,nsc->nls", c0, c1) / c0.shape[-1] ** .5
        return F.softmax(sim, 1) * F.softmax(sim, 2)


class E2E(nn.Module):
    """forward asli TopicFM(data); get_coarse_match diganti versi statis top-K; gt_mask = semua False
    (agar filter `true_matches` tidak dinamis; entri tak valid berakhir mconf=0 karena mconf_fine*mconf_coarse)."""
    def __init__(self, m, thr, border, hw_c, scale, K):
        super().__init__(); self.m = m
        import src.models.modules.fine_preprocess as FP
        C.shim_F(FP)
        C.wrap_int_hw(m.fine_preprocess); C.wrap_int_hw(m.fine_matching)

        def gcm(conf_mask, data):
            d = C.static_coarse_match(data["conf_matrix"], thr, border, hw_c, hw_c, K, scale)
            d.update(gt_mask=torch.zeros_like(d["i_ids"], dtype=torch.bool),
                     all_mkpts0_c=d["mkpts0_c"], all_mkpts1_c=d["mkpts1_c"])
            return d
        m.coarse_matching.get_coarse_match = gcm
        fm = m.fine_matching

        def gfm(data, coords0_normed, feat_f0_picked, feat_f1, mask_b_ids=None):
            # = DynamicFineMatching.get_fine_match TANPA indeks boolean (mask_b_ids / true_matches
            # selalu semua-True di inferensi, tetapi indeks boolean -> NonZero -> bentuk dinamis di ONNX)
            sim = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1) / fm.C ** .5
            heat1 = torch.softmax(sim, dim=1)
            mconf, _ = torch.max(heat1, dim=1)
            from kornia.geometry.subpix import dsnt
            c1 = dsnt.spatial_expectation2d(heat1.view(-1, fm.W, fm.W)[None], True)[0]
            k0 = data['all_mkpts0_c'] + coords0_normed * (fm.W // 2) * fm.scale0
            k1 = data['all_mkpts1_c'] + c1 * (fm.W // 2) * fm.scale1
            return {"mconf": mconf, "mkpts0_f": k0, "mkpts1_f": k1, "all_mkpts0_f": k0, "all_mkpts1_f": k1}
        fm.get_fine_match = gfm

    def forward(self, img0, img1):
        data = {"image0": img0, "image1": img1}
        self.m(data)
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]


def main():
    import stitch_demo as SD
    w = SD.TopicFMModel("fast")
    m0 = w.model.cuda().eval()
    g = C.load_pair_gray()
    d = {"image0": g[0], "image1": g[1]}
    with torch.no_grad():
        m0(d)
    ref_full = (d["mkpts0_f"].float().cpu().numpy(), d["mkpts1_f"].float().cpu().numpy())
    hw_c = tuple(int(v) for v in d["hw0_c"]); scale = C.H / hw_c[0]
    cm = m0.coarse_matching
    print("M asli:", len(ref_full[0]), "hw_c", hw_c, "thr", cm.thr, "border", cm.border_rm)
    g16 = [x.half() for x in g]
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse", CoarseConf(copy.deepcopy(m0)).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=d["conf_matrix"],
                   changes="tidak ada (backbone FPN + TopicFormer n_samples=0 + dual-softmax einsum/sqrt(C) = rumus repo)")
        torch.cuda.empty_cache()
    args = (cm.thr, cm.border_rm, hw_c, scale, K)
    sv = C.static_vs_original(E2E(copy.deepcopy(m0), *args).eval(), g, ref_full)
    print("static fp32 vs asli:", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(m0), *args).half().eval(), g16,
               ["image0", "image1"], ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=f"get_coarse_match -> top-K statis (K={K}; mutual-NN+thr {cm.thr}+border {cm.border_rm}); "
                       "get_fine_match tanpa indeks boolean (mask_b_ids/true_matches = semua True di inferensi); entri tak valid mconf=0; "
                       "stride F.unfold dijadikan int; koordinat kasar FP32",
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
