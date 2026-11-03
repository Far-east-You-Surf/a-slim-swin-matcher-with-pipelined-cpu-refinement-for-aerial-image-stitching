"""P2c -- EfficientLoFTR (full, outdoor, reparameterized): (a) coarse -> conf_matrix, (b) e2e statis top-K.
Pakai: python revisi_d/scripts/p2c_efficientloftr.py [K]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F
import tp_matchers as TM

METHOD = "efficientloftr"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


class CoarseConf(nn.Module):           # = scripts/e12_eloftr_engine.CoarseConf
    def __init__(self, m, t):
        super().__init__(); self.m, self.t = m, t

    def forward(self, img0, img1):
        f = self.m.backbone(torch.cat([img0, img1], 0))["feats_c"]
        f0, f1 = self.m.loftr_coarse(f[:1], f[1:], None, None)
        f0 = f0.flatten(2).transpose(1, 2); f1 = f1.flatten(2).transpose(1, 2)
        f0, f1 = f0 / f0.shape[-1] ** .5, f1 / f1.shape[-1] ** .5
        sim = torch.einsum("nlc,nsc->nls", f0, f1) / self.t
        return F.softmax(sim, 1) * F.softmax(sim, 2)


class E2E(nn.Module):
    """forward asli LoFTR(data) tanpa diubah, kecuali coarse_matching.get_coarse_match diganti versi
    statis top-K (p2c_common.static_coarse_match) dan replace_nan=False (cabang isnan data-dependen)."""
    def __init__(self, m, cfg, K):
        super().__init__(); self.m = m
        mc = cfg["match_coarse"]
        m.config["replace_nan"] = False
        import src.loftr.loftr_module.fine_preprocess as FP
        C.shim_F(FP)          # stride F.unfold = tensor-0d saat tracing -> int
        hw = (C.H // 8, C.W // 8)
        m.coarse_matching.get_coarse_match = lambda conf, data: C.static_coarse_match(
            conf, mc["thr"], mc["border_rm"], hw, hw, K, 8.0)

    def forward(self, img0, img1):
        data = {"image0": img0, "image1": img1}
        self.m(data)
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]


def main():
    g = C.load_pair_gray()
    m0, cfg = TM.BUILDERS[METHOD](); m0 = m0.cuda().eval()
    d = {"image0": g[0], "image1": g[1]}
    with torch.no_grad():
        m0(d)
    ref_full = (d["mkpts0_f"].float().cpu().numpy(), d["mkpts1_f"].float().cpu().numpy())
    conf32 = d["conf_matrix"]
    t = cfg["match_coarse"]["dsmax_temperature"]
    g16 = [x.half() for x in g]
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse", CoarseConf(copy.deepcopy(m0), t).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=conf32,
                   changes="tidak ada (jalur kasar = backbone RepVGG + transformer kasar + dual-softmax; = E1.2)")
        torch.cuda.empty_cache()
    e32 = E2E(copy.deepcopy(m0), copy.deepcopy(cfg), K).eval()
    sv = C.static_vs_original(e32, g, ref_full); del e32
    print("static fp32 vs asli:", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(m0), copy.deepcopy(cfg), K).half().eval(), g16,
               ["image0", "image1"], ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=f"get_coarse_match -> top-K statis (K={K}; mutual-NN+thr {cfg['match_coarse']['thr']}+border "
                       f"{cfg['match_coarse']['border_rm']} via max-baris/kolom, TopK atas skor baris); entri tak valid "
                       "tetap melalui tahap fine (mconf=0); replace_nan=False; koordinat kasar FP32",
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
