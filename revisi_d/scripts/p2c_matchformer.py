"""P2c -- MatchFormer outdoor large-LA (tp_matchers): (a) coarse -> conf_matrix, (b) e2e statis top-K.
Pakai: python revisi_d/scripts/p2c_matchformer.py [K] [e2e]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F
import tp_matchers as TM

METHOD = "matchformer"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


ATTN32 = "attn16" not in sys.argv     # default: attention linear dihitung FP32 (lihat catatan)


def attn_fp32_forward(self, x):
    """= Attention.forward repo (match_LA_large) persis, tetapi feature-map/einsum/normalisasi Z dihitung FP32.
    Alasan: dlm engine TRT FP16 murni seluruh conf_matrix NaN (K.sum atas 19.200 token di tahap 1/4 dan
    value/v_length ~5e-5 berada di luar rentang normal FP16), padahal PyTorch FP16 masih hingga."""
    x_q, x_kv = x, x
    B, N, C = x_q.shape
    MiniB = B // 2
    query = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads)
    kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
    if self.cross:
        k1, k2 = kv[0].split(MiniB); v1, v2 = kv[1].split(MiniB)
        key = torch.cat([k2, k1], dim=0); value = torch.cat([v2, v1], dim=0)
    else:
        key, value = kv[0], kv[1]
    dt = query.dtype
    Q = self.feature_map(query.float()); K = self.feature_map(key.float()); value = value.float()
    v_length = value.size(1)
    value = value / v_length
    KV = torch.einsum("nshd,nshv->nhdv", K, value)
    Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
    x = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
    return x.contiguous().view(B, -1, C).to(dt)


def patch_attn(m):
    import types
    n = 0
    for mod in m.modules():
        if type(mod).__name__ == "Attention" and hasattr(mod, "feature_map"):
            mod.forward = types.MethodType(attn_fp32_forward, mod); n += 1
    return n


class CoarseConf(nn.Module):
    """backbone (attention interleaved, large-LA) -> feats 1/8 -> dual-softmax (rumus CoarseMatching repo)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, img0, img1):
        feats_c, _ = self.m.backbone(torch.cat([img0, img1], 0))
        c0, c1 = feats_c.split(1)
        c0 = c0.flatten(2).transpose(1, 2); c1 = c1.flatten(2).transpose(1, 2)
        c0, c1 = c0 / c0.shape[-1] ** .5, c1 / c1.shape[-1] ** .5
        sim = torch.einsum("nlc,nsc->nls", c0, c1) / self.m.coarse_matching.temperature
        return F.softmax(sim, 1) * F.softmax(sim, 2)


class E2E(nn.Module):
    def __init__(self, m, thr, border, hw_c, scale, K):
        super().__init__(); self.m = m
        C.shim_F(sys.modules[type(m.fine_preprocess).__module__])
        C.wrap_int_hw(m.fine_preprocess); C.wrap_int_hw(m.fine_matching)

        def gcm(conf, data):
            d = C.static_coarse_match(conf, thr, border, hw_c, hw_c, K, scale)
            d["gt_mask"] = torch.zeros_like(d["i_ids"], dtype=torch.bool)
            return d
        m.coarse_matching.get_coarse_match = gcm

    def forward(self, img0, img1):
        data = {"image0": img0, "image1": img1}
        self.m(data)
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]


def main():
    m0, cfg = TM.BUILDERS[METHOD](); m0 = m0.cuda().eval()
    g = C.load_pair_gray()
    d = {"image0": g[0], "image1": g[1]}
    with torch.no_grad():
        m0(d)
    ref_full = (d["mkpts0_f"].float().cpu().numpy(), d["mkpts1_f"].float().cpu().numpy())
    cm = m0.coarse_matching
    hw_c = tuple(int(v) for v in d["hw0_c"]); scale = C.H / hw_c[0]
    print("M asli:", len(ref_full[0]), "hw_c", hw_c, "thr", cm.thr, "border", cm.border_rm, cm.match_type)
    g16 = [x.half() for x in g]
    if ATTN32:
        print("attention FP32 dipatch:", patch_attn(m0))
    a32 = "; attention linear (elu+1) dihitung FP32 di dalam graf FP16 (FP16 murni -> engine NaN, lihat compile.json matchformer/*_attn16)" if ATTN32 else ""
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse" if ATTN32 else "coarse_attn16", CoarseConf(copy.deepcopy(m0)).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=d["conf_matrix"],
                   changes="backbone large-LA dgn self/cross linear attention + dual-softmax" + a32)
        torch.cuda.empty_cache()
    args = (cm.thr, cm.border_rm, hw_c, scale, K)
    sv = C.static_vs_original(E2E(copy.deepcopy(m0), *args).eval(), g, ref_full)
    print("static fp32 vs asli:", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    if not ATTN32:
        lvl += "_attn16"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(m0), *args).half().eval(), g16,
               ["image0", "image1"], ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=f"get_coarse_match -> top-K statis (K={K}; mutual-NN+thr {cm.thr}+border {cm.border_rm}); "
                       "entri tak valid tetap melalui tahap fine (mconf=0); stride F.unfold & hw dijadikan int; "
                       "koordinat kasar FP32" + a32,
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
