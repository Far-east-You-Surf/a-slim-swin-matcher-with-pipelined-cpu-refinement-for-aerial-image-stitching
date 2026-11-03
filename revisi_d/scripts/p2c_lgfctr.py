"""P2c -- LGFCTR outdoor (tp_matchers, config lgfctr_ds_eval): (a) coarse -> conf_matrix, (b) e2e statis top-K.
Pakai: python revisi_d/scripts/p2c_lgfctr.py [K] [e2e]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F
import tp_matchers as TM

METHOD = "lgfctr"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


ATTN32 = "attn16" not in sys.argv


def patch_attn(m):
    """LinearAttention (elu+1) dihitung FP32: q,k,v dicast FP32, keluaran kembali FP16 (kode repo tidak disalin).
    Alasan: dlm PyTorch FP16 murni de_layers.1.0.attention sudah menghasilkan inf/NaN (conf_matrix NaN)."""
    n = 0
    for mod in m.modules():
        if type(mod).__name__ == "LinearAttention":
            of = mod.forward

            def f(q, k, v, q_mask=None, kv_mask=None, of=of):
                return of(q.float(), k.float(), v.float(), q_mask, kv_mask).to(q.dtype)
            mod.forward = f; n += 1
    return n


class CoarseConf(nn.Module):
    """backbone encoder-decoder (self/cross transformer multi-skala) -> feats 1/8 -> dual-softmax (rumus repo)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, img0, img1):
        c0, _, c1, _ = self.m.backbone(img0, img1, None, None)
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
    print("M asli:", len(ref_full[0]), "hw_c", hw_c, "thr", cm.thr, "border", cm.border_rm, getattr(cm, "match_type", "dual_softmax"))
    g16 = [x.half() for x in g]
    a32 = ""
    if ATTN32:
        print("LinearAttention FP32:", patch_attn(m0))
        a32 = "; LinearAttention dihitung FP32 di dalam graf FP16 (FP16 murni: PyTorch FP16 sendiri NaN, lihat *_attn16)"
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse" if ATTN32 else "coarse_attn16", CoarseConf(copy.deepcopy(m0)).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=d["conf_matrix"],
                   changes="tidak ada (backbone LGFCTR encoder-decoder + dual-softmax)" + a32)
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
