"""P2c -- ASpanFormer outdoor (tp_matchers; jalur demo online_resize=True: pos-enc diskalakan ke TRAIN_RES 832,
ds=[4,4]; 480x640 habis dibagi 32 -> tanpa resize). (a) coarse -> conf_matrix, (b) e2e statis top-K.
Pakai: python revisi_d/scripts/p2c_aspanformer.py [K] [e2e]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F
import tp_matchers as TM

METHOD = "aspanformer"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


def fp16_fixes(m):
    """PE 'online scaling' dibangkitkan FP32 tiap forward (konstanta saat tracing) -> cast ke dtype fitur,
    agar seluruh graf tetap FP16 (tanpa ini x+pe naik ke FP32 dan conv berikutnya gagal tipe)."""
    pe = m.pos_encoding
    orig = pe.forward

    def f(x, scaling=None):
        a, b = orig(x, scaling)
        return a.to(x.dtype), b.to(x.dtype)
    pe.forward = f
    # partition_token: grid = sample_pixel / torch.tensor([...]).cuda() (FP32) -> grid_sample dihitung FP32
    # (masukan k/v dicast FP32, hasil kembali ke FP16); kode repo sendiri tidak diubah.
    import src.ASpanFormer.aspan_module.attention as AT

    class GS(C._FShim):
        def grid_sample(self, inp, grid, *a, **kw):
            import torch.nn.functional as F_
            if inp.dtype != grid.dtype:
                return F_.grid_sample(inp.float(), grid.float(), *a, **kw).to(inp.dtype)
            return F_.grid_sample(inp, grid, *a, **kw)
    AT.F = GS()
    return m


class CoarseConf(nn.Module):
    """= ASpanFormer.forward(online_resize=True) s.d. conf_matrix (backbone, PE berskala, transformer
    ASpan kasar global-lokal dgn flow, dual-softmax: einsum * temperature)."""
    def __init__(self, m):
        super().__init__(); self.m = fp16_fixes(m)

    def forward(self, img0, img1):
        m = self.m
        tr = m.config['coarse']['train_res']
        th, tw = (tr, tr) if len(tr) == 1 else (tr[0], tr[1])
        ps = [th / img0.shape[2], tw / img0.shape[3]]
        feats_c, _ = m.backbone(torch.cat([img0, img1], 0))
        c0, c1 = feats_c.split(1)
        [c0, p0], [c1, p1] = m.pos_encoding(c0, ps), m.pos_encoding(c1, ps)
        c0, c1, _ = m.loftr_coarse(c0, c1, p0, p1, None, None, [4, 4], [4, 4])
        c0, c1 = c0 / c0.shape[-1] ** .5, c1 / c1.shape[-1] ** .5
        sim = torch.einsum("nlc,nsc->nls", c0, c1) * m.coarse_matching.temperature
        return F.softmax(sim, 1) * F.softmax(sim, 2)


class E2E(nn.Module):
    def __init__(self, m, thr, border, hw_c, scale, K):
        super().__init__(); self.m = fp16_fixes(m)
        C.shim_F(sys.modules[type(m.fine_preprocess).__module__])
        C.wrap_int_hw(m.fine_preprocess); C.wrap_int_hw(m.fine_matching)

        def gcm(conf, data):
            d = C.static_coarse_match(conf, thr, border, hw_c, hw_c, K, scale)
            d["gt_mask"] = torch.zeros_like(d["i_ids"], dtype=torch.bool)
            return d
        m.coarse_matching.get_coarse_match = gcm
        m.coarse_matching.get_offset_match = lambda *a, **k: None   # hanya visualisasi flow (torch.where), tak memengaruhi kecocokan

    def forward(self, img0, img1):
        data = {"image0": img0, "image1": img1}
        self.m(data, online_resize=True)
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]


def main():
    m0, cfg = TM.BUILDERS[METHOD](); m0 = m0.cuda().eval()
    g = C.load_pair_gray()
    d = {"image0": g[0], "image1": g[1]}
    with torch.no_grad():
        m0(d, online_resize=True)
    ref_full = (d["mkpts0_f"].float().cpu().numpy(), d["mkpts1_f"].float().cpu().numpy())
    cm = m0.coarse_matching
    hw_c = tuple(int(v) for v in d["hw0_c"]); scale = C.H / hw_c[0]
    print("M asli:", len(ref_full[0]), "hw_c", hw_c, "thr", cm.thr, "border", cm.border_rm, cm.match_type)
    with torch.no_grad():
        c32 = CoarseConf(m0)(*g)
    print("kontrol CoarseConf vs conf_matrix asli:", float((c32 - d["conf_matrix"]).abs().max()))
    g16 = [x.half() for x in g]
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse", CoarseConf(copy.deepcopy(m0)).half().eval(), g16, ["image0", "image1"],
                   ["conf_matrix"], ref32=d["conf_matrix"],
                   changes="tidak ada (backbone ResNet-FPN + transformer ASpan global-lokal [grid_sample] + dual-softmax)")
        torch.cuda.empty_cache()
    args = (cm.thr, cm.border_rm, hw_c, scale, K)
    sv = C.static_vs_original(E2E(copy.deepcopy(m0), *args).eval(), g, ref_full)
    print("static fp32 vs asli:", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(m0), *args).half().eval(), g16,
               ["image0", "image1"], ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=f"get_coarse_match -> top-K statis (K={K}; mutual-NN+thr {cm.thr}+border {cm.border_rm}); "
                       "get_offset_match (visualisasi flow, torch.where) dimatikan; entri tak valid tetap melalui "
                       "tahap fine (mconf=0); stride F.unfold & hw dijadikan int; koordinat kasar FP32",
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
