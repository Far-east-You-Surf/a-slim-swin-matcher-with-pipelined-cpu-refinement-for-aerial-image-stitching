"""P2c -- JamMa (CVPR 2025, bobot resmi; = stitch_demo.JamMaModel, masukan RGB ternormalisasi ImageNet).
Kernel kustom mamba-ssm (selective-scan CUDA, causal-conv1d, RMSNorm Triton) TIDAK dapat ditrace (lihat
gate_arch_results/thirdparty_export/jamma.json). Jalur yang dipakai = 'mode rujukan' (gate_thirdparty_export_refmode2):
selective_scan_fn -> selective_scan_ref (loop Python per langkah -> graf ter-unroll), RMSNorm Triton -> RMSNorm PyTorch.
(a) coarse: backbone -> kenc -> JointMamba -> final_proj -> conf = softmax(sim,2)*softmax(sim,1) (batas = gate lama)
(b) e2e: forward asli JamMa, get_coarse_match_inference & get_fine_sub_match diganti versi statis (lihat bawah).
Pakai: python revisi_d/scripts/p2c_jamma.py [K] [e2e|coarse]
"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2c_common as C
import torch, torch.nn as nn, torch.nn.functional as F

METHOD = "jamma"
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072


def refmode(model):
    import mamba_ssm.modules.mamba_simple as ms
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    ms.selective_scan_fn = selective_scan_ref
    ms.causal_conv1d_fn = None
    ms.layer_norm_fn = None
    ms.rms_norm_fn = None

    class PlainRMSNorm(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.weight = src.weight
            self.bias = getattr(src, "bias", None)
            self.eps = getattr(src, "eps", 1e-5)

        def forward(self, x, z=None, residual=None, prenorm=False, **kw):
            if residual is not None:
                x = x + residual
            res = x
            xf = x.float()                       # norma di FP32 (spt kernel Triton yang menghitung dalam fp32)
            y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight
            if self.bias is not None:
                y = y + self.bias
            if z is not None:
                y = y * F.silu(z)
            return (y, res) if prenorm else y
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__ in ("RMSNorm", "LayerNorm") and child.__class__.__module__.startswith("mamba_ssm"):
                setattr(parent, name, PlainRMSNorm(child)); n += 1
    for mod in model.modules():
        if type(mod).__name__ == "Mamba":
            mod.use_fast_path = False
        if hasattr(mod, "fused_add_norm"):
            mod.fused_add_norm = False
    # residual_in_fp32=True -> keluaran blok Mamba FP32; agregator GLU_3 (conv) diberi cast ke dtype bobotnya
    for mod in model.modules():
        if type(mod).__name__ == "GLU_3":
            mod.register_forward_pre_hook(lambda md, a: (a[0].to(md.W.weight.dtype),))
    return n


class CoarseConf(nn.Module):
    def __init__(self, full):
        super().__init__(); self.backbone, self.m = full.backbone, full.matcher

    def forward(self, img0, img1):
        from src.jamma.utils.utils import normalize_keypoints
        data = {"imagec_0": img0, "imagec_1": img1}
        self.backbone(data)
        d0, d1 = data["feat_8_0"].flatten(2, 3), data["feat_8_1"].flatten(2, 3)
        k0 = normalize_keypoints(data["grid_8"], img0.shape[-2:]).transpose(1, 2)
        k1 = normalize_keypoints(data["grid_8"], img1.shape[-2:]).transpose(1, 2)
        data["feat_8_0"] = d0 + self.m.kenc(k0.to(d0.dtype))
        data["feat_8_1"] = d1 + self.m.kenc(k1.to(d1.dtype))
        self.m.joint_mamba(data)
        cm = self.m.coarse_matching
        f0 = cm.final_proj(data["feat_8_0"].transpose(1, 2)); f1 = cm.final_proj(data["feat_8_1"].transpose(1, 2))
        f0, f1 = f0 / f0.shape[-1] ** .5, f1 / f1.shape[-1] ** .5
        sim = torch.einsum("nlc,nsc->nls", f0, f1) / cm.temperature
        return F.softmax(sim, 2) * F.softmax(sim, 1)


def static_coarse_jamma(sim, thr, border, hw, K, scale):
    """Setara get_coarse_match_inference JamMa: mask = (sm_baris>thr & argmax baris) OR (sm_kolom>thr & argmax kolom),
    border dibuang, mconf = sim. Statis: kandidat baris (i, j_r(i)) dan kandidat kolom (i_c(j), j);
    kandidat kolom yang sudah sama dgn kandidat baris valid dibuang (duplikat OR). Top-K skor dari gabungan 2*L."""
    import numpy as np
    h, w = hw
    m = np.ones((h, w), bool)
    if border > 0:
        m[:border] = m[-border:] = False; m[:, :border] = m[:, -border:] = False
    bm = torch.from_numpy(m.flatten()).to(sim.device)
    pr = F.softmax(sim, 2); pc = F.softmax(sim, 1)
    vr, jr = pr.max(2)                       # (1,L)
    vc, ic = pc.max(1)                       # (1,S)
    valid_r = (vr > thr) & bm[None] & torch.gather(bm[None], 1, jr)
    valid_c = (vc > thr) & bm[None] & torch.gather(bm[None], 1, ic)
    dup = valid_c & torch.gather(valid_r, 1, ic) & (torch.gather(jr, 1, ic) == torch.arange(sim.shape[2], device=sim.device)[None])
    valid_c = valid_c & ~dup
    i_all = torch.cat([torch.arange(sim.shape[1], device=sim.device)[None], ic], 1)
    j_all = torch.cat([jr, torch.arange(sim.shape[2], device=sim.device)[None]], 1)
    v_all = torch.cat([valid_r, valid_c], 1)
    s_all = torch.cat([vr, vc], 1)
    score = torch.where(v_all, s_all + 1.0, torch.zeros_like(s_all))   # urut: valid dulu (skor softmax + 1)
    top, idx = torch.topk(score, K, dim=1)
    i_ids = torch.gather(i_all, 1, idx)[0]; j_ids = torch.gather(j_all, 1, idx)[0]
    valid = top[0] > 0
    mconf = sim[0, i_ids, j_ids]
    b_ids = torch.zeros_like(i_ids)
    mk0 = torch.stack([i_ids % w, torch.div(i_ids, w, rounding_mode="trunc")], 1).float() * scale
    mk1 = torch.stack([j_ids % w, torch.div(j_ids, w, rounding_mode="trunc")], 1).float() * scale
    return dict(b_ids=b_ids, i_ids=i_ids, j_ids=j_ids, mconf=mconf, m_bids=b_ids, mkpts0_c=mk0, mkpts1_c=mk1,
                valid_c=valid)


class E2E(nn.Module):
    def __init__(self, full, K):
        super().__init__(); self.full = full; mt = full.matcher
        import src.jamma.jamma as JJ
        C.shim_F(JJ)
        cm, fm = mt.coarse_matching, mt.fine_matching
        hw = (C.H // 8, C.W // 8)
        state = {}

        def gcm(sim, data):
            d = static_coarse_jamma(sim, cm.thr, cm.border_rm, hw, K, 8.0)
            state["valid_c"] = d.pop("valid_c")
            return d
        cm.get_coarse_match_inference = gcm

        def gfsm(conf_f, f0u, f1u, data):
            # = FineSubMatching.get_fine_sub_match (inferensi) TANPA torch.where: per kecocokan kasar m,
            # (i,j) = argmax global matriks fine; valid bila > fine_thr (dan kasar valid). Paling banyak 1 per m.
            Wf = fm.W_f
            M = conf_f.shape[0]
            v, idx = conf_f.reshape(M, -1).max(1)
            i_ids = torch.div(idx, Wf * Wf, rounding_mode="trunc"); j_ids = idx % (Wf * Wf)
            valid = (v > fm.fine_thr) & state["valid_c"]
            ic, jc = data["i_ids"], data["j_ids"]
            sfc = 4                               # hw0_f / hw0_c = 240/60
            mk0c = torch.stack([ic % hw[1], torch.div(ic, hw[1], rounding_mode="trunc")], 1) * sfc
            mk1c = torch.stack([jc % hw[1], torch.div(jc, hw[1], rounding_mode="trunc")], 1) * sfc
            scale = 2.0                           # hw0_i / hw0_f = 480/240
            w0 = torch.stack([i_ids % Wf, torch.div(i_ids, Wf, rounding_mode="trunc")], 1)
            w1 = torch.stack([j_ids % Wf, torch.div(j_ids, Wf, rounding_mode="trunc")], 1)
            ar = torch.arange(M, device=conf_f.device)
            sub = fm.subpixel_mlp(torch.cat([f0u[ar, i_ids], f1u[ar, j_ids]], dim=-1))
            s0, s1 = torch.chunk(sub, 2, dim=1)
            s0 = torch.tanh(s0) * 0.5; s1 = torch.tanh(s1) * 0.5
            pad = 0 if Wf % 2 == 0 else Wf // 2
            k0 = (w0 + mk0c - pad).float() * scale + s0.float() * scale
            k1 = (w1 + mk1c - pad).float() * scale + s1.float() * scale
            return {"mkpts0_f": k0, "mkpts1_f": k1, "mconf_f": torch.where(valid, v, torch.zeros_like(v))}
        fm.get_fine_sub_match = gfsm
        # kenc menerima keypoint grid FP32 (normalize_keypoints) -> cast ke dtype bobot (= CoarseConf: k.to(d.dtype))
        mt.kenc.register_forward_pre_hook(lambda md, a: (a[0].to(next(md.parameters()).dtype),))

    def forward(self, img0, img1):
        data = {"imagec_0": img0, "imagec_1": img1}
        self.full(data)
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf_f"]


def main():
    import stitch_demo as SD
    w = SD.JamMaModel()
    full = w.model.cuda().eval()
    g = C.load_pair_rgb_imagenet()
    d = {"imagec_0": g[0], "imagec_1": g[1]}
    with torch.no_grad():
        full(d)                                              # rujukan = JamMa ASLI (kernel fused)
    ref_full = (d["mkpts0_f"].float().cpu().numpy(), d["mkpts1_f"].float().cpu().numpy())
    with torch.no_grad():
        conf_fused = CoarseConf(full)(*g)
    print("M asli:", len(ref_full[0]))
    nsw = refmode(full)
    print("norm diganti:", nsw)
    g16 = [x.half() for x in g]
    common = "mode rujukan: selective_scan_ref (scan ter-unroll per langkah, dihitung FP32 oleh kode rujukan) + RMSNorm PyTorch (FP32) menggantikan kernel CUDA/Triton mamba-ssm"
    if "e2e" not in sys.argv:
        C.do_level(METHOD, "coarse", CoarseConf(copy.deepcopy(full)).half().eval(), g16, ["image0", "image1"], ["conf_matrix"],
                   ref32=conf_fused, changes=common + "; keluaran = softmax(sim,2)*softmax(sim,1)",
                   note="ref32 = JamMa fused-kernel FP32 (asli)")
        torch.cuda.empty_cache()
    if "coarse" in sys.argv:
        return
    sv = C.static_vs_original(E2E(copy.deepcopy(full), K).eval(), g, ref_full)
    print("static(ref-mode) fp32 vs asli(fused):", sv)
    lvl = "e2e" if K == 3072 else f"e2e_k{K}"
    C.do_level(METHOD, lvl, E2E(copy.deepcopy(full), K).half().eval(), g16, ["image0", "image1"],
               ["mkpts0_f", "mkpts1_f", "mconf"], ref_full=ref_full,
               changes=common + f"; coarse OR-matching -> kandidat baris+kolom top-K statis (K={K}, duplikat OR dibuang); "
                       "fine: argmax global per jendela (bukan torch.where) + valid=conf>fine_thr; stride unfold int",
               extra=dict(K=K, static_fp32_vs_original=sv))


if __name__ == "__main__":
    main()
