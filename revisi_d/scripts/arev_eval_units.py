"""Revisi A IJIES -- pelari evaluasi registrasi/akurasi per unit (pencocok × urutan × d) yang tahan putus daya.

Membungkus `evaltool.evaluate` (LOCKED, tidak disunting). Tiap unit dievaluasi ke folder sementara lalu dipindah
atomik; penanda `<unit>.done` memuat sha256 keluaran + ringkasan. Unit yang penandanya sah dilewati, unit yang
terpotong diulang dari awal (satu unit UseGeo ±1–4 menit). Batas jumlah kecocokan dimatikan dengan STITCH_CAP
(G0: pembanding sebagaimana terbit, tanpa batas 3200), kecuali --cap diberikan.

Pakai (dari mana saja; venv aerial_fp8_env):
  python revisi_d/scripts/arev_eval_units.py --matcher matchformer --plan usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3 \
         --out revisi_d/results/arev/g0 [--cap 3200] [--no-pose]
Baris kemajuan: "PROGRESS <unit selesai>/<total> <unit>".
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
LOWTEX = {"seneca": AERIAL_DATA + "/odm_lowtex/odm_data_seneca/images",
          "bellus": AERIAL_DATA + "/odm_lowtex/odm_data_bellus/images"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_plan(s):
    out = []
    for part in s.split(","):
        seq, ds = part.split(":")
        if "-" in ds:
            lo, hi = ds.split("-")
            dl = list(range(int(lo), int(hi) + 1))
        else:
            dl = [int(x) for x in ds.split("+")]
        out += [(seq, d) for d in dl]
    return out


def unit_ok(done_path, out_dir, tag):
    if not os.path.exists(done_path):
        return False
    try:
        info = json.load(open(done_path))
        return all(sha256(f"{out_dir}/{tag}{ext}") == info["sha256"][ext] for ext in (".jsonl", ".npz"))
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True, help="nama evaltool atau spes p16ckpt=/bbckpt=")
    ap.add_argument("--plan", required=True, help="mis. usegeo:1-8,sheffield:1-3,seneca:1-3,bellus:1-3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=None, help="batas kecocokan (bawaan: tanpa batas)")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--label", default=None, help="label keluaran pengganti m.label")
    a = ap.parse_args()
    os.environ["STITCH_CAP"] = str(a.cap if a.cap else 10 ** 9)
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, f"{ROOT}/revisi_d/scripts")
    import torch
    import stitch_demo as SD
    import evaltool as ET
    if a.matcher.startswith("arev="):
        import arev_models
        arev_models.install_into_evaltool()
    if a.matcher == "edm":                         # pencocok efisien tambahan (G3), didaftarkan tanpa menyunting tp_matchers
        import arev_tp_extra
        arev_tp_extra.install()
    if os.environ.get("AREV_TP_WEIGHTS"):          # "<nama>=<jalur>": bobot pembanding hasil latih ulang (T3)
        import tp_matchers as TM
        n, p = os.environ["AREV_TP_WEIGHTS"].split("=", 1)
        TM.WEIGHTS[n] = p
        meta_w = dict(tp_weights=p)
    else:
        meta_w = {}
    SD.FLIGHTS.update(LOWTEX)
    out_dir = os.path.abspath(a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out))
    os.makedirs(out_dir, exist_ok=True)
    units = parse_plan(a.plan)
    m = None
    meta = dict(cap=SD.CAP, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                cvd=os.environ.get("CUDA_VISIBLE_DEVICES"), host_t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **meta_w)
    for k, (seq, d) in enumerate(units):
        label = a.label or a.matcher
        tag = f"{seq.replace(' ', '_')}__{label}__d{d}"
        done = f"{out_dir}/{tag}.done"
        if unit_ok(done, out_dir, tag):
            print(f"PROGRESS {k + 1}/{len(units)} {tag} (sudah)", flush=True)
            continue
        if m is None:
            m = ET.build_matchers(a.matcher)[0]
            if a.label:
                m.label = a.label
            meta["matcher_label"] = m.label
        tmp = f"{out_dir}/_tmp_{tag}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        want_pose = (seq == "usegeo") and not a.no_pose
        t0 = time.time()
        s = ET.evaluate(m, seq, d, "registrasi", tmp, want_pose=want_pose)
        if s.get("n_error"):
            # jangan pernah menandai selesai unit yang berisi galat (mis. cuDNN tercampur); pekerjaan akan diulang
            raise RuntimeError(f"{tag}: {s['n_error']}/{s['n']} pasangan galat -- unit tidak ditandai selesai")
        src_tag = f"{seq.replace(' ', '_')}__{m.label}__d{d}"
        for ext in (".jsonl", ".npz"):
            with open(f"{tmp}/{src_tag}{ext}", "rb") as f:
                os.fsync(f.fileno())
            os.replace(f"{tmp}/{src_tag}{ext}", f"{out_dir}/{tag}{ext}")
        shutil.rmtree(tmp, ignore_errors=True)
        s["jsonl"] = f"{out_dir}/{tag}.jsonl"
        info = dict(summary=s, meta=meta, seconds=round(time.time() - t0, 1),
                    sha256={ext: sha256(f"{out_dir}/{tag}{ext}") for ext in (".jsonl", ".npz")})
        write_atomic(done, json.dumps(info, indent=1))
        print(json.dumps(s), flush=True)
        print(f"PROGRESS {k + 1}/{len(units)} {tag}", flush=True)
    job_done = f"{out_dir}/{(a.label or a.matcher).replace('=', '_').replace('/', '_')}__{a.plan.replace(':', '').replace(',', '_')}.alldone"
    write_atomic(job_done, json.dumps(dict(units=len(units), t=time.strftime("%Y-%m-%d %H:%M:%S"))))


if __name__ == "__main__":
    main()
