"""Revisi A IJIES (T2) -- susun set latih PUBLIK untuk replikasi (prareg amandemen 1).

Sumber potongan sintetis (pengganti ortho privat): (a) ortho UAV OpenAerialMap yang lisensinya tercatat di manifest
(CC-BY 4.0 / CC BY-NC 4.0, platform uav; yang berlabel satellite dilewati), dan (b) foto nadir mentah penerbangan publik
yang juga menjadi sumber pasangan pseudo-GT (ODM Monterey Road Bend, CC-BY-SA 4.0; DroneMapper Back 9, lisensi tidak
tercatat -> ditandai). Dua ortho OAM disisihkan sebagai validasi (val loss saja; checkpoint = epoch 12).
Semua sumber > 100 km dari situs uji (UseGeo, sheffield_park_1, Seneca, Bellus).
Keluaran: $AERIAL_DATA/data_publik_arev/{orthos_train, orthos_val}/ + public_sources.json
Pakai: python revisi_d/scripts/arev_public_assemble.py
"""
import glob
import json
import os
import time
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))
import cv2

BASE = AERIAL_DATA + "/data_publik_arev"
OAM = AERIAL_DATA + "/data_tambahan/openaerialmap"
FLIGHTS = {"odm_data_monterey_road_bend": f"{BASE}/flights/odm_data_monterey_road_bend/images",
           "DroneMapper_Golf9": AERIAL_DATA + "/Data Uji Coba/DroneMapper_Golf9_May2016"}
LICENCE = {"odm_data_monterey_road_bend": "CC-BY-SA 4.0 (OpenDroneMap)",
           "DroneMapper_Golf9": "DroneMapper sample data (licence not stated)"}
VAL = {"Talon_Ridge_Middle_School.tif", "Salut.tif"}


def main():
    tr, va = f"{BASE}/orthos_train", f"{BASE}/orthos_val"
    os.makedirs(tr, exist_ok=True)
    os.makedirs(va, exist_ok=True)
    man = json.load(open(f"{OAM}/manifest.json"))
    src = []
    for r in man:
        if r.get("platform") != "uav" or "CC" not in (r.get("license") or ""):
            continue
        im = cv2.imread(f"{OAM}/{r['file']}", cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3:
            continue
        if im.shape[2] == 4:
            im[im[..., 3] == 0, :3] = 0
            im = im[..., :3]
        name = os.path.splitext(r["file"])[0] + ".jpg"
        dst = f"{va if r['file'] in VAL else tr}/oam_{name}"
        if not os.path.exists(dst):
            cv2.imwrite(dst + ".tmp.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.replace(dst + ".tmp.jpg", dst)
        src.append(dict(kind="ortho", file=os.path.basename(dst), split="val" if r["file"] in VAL else "train",
                        title=r.get("title"), provider=r.get("provider"), license=r.get("license"), gsd=r.get("gsd"),
                        shape=list(im.shape)))
    for fl, d in FLIGHTS.items():
        for p in sorted(glob.glob(f"{d}/*")):
            if not p.lower().endswith((".jpg", ".jpeg")):
                continue
            dst = f"{tr}/{fl}__{os.path.basename(p)}"
            if not os.path.lexists(dst):
                os.symlink(p, dst)
            src.append(dict(kind="raw_photo", file=os.path.basename(dst), split="train", flight=fl, license=LICENCE[fl]))
    json.dump(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), sources=src), open(f"{BASE}/public_sources.json", "w"), indent=1)
    n = {k: sum(1 for s in src if s["kind"] == k and s["split"] == "train") for k in ("ortho", "raw_photo")}
    print(json.dumps(dict(train=n, val=sum(1 for s in src if s["split"] == "val"))), flush=True)
    open(f"{BASE}/assemble.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
