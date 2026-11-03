"""Revisi A IJIES -- X8: pemisahan situs latih / seleksi / uji (butir R2.4).

Lokasi dari GPS EXIF bingkai pertama tiap penerbangan (foto mentah), pose UseGeo (koordinat proyeksi; zona UTM tidak
tercatat di data, jadi hanya dicantumkan sebagai lokal), dan bbox OpenAerialMap. Ortho privat tanpa georeferensi
dicakup oleh uji pencarian-citra yang sudah ada (SIFT+RANSAC tiap set uji terhadap ke-38 ortho: inlier maksimum = lantai
derau; results/kontaminasi/ dan log §67/§95.4). Jarak terdekat antarsitus dihitung haversine.
Keluaran: revisi_d/results/arev/x8/sites.json
"""
import glob
import json
import math
import os

from PIL import Image
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

OUT = AERIAL_ROOT + "/revisi_d/results/arev/x8"
SETS = {
    "train_pseudo": [AERIAL_DATA + "/Data Disertasi/Data Latih Pre-Sticth/*"],
    "select_q15": [AERIAL_DATA_UJICOBA + "/G Uji"],
    "test": [AERIAL_DATA + "/Data Uji Bersih/drone_dataset_sheffield_park_1-master",
             AERIAL_DATA + "/odm_lowtex/odm_data_seneca/images",
             AERIAL_DATA + "/odm_lowtex/odm_data_bellus/images"],
    "public_train": [AERIAL_DATA + "/data_publik_arev/flights/odm_data_monterey_road_bend/images",
                     AERIAL_DATA + "/Data Uji Coba/DroneMapper_Golf9_May2016"],
}


def gps(p):
    try:
        g = (Image.open(p)._getexif() or {}).get(34853)
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600  # noqa: E731
        return (round(dms(g[2]) * (1 if g[1] == "N" else -1), 4), round(dms(g[4]) * (1 if g[3] == "E" else -1), 4))
    except Exception:  # noqa: BLE001
        return None


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def main():
    os.makedirs(OUT, exist_ok=True)
    sites = []
    for role, pats in SETS.items():
        for pat in pats:
            for d in sorted(glob.glob(pat)):
                if not os.path.isdir(d):
                    continue
                imgs = sorted(p for p in glob.glob(f"{d}/*") if p.lower().endswith((".jpg", ".jpeg")))
                loc = next((g for g in (gps(p) for p in imgs[:5]) if g), None)
                sites.append(dict(role=role, name=os.path.basename(d.rstrip("/").replace("/images", "")) if not d.endswith("images")
                                  else os.path.basename(os.path.dirname(d)), n_images=len(imgs), gps=loc))
    for s in sites:
        others = [t for t in sites if t["role"] == "test" and t is not s and t["gps"] and s["gps"]]
        s["nearest_test_km"] = round(min(km(s["gps"], t["gps"]) for t in others)) if others and s["role"] != "test" else None
    tr = [s for s in sites if s["role"] in ("train_pseudo", "select_q15", "public_train") and s["gps"]]
    te = [s for s in sites if s["role"] == "test" and s["gps"]]
    res = dict(sites=sites, min_train_to_test_km=round(min(km(a["gps"], b["gps"]) for a in tr for b in te)) if tr and te else None,
               usegeo="pose UseGeo dalam koordinat proyeksi lokal (X0 ±4,9e5, Y0 ±4,38e6); situs berbeda dari semua situs latih "
                      "menurut uji pencarian-citra (results/kontaminasi)",
               private_orthos="tanpa georeferensi; diuji dengan SIFT+RANSAC terhadap set uji (inlier maks = lantai derau)")
    json.dump(res, open(f"{OUT}/sites.json", "w"), indent=1)
    for s in sites:
        print(s)
    print("jarak minimum latih->uji:", res["min_train_to_test_km"], "km")


if __name__ == "__main__":
    main()
