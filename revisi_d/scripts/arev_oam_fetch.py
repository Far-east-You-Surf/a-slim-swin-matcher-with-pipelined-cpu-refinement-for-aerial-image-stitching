"""Revisi A IJIES (T2) -- ortho UAV publik dari OpenAerialMap untuk replikasi data publik.

Saringan: platform uav, lisensi CC-BY 4.0 (lisensi bawaan OAM; rekaman berlisensi lain dilewati kecuali CC BY-NC),
GSD 1-10 cm, ukuran berkas 15-250 MB, pusat bbox > 100 km dari situs uji (sheffield_park_1, Seneca, Bellus, sand_key)
dan dari kandidat lokasi UseGeo (lintang 39,56° pada meridian tengah zona UTM 29-35). Unduhan .part lalu rename;
manifest JSON per berkas (id, judul, penyedia, lisensi, gsd, bbox, url, sha256) -> dapat diulang dan disitasi.
Konversi: GeoTIFF -> JPEG kualitas 95 (alfa = 0 dijadikan hitam), dua berkas terakhir ke folder validasi.
Pakai: python revisi_d/scripts/arev_oam_fetch.py [jumlah=40]
"""
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))
import cv2
import numpy as np

BASE = AERIAL_DATA + "/data_publik_arev"
RAW, TR, VA = f"{BASE}/oam_raw", f"{BASE}/orthos_train", f"{BASE}/orthos_val"
API = "https://api.openaerialmap.org/meta?limit=100&page=%d&platform=uav&order_by=acquisition_end&sort=desc"
SITES = [(28.0391, -82.6975), (27.9584, -82.8322), (41.0347, -83.3057), (41.2257, -81.7025)] + \
        [(39.56, cm) for cm in (-9, -3, 3, 9, 15, 21, 27)]


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def get(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-fetch/1.0"}), timeout=timeout)


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for d in (RAW, TR, VA):
        os.makedirs(d, exist_ok=True)
    man_p = f"{BASE}/oam_manifest.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else []
    have = {m["_id"] for m in man}
    recs, page = [], 1
    while page <= 12:
        try:
            d = json.loads(get(API % page).read())
        except Exception as e:  # noqa: BLE001
            print("API gagal", page, e, flush=True)
            break
        rs = d.get("results", [])
        if not rs:
            break
        recs += rs
        page += 1
    print(f"{len(recs)} rekaman UAV", flush=True)
    for r in recs:
        if len(man) >= want:
            break
        lic = (r.get("properties") or {}).get("license") or "CC-BY 4.0"
        mb = (r.get("file_size") or 0) / 1e6
        gsd = r.get("gsd") or 0
        bb = r.get("bbox") or [0, 0, 0, 0]
        c = ((bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2)
        if r["_id"] in have or not ("CC-BY" in lic.replace(" ", "-") or "CC BY" in lic):
            continue
        if not (0.01 <= gsd <= 0.10 and 15 <= mb <= 250) or min(km(c, s) for s in SITES) < 100:
            continue
        url = r.get("uuid") or ""
        if not url.endswith(".tif"):
            continue
        name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (r.get("title") or r["_id"]))[:40] + "_" + r["_id"][-6:]
        raw = f"{RAW}/{name}.tif"
        try:
            if not os.path.exists(raw):
                with get(url, 900) as resp, open(raw + ".part", "wb") as f:
                    while True:
                        b = resp.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                os.replace(raw + ".part", raw)
            im = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or min(im.shape[:2]) < 1100:
                print("lewat (tak terbaca/kecil)", name, flush=True)
                continue
            if im.shape[2] == 4:
                im[im[..., 3] == 0, :3] = 0
                im = im[..., :3]
            if im.dtype != np.uint8:
                continue
            h = hashlib.sha256(open(raw, "rb").read()).hexdigest()
            man.append(dict(_id=r["_id"], name=name, title=r.get("title"), provider=r.get("provider"), license=lic,
                            gsd=gsd, bbox=bb, url=url, mb=round(mb, 1), shape=list(im.shape), sha256=h,
                            t=time.strftime("%Y-%m-%d %H:%M:%S")))
            dst = f"{TR}/{name}.jpg"
            cv2.imwrite(dst + ".tmp.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.replace(dst + ".tmp.jpg", dst)
            json.dump(man, open(man_p + ".tmp", "w"), indent=1)
            os.replace(man_p + ".tmp", man_p)
            print(f"PROGRESS {len(man)}/{want} {name} {im.shape} gsd {gsd:.3f} {mb:.0f} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            print("gagal", name, e, flush=True)
            for p in (raw + ".part",):
                if os.path.exists(p):
                    os.remove(p)
    # dua ortho terakhir menjadi validasi (val loss saja; seleksi checkpoint run publik = epoch 12)
    tr = sorted(os.listdir(TR))
    if len(tr) > 4 and not os.listdir(VA):
        for f in tr[-2:]:
            os.replace(f"{TR}/{f}", f"{VA}/{f}")
    open(f"{BASE}/oam.alldone", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
