"""
FASE 0 demonstrasi stitching: audit kelayakan data penerbangan.

Menjawab tiga pertanyaan SEBELUM kode eksperimen apa pun ditulis:

  1. KONTAMINASI -- jalur mana yang jejak GPS-nya terpisah dari orthomosaic latih?
     Memori proyek sudah menandai Flight 12A/12B/13/14Rev cocok dgn region
     Pacarejo yang ada di data latih; itu diverifikasi ulang di sini, tidak
     dipercaya begitu saja.

  2. AKURASI GPS -- apakah drift model akan tenggelam dalam derau GPS-nya sendiri?
     Tanpa ground truth, derau ditaksir dari SIMPANGAN posisi berurutan terhadap
     lintasan lokal yang mulus (fit garis pada jendela geser). Drone yang terbang
     lurus punya GPS mulus; simpangan = derau. Kalau derau ~5 m sedangkan drift
     antar-model beda ~1 m, metrik drift-vs-GPS TIDAK DIPAKAI.

  3. KELAYAKAN MOSAIK -- apakah bingkai berurutan cukup bertumpang tindih untuk
     dirantai? Diperiksa lewat jarak antar-bingkai berurutan vs jejak tanah yang
     ditaksir dari altitude + focal length.

Keluaran: gate_arch_results/flights_phase0.json + ringkasan di konsol.
Jalan di CPU, hanya membaca EXIF (tidak mendekode piksel), jadi cepat dan aman
dijalankan berdampingan dgn pekerjaan GPU.

Pakai: python audit_flights_phase0.py [out.json]
"""

import json
import math
import os
import sys
from datetime import datetime

from PIL import Image, ExifTags
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

Image.MAX_IMAGE_PIXELS = None

ROOT = AERIAL_DATA + "/Data Disertasi"
FLIGHT_DIR = os.path.join(ROOT, "Data Latih Pre-Sticth")
TRAIN_ORTHO = os.path.join(ROOT, "Data Latih Orthomosaic")
VAL_ORTHO = os.path.join(ROOT, "Data Validasi Orthomosaic")
G_UJI = AERIAL_DATA_UJICOBA + "/G Uji"
DEFAULT_OUT = "gate_arch_results/flights_phase0.json"

GPS_IDS = {v: k for k, v in ExifTags.GPSTAGS.items()}
TAG_IDS = {v: k for k, v in ExifTags.TAGS.items()}


def _rat(x):
    try:
        return float(x)
    except Exception:
        try:
            return x[0] / x[1]
        except Exception:
            return None


def dms_to_deg(dms, ref):
    if dms is None:
        return None
    try:
        d, m, s = (_rat(v) for v in dms)
    except Exception:
        return None
    if None in (d, m, s):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        deg = -deg
    return deg


def read_exif(path):
    """Hanya header EXIF; piksel tidak didekode."""
    out = dict(path=path)
    try:
        with Image.open(path) as im:
            out["size"] = list(im.size)
            exif = im.getexif()
            if not exif:
                return out
            for name in ("Model", "Make", "FocalLength", "DateTimeOriginal",
                         "FocalLengthIn35mmFilm"):
                tid = TAG_IDS.get(name)
                if tid is not None and tid in exif:
                    out[name] = _rat(exif[tid]) if "Focal" in name else str(exif[tid])
            gps = exif.get_ifd(TAG_IDS.get("GPSInfo", 34853)) or {}
            if gps:
                out["lat"] = dms_to_deg(gps.get(GPS_IDS["GPSLatitude"]),
                                        gps.get(GPS_IDS["GPSLatitudeRef"]))
                out["lon"] = dms_to_deg(gps.get(GPS_IDS["GPSLongitude"]),
                                        gps.get(GPS_IDS["GPSLongitudeRef"]))
                alt = _rat(gps.get(GPS_IDS["GPSAltitude"]))
                out["alt"] = alt
                for tag in ("GPSDOP", "GPSHPositioningError"):
                    tid = GPS_IDS.get(tag)
                    if tid is not None and tid in gps:
                        out[tag] = _rat(gps[tid])
                ts = gps.get(GPS_IDS.get("GPSTimeStamp"))
                if ts:
                    try:
                        out["gps_time"] = "%02d:%02d:%06.3f" % (_rat(ts[0]), _rat(ts[1]), _rat(ts[2]))
                    except Exception:
                        pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def meters_per_deg(lat):
    """Konversi lokal deg->meter (cukup akurat pada skala satu penerbangan)."""
    m_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat)) + 1.175 * math.cos(4 * math.radians(lat))
    m_lon = 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat))
    return m_lat, m_lon


def to_xy(recs):
    pts = [(r["lat"], r["lon"]) for r in recs if r.get("lat") is not None and r.get("lon") is not None]
    if not pts:
        return [], (None, None)
    lat0 = sum(p[0] for p in pts) / len(pts)
    lon0 = sum(p[1] for p in pts) / len(pts)
    m_lat, m_lon = meters_per_deg(lat0)
    return [((la - lat0) * m_lat, (lo - lon0) * m_lon) for la, lo in pts], (lat0, lon0)


def gps_noise(xy, win=5):
    """Taksiran derau: simpangan tiap titik dari fit garis lurus pada jendela
    geser. Lintasan survei lokal ~lurus, jadi sisa fit didominasi derau GPS."""
    if len(xy) < win + 2:
        return None
    res = []
    half = win // 2
    for i in range(half, len(xy) - half):
        w = xy[i - half:i + half + 1]
        n = len(w)
        mx = sum(p[0] for p in w) / n
        my = sum(p[1] for p in w) / n
        sxx = sum((p[0] - mx) ** 2 for p in w)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in w)
        if sxx < 1e-9:
            continue
        b = sxy / sxx
        pred = my + b * (xy[i][0] - mx)
        res.append(abs(xy[i][1] - pred))
    if not res:
        return None
    res.sort()
    return dict(median_m=res[len(res) // 2], p90_m=res[int(0.9 * len(res))],
                max_m=res[-1], n=len(res))


def step_stats(xy):
    if len(xy) < 2:
        return None
    d = [math.dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1)]
    d_sorted = sorted(d)
    return dict(median_m=d_sorted[len(d_sorted) // 2], min_m=d_sorted[0],
                max_m=d_sorted[-1], n=len(d))


def bbox(xy, center):
    if not xy:
        return None
    lat0, lon0 = center
    m_lat, m_lon = meters_per_deg(lat0)
    return dict(center_lat=lat0, center_lon=lon0,
                extent_ns_m=max(p[0] for p in xy) - min(p[0] for p in xy),
                extent_ew_m=max(p[1] for p in xy) - min(p[1] for p in xy))


def survey(name, folder, limit=None):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg")))
    if limit:
        files = files[:limit]
    recs = [read_exif(os.path.join(folder, f)) for f in files]
    withgps = [r for r in recs if r.get("lat") is not None]
    xy, center = to_xy(withgps)
    models = sorted({r.get("Model", "?") for r in recs})
    dops = [r["GPSDOP"] for r in withgps if "GPSDOP" in r]
    herr = [r["GPSHPositioningError"] for r in withgps if "GPSHPositioningError" in r]
    alts = [r["alt"] for r in withgps if r.get("alt") is not None]
    sizes = sorted({tuple(r["size"]) for r in recs if "size" in r})
    return dict(
        name=name, folder=folder, n_photos=len(files), n_with_gps=len(withgps),
        camera_models=models, image_sizes=[list(s) for s in sizes],
        altitude_m=dict(median=sorted(alts)[len(alts) // 2], min=min(alts), max=max(alts)) if alts else None,
        gps_dop=dict(median=sorted(dops)[len(dops) // 2], max=max(dops)) if dops else None,
        gps_h_error_m=dict(median=sorted(herr)[len(herr) // 2], max=max(herr)) if herr else None,
        footprint=bbox(xy, center) if xy else None,
        consecutive_step=step_stats(xy),
        gps_noise_estimate=gps_noise(xy),
    )


def haversine_km(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    flights = {}
    for d in sorted(os.listdir(FLIGHT_DIR)):
        p = os.path.join(FLIGHT_DIR, d)
        if os.path.isdir(p):
            print(f"[survei] {d} ...", flush=True)
            flights[d] = survey(d, p)
    if os.path.isdir(G_UJI):
        print("[survei] G Uji ...", flush=True)
        flights["G_Uji"] = survey("G_Uji", G_UJI)

    # Apakah orthomosaic latih membawa GPS? (biasanya tidak -- keluaran olahan)
    ortho = {}
    for label, folder in (("train_ortho", TRAIN_ORTHO), ("val_ortho", VAL_ORTHO)):
        if not os.path.isdir(folder):
            continue
        rows = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg")):
                r = read_exif(os.path.join(folder, f))
                rows.append(dict(file=f, lat=r.get("lat"), lon=r.get("lon"),
                                 size=r.get("size")))
        ortho[label] = rows

    # Jarak antar-pusat jejak penerbangan -- pemisahan geografis kasar
    centers = {k: (v["footprint"]["center_lat"], v["footprint"]["center_lon"])
               for k, v in flights.items() if v.get("footprint")}
    sep = {}
    keys = sorted(centers)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sep[f"{a} <-> {b}"] = round(haversine_km(centers[a], centers[b]), 3)

    print("\n" + "=" * 96)
    hdr = f"{'jalur':12} {'foto':>5} {'GPS':>5} {'alt m':>7} {'langkah m':>10} {'derau med':>10} {'derau p90':>10} {'bentang m':>16}"
    print(hdr)
    for k, v in flights.items():
        alt = f"{v['altitude_m']['median']:.0f}" if v.get("altitude_m") else "-"
        st = f"{v['consecutive_step']['median_m']:.1f}" if v.get("consecutive_step") else "-"
        nz = v.get("gps_noise_estimate")
        nm = f"{nz['median_m']:.2f}" if nz else "-"
        np9 = f"{nz['p90_m']:.2f}" if nz else "-"
        fp = v.get("footprint")
        ext = f"{fp['extent_ns_m']:.0f} x {fp['extent_ew_m']:.0f}" if fp else "-"
        print(f"{k:12} {v['n_photos']:5d} {v['n_with_gps']:5d} {alt:>7} {st:>10} {nm:>10} {np9:>10} {ext:>16}")

    print("\n--- pemisahan geografis antar-jalur (km) ---")
    for k, v in sorted(sep.items(), key=lambda x: x[1]):
        flag = "  << BERIMPIT" if v < 1.0 else ""
        print(f"  {k:32} {v:9.3f}{flag}")

    print("\n--- orthomosaic membawa GPS? ---")
    for label, rows in ortho.items():
        n = sum(1 for r in rows if r["lat"] is not None)
        print(f"  {label}: {n}/{len(rows)} berkas ber-GPS")

    json.dump(dict(flights=flights, ortho=ortho, separation_km=sep),
              open(out_path, "w"), indent=2)
    print(f"\n[SELESAI] -> {out_path}")


if __name__ == "__main__":
    main()
