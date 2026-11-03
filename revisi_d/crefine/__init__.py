"""crefine -- pustaka komponen penghalus untuk prareg P39 (2026-10-22), tanpa efek samping impor.

Aturan: tidak ada os.chdir, cv2.setNumThreads, CUDA, atau impor torch pada saat impor modul. Proses pekerja memanggil
`configure_worker()` sekali. Fungsi yang disalin dari skrip lama ditandai "verbatim <asal>" dan diuji setara bit demi bit
(tests/test_legacy_equiv.py, tests/test_p35_repro.py)."""
import os
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_ROOT = _os.environ.get("AERIAL_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
AERIAL_DATA = _os.environ.get("AERIAL_DATA", "")  # see DATA.md

ROOT = AERIAL_ROOT + ""
REVD = f"{ROOT}/revisi_d"
USEGEO = AERIAL_DATA + "/data_tambahan/usegeo"
MD = AERIAL_DATA + "/megadepth1500/megadepth1500"
LOWTEX = {"seneca": AERIAL_DATA + "/odm_lowtex/odm_data_seneca/images",
          "bellus": AERIAL_DATA + "/odm_lowtex/odm_data_bellus/images"}


def configure_worker():
    """Dipanggil sekali per proses pekerja CPU: OpenCV satu utas, OpenCL mati, BLAS satu utas (cek)."""
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(k) != "1":
            raise RuntimeError(f"{k} harus 1 sebelum Python mulai (diekspor peluncur), bukan {os.environ.get(k)!r}")
