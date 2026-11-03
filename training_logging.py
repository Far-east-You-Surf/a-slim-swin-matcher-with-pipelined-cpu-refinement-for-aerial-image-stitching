"""
Helper kecil supaya semua print() di skrip training otomatis terekam ke file
log (selain tetap tampil di terminal), tanpa perlu user mengingat utk redirect
manual (`python train_x.py | tee log.txt`). Dipakai monitor_training.py utk
memonitor training dari terminal lain.
"""
import sys
import time


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def setup_tee_logging(log_path):
    """Duplikasi stdout & stderr ke `log_path` (append), ditandai marker
    RUN START supaya monitor_training.py bisa membedakan run terbaru dari
    sisa log run sebelumnya di file yang sama."""
    f = open(log_path, "a", buffering=1)
    marker = f"\n===RUN START {time.strftime('%Y-%m-%d %H:%M:%S')}===\n"
    f.write(marker)
    f.flush()
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f
