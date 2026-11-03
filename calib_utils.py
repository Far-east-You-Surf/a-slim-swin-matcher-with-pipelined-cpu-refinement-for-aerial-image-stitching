"""
Utilitas kalibrasi PTQ memakai gambar aerial ASLI (bukan noise Gaussian acak),
supaya skala kuantisasi aktivasi merepresentasikan distribusi data sesungguhnya.

Gambar 001.JPG/002.JPG sengaja DIKECUALIKAN dari kalibrasi karena dipakai sebagai
data uji akurasi (held-out) di evaluate_all_precisions.py -- menghindari kebocoran
data kalibrasi ke data evaluasi.
"""
import glob
import os
import cv2
import torch
import torchvision.transforms as transforms
import os as _os  # added for the public repository: paths come from the environment (README.md)
AERIAL_DATA_UJICOBA = _os.environ.get("AERIAL_DATA_UJICOBA", "")  # see DATA.md

DATASET_DIR = AERIAL_DATA_UJICOBA + "/Dataset Aerial234"
HELD_OUT = {"001.JPG", "002.JPG"}


def load_real_calibration_pairs(n_pairs=16, size=(256, 256), device="cuda"):
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.JPG")))
    files = [f for f in files if os.path.basename(f) not in HELD_OUT]
    assert len(files) >= n_pairs + 1, "Dataset kalibrasi tidak cukup banyak gambar."

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
    ])

    pairs = []
    for i in range(n_pairs):
        img1_cv = cv2.imread(files[i])
        img2_cv = cv2.imread(files[i + 1])
        img1 = transform(cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device).half()
        img2 = transform(cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device).half()
        pairs.append((img1, img2))
    return pairs


def make_real_calibrate_loop(n_pairs=16, size=(256, 256)):
    """Kembalikan fungsi calibrate_loop(model) yang memakai gambar aerial asli,
    dengan signature drop-in-replacement untuk forward_loop di mtq.quantize()."""
    def calibrate_loop(model):
        device = next(model.parameters()).device
        pairs = load_real_calibration_pairs(n_pairs=n_pairs, size=size, device=device)
        with torch.no_grad():
            for img1, img2 in pairs:
                model(img1, img2)
    return calibrate_loop
