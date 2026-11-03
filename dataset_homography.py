import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # Bypass OpenCV pixel limit for giant orthomosaics
import glob
import cv2
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFile

# Mencegah error jika resolusi JPEG terlalu raksasa
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class HomographyOrthoDataset(Dataset):
    def __init__(self, data_path, epoch_size=5000, extra_files=None, exclude_names=None, max_cache_images=4):
        """
        data_path: Folder yang berisi gambar Orthomosaic JPEG
        epoch_size: Jumlah sampel virtual per epoch (karena kita memotong acak dari orthomosaic)
        extra_files: daftar path file tambahan dari folder lain, ikut dimasukkan.
                     Dipakai utk memindahkan tile scr LOGIS antar split (train/val)
                     tanpa memindah file fisik -- lihat config.yaml `dataset.val_exclude`.
        exclude_names: daftar basename yang dikecualikan dari data_path (pasangan dari extra_files di split lain).
        max_cache_images: batas gambar yang disimpan di RAM sekaligus (LRU, per worker
                     proses DataLoader). Dulu SEMUA orthomosaic dimuat eager saat init --
                     mubazir dgn 8 file lama, tapi OOM nyata dgn 36 file/21GP baru
                     (RAM 91GB penuh, proses dibunuh kernel OOM-killer -- diverifikasi
                     2026-07-10). Lazy-load + cache kecil ini menjaga perilaku statistik
                     crop persis sama, cuma membatasi berapa gambar utuh yg nangkring di RAM.
        """
        self.data_path = data_path
        self.epoch_size = epoch_size
        self.image_files = glob.glob(os.path.join(data_path, "*.JPG")) + glob.glob(os.path.join(data_path, "*.jpeg")) + glob.glob(os.path.join(data_path, "*.jpg"))

        if exclude_names:
            exclude_set = set(exclude_names)
            self.image_files = [f for f in self.image_files if os.path.basename(f) not in exclude_set]
        if extra_files:
            self.image_files += list(extra_files)

        if len(self.image_files) == 0:
            raise RuntimeError(f"Tidak ada file JPEG/JPG ditemukan di {data_path}")

        print(f"{len(self.image_files)} Orthomosaic terdaftar (lazy-load, cache LRU maks "
              f"{max_cache_images} gambar/worker -- bukan eager-load semua spt dulu, lihat catatan RAM).")

        self.max_cache_images = max_cache_images
        self._cache = OrderedDict()  # path -> np.ndarray RGB, LRU

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    def __len__(self):
        return self.epoch_size

    def _load_ortho(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Gagal memuat {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._cache[path] = img
        self._cache.move_to_end(path)
        if len(self._cache) > self.max_cache_images:
            self._cache.popitem(last=False)  # buang yg paling lama tidak dipakai
        return img

    def get_random_homography(self):
        # Membuat variasi rotasi, scale, dan perspective
        # Titik sudut asal
        pts1 = np.float32([[0, 0], [1024, 0], [0, 768], [1024, 768]])
        
        # Variasi acak untuk titik tujuan (Perspective Skew & Rotation)
        margin = 150 # Maksimum geser piksel untuk perspective
        pts2 = np.float32([
            [0 + np.random.randint(-margin, margin), 0 + np.random.randint(-margin, margin)],
            [1024 + np.random.randint(-margin, margin), 0 + np.random.randint(-margin, margin)],
            [0 + np.random.randint(-margin, margin), 768 + np.random.randint(-margin, margin)],
            [1024 + np.random.randint(-margin, margin), 768 + np.random.randint(-margin, margin)]
        ])
        
        H = cv2.getPerspectiveTransform(pts1, pts2)
        return H

    def __getitem__(self, idx):
        # 1. Pilih Orthomosaic secara acak
        path = self.image_files[np.random.randint(0, len(self.image_files))]
        ortho = self._load_ortho(path)
        H_img, W_img, _ = ortho.shape
        
        # 2. Potong area yang lebih besar (misal 1500x1200) untuk menghindari batas hitam saat di-warp
        crop_w, crop_h = 1024, 768
        
        # Cari koordinat top-left acak
        x = np.random.randint(0, W_img - crop_w)
        y = np.random.randint(0, H_img - crop_h)
        
        img1_cv = ortho[y:y+crop_h, x:x+crop_w].copy()
        
        # 3. Buat matriks Homografi Acak
        H = self.get_random_homography()
        
        # 4. Terapkan Warp ke img1 untuk membuat img2 (Simulasi Drone 2)
        img2_cv = cv2.warpPerspective(img1_cv, H, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 5. Transformasi ke Tensor PyTorch
        img1_tensor = self.transform(img1_cv)
        img2_tensor = self.transform(img2_cv)
        
        # 6. Kalkulasi Ground Truth Matrix 768x768 murni dari rumus matematika Homografi!
        W_feat, H_feat = 32, 24
        scale_x = crop_w / W_feat # 32 piksel
        scale_y = crop_h / H_feat # 32 piksel
        
        N = W_feat * H_feat
        gt_matches = torch.zeros((N, N))
        
        # Susun daftar koordinat pusat piksel fitur di img1
        # Menggunakan grid spasial
        grid_y, grid_x = np.mgrid[0:H_feat, 0:W_feat]
        pixel_x = grid_x * scale_x + (scale_x / 2.0)
        pixel_y = grid_y * scale_y + (scale_y / 2.0)
        
        # Array koordinat [X, Y, 1] untuk perkalian matriks Homografi
        pts1 = np.stack([pixel_x.ravel(), pixel_y.ravel(), np.ones_like(pixel_x.ravel())], axis=0)
        
        # Lakukan perkalian matriks secara simultan ke semua titik
        pts2 = H.dot(pts1)
        pts2 = pts2 / (pts2[2, :] + 1e-8) # Normalisasi koordinat homogen
        
        # Kembalikan koordinat baru ke sistem Grid (Feature Map 32x24)
        grid2_x = np.floor(pts2[0, :] / scale_x).astype(int)
        grid2_y = np.floor(pts2[1, :] / scale_y).astype(int)
        
        # Cari titik mana saja yang mendarat dengan aman di dalam batas img2
        valid_mask = (grid2_x >= 0) & (grid2_x < W_feat) & (grid2_y >= 0) & (grid2_y < H_feat)
        
        # Masukkan ke dalam matriks Ground Truth
        for idx1 in range(N):
            if valid_mask[idx1]:
                idx2 = grid2_y[idx1] * W_feat + grid2_x[idx1]
                gt_matches[idx1, idx2] = 1.0
                
        return img1_tensor, img2_tensor, gt_matches
