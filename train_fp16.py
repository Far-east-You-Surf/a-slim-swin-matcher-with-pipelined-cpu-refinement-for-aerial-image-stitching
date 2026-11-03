import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from model import AerialStitchingModel
import os
import time
import yaml
try:
    import psutil
except ImportError:
    psutil = None

def compute_loss(conf_matrix, gt_matches):
    loss = -torch.log(conf_matrix + 1e-6) * gt_matches
    return torch.sum(loss) / (torch.sum(gt_matches) + 1e-6)

def generate_report(stats, filename="training_report_fp16.md"):
    report_content = f"""# Laporan Pelatihan V-Transformer FP16 (Baseline)

## 1. Ringkasan Waktu
- **Mulai Pelatihan:** {stats['start_time_str']}
- **Selesai Pelatihan:** {stats['end_time_str']}
- **Total Durasi:** {stats['duration']}
- **Status Berhenti:** {stats['stop_reason']}

## 2. Keterangan Batch & Epoch
- **Total Epoch Dilalui:** {stats['completed_epochs']} / {stats['total_epochs']}
- **Total Batch per Epoch:** {stats['total_batches']}

## 3. Informasi Data Training
- **Lokasi Data:** {stats['data_path']}
- **Jumlah Pasangan Gambar Virtual per Epoch:** {stats['total_samples']}

## 4. Rata-rata Penggunaan Memori
- **Peak VRAM GPU (CUDA):** {stats['avg_vram']:.2f} GB

## 5. Metrik Loss
- **Loss Terbaik Dicapai:** {stats['best_loss']:.4f}
- **Loss Terakhir:** {stats['final_loss']:.4f}
"""
    with open(filename, "w") as f:
        f.write(report_content)
    print(f"\n[Laporan] Laporan lengkap pelatihan telah disimpan ke '{filename}'")

def train(model, dataloader, optimizer, scaler, device, epochs=10, patience=10, data_path=""):
    model.train()
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    
    best_loss = float('inf')
    final_loss = float('inf')
    patience_counter = 0
    completed_epochs = 0
    stop_reason = "Selesai (Full Epoch)"
    vram_stats = []
    
    accumulation_steps = 8
    
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (img1, img2, gt_matches) in enumerate(dataloader):
                img1, img2 = img1.to(device), img2.to(device)
                gt_matches = gt_matches.to(device)
                
                if batch_idx == 0:
                    optimizer.zero_grad()
                    
                # Gunakan float16 untuk Mixed Precision murni
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    conf_matrix = model(img1, img2)
                    loss = compute_loss(conf_matrix, gt_matches)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                
                epoch_loss += loss.item() * accumulation_steps
                vram_stats.append(torch.cuda.max_memory_allocated() / (1024**3))
                torch.cuda.reset_peak_memory_stats()
                
                if batch_idx % 50 == 0:
                    real_batch_loss = loss.item() * accumulation_steps
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {real_batch_loss:.4f} | VRAM: {vram_stats[-1]:.2f}GB")
                    
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Simpan model murni ke FP16 state dict
                model_fp16 = AerialStitchingModel(d_model=128, nhead=8, num_layers=8).to(device)
                model_fp16.load_state_dict(model.state_dict())
                model_fp16.half() # Konversi paksa ke fp16
                torch.save(model_fp16.state_dict(), "aerial_stitching_fp16_best.pth")
                print(f"--> [Model Terbaik Disimpan] Loss turun ke {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"==== Early Stopping Terpicu pada Epoch {epoch+1}! ====")
                    stop_reason = "Terpicu Early Stopping"
                    break
                    
            completed_epochs += 1
            final_loss = avg_loss

    except KeyboardInterrupt:
        print("\n[User Interrupt] Pelatihan dihentikan secara manual!")
        stop_reason = "Dihentikan Manual"
        
    finally:
        end_time = time.time()
        duration_sec = int(end_time - start_time)
        hours, remainder = divmod(duration_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        avg_vram = sum(vram_stats)/len(vram_stats) if vram_stats else 0
        
        stats = {
            'start_time_str': start_time_str,
            'end_time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            'duration': f"{hours}h {minutes}m {seconds}s",
            'stop_reason': stop_reason,
            'completed_epochs': completed_epochs,
            'total_epochs': epochs,
            'total_batches': len(dataloader),
            'data_path': data_path,
            'total_samples': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0,
            'avg_vram': avg_vram,
            'best_loss': best_loss,
            'final_loss': final_loss,
        }
        generate_report(stats, "training_report_fp16.md")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    num_layers = config["model"].get("num_layers", 8)
    model = AerialStitchingModel(d_model=128, nhead=8, num_layers=num_layers).to(device)
    
    print("Menggunakan presisi murni FP16 / FP32 (Mixed Precision Native). Tidak ada kuantisasi FP8.")

    from dataset_homography import HomographyOrthoDataset
    data_path = config["dataset"]["train_path"]
    
    # Supaya latihannya selesai sedikit lebih cepat tapi tetap merepresentasikan baseline,
    # kita gunakan ukuran dataset yang sesuai dengan config awal.
    full_dataset = HomographyOrthoDataset(data_path=data_path, epoch_size=4000)
    dataloader = DataLoader(full_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = config["training"]["epochs"]
    patience = config["training"].get("early_stopping_patience", 10)
    
    train(model, dataloader, optimizer, scaler, device, epochs=epochs, patience=patience, data_path=data_path)
