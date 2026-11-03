# Pre-registered decision rules

English summary. The rules below were committed before the new training runs were evaluated; amendments and clarifications are dated, and none changed a criterion after its results were seen. Units: UseGeo pairs along the flight lines at d = 1-2 for accuracy, all pairs at d = 1-8 for robustness; registration success = geometric gate passed and median homography grid error at most 5% of the image width. Uncertainty: cluster bootstrap over straight segments within each flight, hierarchical over seeds for trained variants. Rules: pruning (8 vs 18 blocks) is called equivalent when the 90% CI lies within ±1% of AUC@5 and ±0.10 px of median error; the variant without attention body is non-inferior when its 90% CI excludes losses beyond those margins and 5 points of registration success at overlap 0.2-0.6, in which case the claim that the body is needed is withdrawn; other factors are described, with attributions only where the 95% CI excludes zero; the public reproduction is said to replicate within the margin under the same thresholds. Amendment 3 (3 November 2026) concerns measurement, not a rule: the whole-pipeline throughput was repeated five times with the GPU in a power-saving setting (core clock capped at 2230 MHz, lowered voltage curve), the earlier stock repetition is kept as a reference, and two defects found on the way were corrected before any throughput result was used (the SIFT keypoint cap of the throughput script, now 8000 as in the accuracy pipeline, and the wrap-around of the RAPL energy counter). In the copy below, the name of the local GPU power profile and one operational note about the workstation, both unrelated to the study, are given in neutral wording; rules, dates, and results are unchanged.

---

Original (Indonesian):

# Pra-registrasi revisi Artikel A (IJIES 20267167) — ditulis 2026-11-01 23:55 WIB, SEBELUM hasil pelatihan baru mana pun

Status saat ditulis: pelatihan T_d8_s2 dan T_d8_s3 baru di epoch 1; belum ada checkpoint, kurva q15, atau angka UseGeo dari
run baru. Analisis ulang data LAMA (X1, X2, X3 pada keluaran Oktober, SIFT P26/P30) sudah dilihat dan dipakai untuk
menetapkan kriteria benar-GT di bawah (tercatat di log PROGRESS 1 Nov). Rencana induk: `RENCANA_REVISI_A_IJIES.md`.

## 1. Unit, data, protokol tetap
- Akurasi: UseGeo, pasangan sepanjang jalur d = 1–2 (1418; sepanjang jalur = tiap langkah |Δκ| < 15° dan < 60 m),
  penilai e13 (`e13_refine.score`), keluaran kasar ('none') dan dengan penghalus CPU ('B' = MAGSAC++ 5 px → pra-lengkung
  H0⁻¹ → LK piramida 21×21, maxLevel 3, 30 iterasi/0,01 → petakan balik), tanpa perubahan parameter.
- Ketahanan: UseGeo d = 1–8, SEMUA pasangan (sepanjang jalur + belokan). Registrasi sukses = gate_v2 lolos DAN galat
  kisi-H (kisi 32×24 ber-LiDAR) ≤ 99,45 px (5% lebar 1989 px) [utama]; ≤ 39,78 px (2%) [sekunder]. Diringkas per bin
  tampalan terukur (pose GT + kedalaman, `results/a4/overlap_usegeo_gt.json`): [0;0,2), [0,2;0,4), [0,4;0,6), [0,6;0,8),
  [0,8;1]. Ringkasan ketahanan utama = sukses pada tampalan [0,2;0,6).
- Tekstur rendah (Seneca, Bellus d = 1–3) dan sheffield: hanya gerbang, dilaporkan sebagai konsistensi geometris.
- Biaya: TensorRT FP16 batch 1 (protokol `cloud_e14/scripts/measure.py`), energi NVML + RAPL jendela tunak, throughput
  seluruh pipa 5 ulangan; profil daya stok 300 W, GPU tanpa layar, tanpa beban lain di GPU.
- Seleksi checkpoint semua run (termasuk pembanding hasil latih ulang): rerata inlier RANSAC tertinggi pada q15,
  epoch 1–12. Pembanding sebagaimana terbit: bobot resmi, tanpa batas jumlah kecocokan.

## 2. Ketidakpastian
- Bootstrap klaster: segmen jalur (`crefine/data.py:segments`) diresampel di dalam tiap penerbangan, 10.000×
  (statistik akhir), benih 20261101.
- Model berbenih: tiap benih dilaporkan; perbandingan varian memakai bootstrap hierarkis (resampel benih, lalu klaster);
  SD antarbenih dilaporkan terpisah dari IK adegan.

## 3. Aturan kalimat klaim (mengikat)
- D1 pemangkasan (8 vs 18 blok, 3 benih masing-masing): margin ekuivalensi AUC@5 −1% relatif terhadap 18 blok
  (konvensi MLPerf 99%, juga dipakai Eksperimen D) dan galat median +0,10 px. IK 90% Δ(8−18) di dalam ±margin →
  "equivalent within the margin"; IK 95% memuat 0 tetapi IK 90% melewati margin → "no detectable loss in this experiment";
  selain itu kerugian dilaporkan. Ketahanan dilaporkan sebagai Δ sukses [0,2;0,6) dengan IK.
- D2 badan perhatian (0, 4, 8 lapis): 0 lapis disebut non-inferior bila batas bawah IK Δ(0−8) AUC@5 > −0,98 poin, batas
  atas IK Δ galat < +0,10 px, DAN batas bawah IK Δ sukses [0,2;0,6) > −5 poin. Bila non-inferior → klaim bahwa badan
  perhatian diperlukan pada rezim survei dicabut; judul dan kontribusi dirumuskan ulang (badan dideskripsikan sebagai
  komponen dengan biaya terukur, bukan keharusan). Bila batas atas IK Δ sukses [0,2;0,6) < −5 poin → peran badan
  dinyatakan sebesar selisih itu. Selain itu "tidak tersimpulkan", angka dilaporkan.
- D3 kisi /32, resolusi internal 480×640, inisialisasi bersih: deskriptif; kalimat atribusi hanya untuk efek yang IK-nya
  tidak memuat 0.
- D4 pelatihan setara pembanding (LGFCTR, MatchFormer): tabel 2×2 {bobot resmi, latih ulang udara} × {tahap halus
  sendiri, jalur kasar + penghalus identik}. "Pelatihan udara menjelaskan sebagian selisih" hanya bila latih ulang
  memperbaiki AUC@5 atau galat pembanding dengan IK yang tidak memuat 0; selain itu "pelatihan udara tidak menjelaskan
  selisih". Bila latih ulang merusak (epoch 1 < 0,8× inlier q15 epoch 0), satu kali ulang dengan lr 2e-5 diizinkan
  dan dilaporkan.
- D5 replikasi data publik: disebut "mereplikasi dalam margin" bila batas bawah IK Δ(publik − privat bersih) AUC@5 >
  −0,98 dan batas atas IK Δ galat < +0,10 px; selain itu selisihnya dilaporkan sebagai batas replikasi.
- D6 pencocok efisien dan SIFT: tanpa keputusan; gambar Pareto; bila usulan didominasi pada akurasi-terkendali,
  ketahanan, dan biaya sekaligus, naskah menyatakannya.
- Presisi di rezim survei tidak diklaim sebagai sifat arsitektur kecuali selisihnya bertahan di skenario terkendali
  (penghalus identik) dengan IK yang tidak memuat 0.

## 4. Penyimpangan
Setiap perubahan sesudah hasil terlihat dicatat di bagian ini dengan tanggal dan alasan; tidak ada perubahan kriteria
diam-diam.

### Amandemen 1 — 2026-11-02 00:00 WIB (sebelum data publik diunduh dan sebelum hasil run baru)
- D5 replikasi data publik: checkpoint = epoch 12 (tanpa seleksi pada data privat q15, agar resepnya sepenuhnya
  publik); checkpoint pilihan q15 dilaporkan sebagai sekunder. Inisialisasi bersih (Swin-B ImageNet, pencocok acak).
- Data publik: ortho UAV OpenAerialMap (lisensi CC-BY 4.0), pasangan pseudo-GT dari penerbangan ODM berlisensi CC0
  (aukerman, lewis, zoo) dengan guru dan saringan yang sama. Dikecualikan: set uji (UseGeo, sheffield_park_1, Seneca,
  Bellus) dan apa pun dalam radius 100 km dari situs uji; sand_key dikeluarkan karena hanya ±16 km dari
  sheffield_park_1 (keduanya di Florida, GPS EXIF 28,04°/−82,70° dan 27,96°/−82,83°).

### Amandemen 2 — 2026-11-02 00:25 WIB (sebelum run publik T_pub_s1 dimulai; belum ada hasil run publik)
Penyimpangan sumber data publik dari amandemen 1, dicatat sebelum pelatihan; aturan eksklusi tidak dilonggarkan.
- Pasangan pseudo-GT: ketiga penerbangan ODM CC0 yang direncanakan gugur oleh aturan 100 km (aukerman 10 km,
  lewis 25 km, zoo 24 km dari Bellus; GPS EXIF). Pengganti: ODM Monterey Road Bend (Australia, CC-BY-SA 4.0,
  14.846 km dari situs uji terdekat) dan DroneMapper Back 9 (Colorado, data contoh publik tanpa lisensi tertulis,
  2.105 km). Guru, saringan (≥ 50 cocok, rasio inlier ≥ 0,4), dan batas 687 pasangan tetap.
- Sumber potongan sintetis: unduhan tambahan OpenAerialMap dihentikan (laju unduh terlalu rendah untuk tenggat);
  dipakai 7 ortho UAV OAM yang sudah ada secara lokal, termasuk 2 berlisensi CC BY-NC 4.0 (penggunaan riset
  nonkomersial; citra tidak didistribusikan ulang), 2 di antaranya disisihkan untuk loss validasi
  (Talon_Ridge_Middle_School, Salut), ditambah foto nadir mentah kedua penerbangan publik di atas sebagai sumber
  potongan (5 ortho + 1059 foto). Daftar sumber + lisensi: `data_publik_arev/public_sources.json`.
- Konsekuensi yang dinyatakan di naskah: set publik lebih kecil dan berbeda komposisinya dari set privat; hasil D5
  dibaca sebagai batas bawah replikasi, bukan ekuivalensi data.

### Klarifikasi 1 — 2026-11-02 07:01 WIB (sebelum statistik berbenih D1–D3 dihitung; yang sudah terlihat hanya epoch terpilih q15)
- Keluaran acuan D1–D3 dan D5 = sistem lengkap dengan penghalus CPU ('B'), seperti Tabel 2–3 versi terkirim; keluaran
  kasar ('none') dilaporkan sebagai sekunder dengan rumus yang sama.
- Metrik kelompok varian = rerata atas benih dari metrik per run; galat median per run dihitung pada subset pasangan
  yang galatnya terdefinisi di SEMUA run yang dibandingkan (estimand Δ_sub); AUC atas semua 1418 pasangan (gagal = 180°).
- Bootstrap hierarkis per ulangan: benih diresampel dengan pengembalian di dalam tiap varian, lalu segmen jalur
  diresampel berstrata penerbangan (resampel segmen yang sama untuk kedua varian karena pasangannya sama); 10.000×,
  benih 20261101. IK 90% (TOST) dan 95% dari persentil.
- Margin D1 AUC@5 = −1% × rerata AUC@5 18 blok (dihitung dari data, dilaporkan); D2 memakai −0,98 poin tetap.
- Sukses ketahanan = registrasi benar-GT (gate_v2 ∧ galat kisi-H ≤ 99,45 px) pada pasangan UseGeo d = 1–8 dengan
  tampalan terukur di [0,2; 0,6); untuk usulan dan variannya dari keluaran akhir (dengan penghalus, X3 'refine').
- (tambahan 07:02, sebelum ada hasil L0, L4, s32, r480, clean, dan publik) Taraf IK: D1 dan D2 memakai IK 90%
  (sisi satu α = 0,05, konsisten dengan TOST D1), demikian pula D5; D3 "IK tidak memuat 0" memakai IK 95%. IK lain
  dilaporkan sebagai pelengkap. Arah: Δ AUC < 0 dan Δ galat > 0 = kerugian; selisih yang menguntungkan dan IK 95%-nya
  tidak memuat 0 dilaporkan sebagai keunggulan, bukan "ekuivalen".

### Amandemen 3 — 2026-11-03 08:30 WIB (throughput seluruh pipa; latensi dan energi engine tidak berubah)
- Profil daya: §1 menetapkan profil stok 300 W ('Bench') untuk biaya. Latensi engine (3 sesi) dan energi engine
  (2 ulangan) di RTX 5070 Ti sudah selesai di profil itu dan tetap dipakai. Atas instruksi operasional peneliti
  (3 Nov 08:15: instruksi peneliti agar GPU tetap di profil hemat daya), throughput seluruh pipa diulang LENGKAP 5 ulangan di
  profil hemat daya LACT (800 mV, clock maks 2230 MHz, batas 250 W) sebagai himpunan `results/arev/g2/thr_powersave/`.
  Alasannya operasional (daya listrik selama kampanye), bukan hasil: ulangan stok yang sudah terlihat (thr/1 lengkap,
  thr/2 sebagian, hanya konfigurasi GPU) disimpan dan dilaporkan sebagai rujukan stok, dan rasio hemat-daya/stok per
  konfigurasi dicatat di `summary.json` (`thr_stock_vs_set`).
- Koreksi SIFT (kesalahan kode, bukan pilihan analisis): pengukur throughput memakai `cv2.SIFT_create()` tanpa batas,
  sedangkan angka akurasi SIFT (P30 → X4) berasal dari `p26_sift.match` dengan `nfeatures=8000`. Pengukur kini sama
  dengan jalur akurasi; hasil SIFT lama dipindah ke `results/arev/g2/thr/_invalid_sift_20261103/` (tidak dipakai).
- Koreksi RAPL (kesalahan kode): `energy_uj` berputar ulang tiap 65,5 kJ; pencacah kini dibaca tiap 1 s dengan koreksi
  putaran ulang. Hanya jalan > ±13 menit yang terdampak (SIFT Sheffield lama); jalan pendek lain sudah benar.
