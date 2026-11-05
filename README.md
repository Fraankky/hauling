# Hauling Offline

Pipeline lokal untuk mendeteksi dan melacak kendaraan dengan YOLO + ByteTrack, mengukur kecepatan antararea, membaca nomor lambung, serta mencatat event. Inferensi memakai model dan video yang disediakan pengguna; kode ini tidak mengakses layanan produksi.

## Menjalankan pipeline

Gunakan Python 3.11+ dan pasang dependensi proyek. Siapkan video serta bobot YOLO secara lokal, lalu jalankan:

```bash
python -m hauling \
  --video /path/to/video.mp4 \
  --vehicle-model /path/to/vehicle.pt \
  --config _track_km18_01_V2.conf \
  --output-dir /path/to/new-run \
  --device cpu
```

Direktori `--output-dir` harus baru. Pipeline membaca video sampai EOF; `--max-frames 300` membatasi uji singkat. `--no-save-video` mematikan video anotasi. Untuk pembacaan nomor lambung, sertakan empat argumen model OCR: `--plate-character-model`, `--plate-letter-model`, `--plate-digit-classifier`, dan `--plate-letter-classifier`.

Model `.pt` dapat memakai CPU atau CUDA. Model TensorRT `.engine` memerlukan GPU NVIDIA/CUDA; gunakan `--device 0` atau indeks GPU yang sesuai. Engine harus cocok dengan GPU, runtime TensorRT, bobot, dan pengaturan input. Mengganti video saja tidak mengharuskan ekspor ulang engine.

## Perbandingan PyTorch dan TensorRT FP32

`hauling_tensorrt_colab.ipynb` menjalankan dua runtime secara **bergantian** dengan lima model, video, konfigurasi, dan opsi yang sama. Urutan selnya:

1. Siapkan GPU Colab dan salin input ke disk lokal.
2. Ekspor lima model `.pt` menjadi `.engine` FP32, atau pakai engine tersimpan setelah pemeriksaan manifest dan checksum. Ekspor dilakukan sebelum pengukuran.
3. Jalankan pipeline sekali dengan `.pt` dan sekali dengan `.engine`, lalu bandingkan hasil event dan waktu tahapnya.

Notebook menyimpan hasil dengan nama unik agar tidak menimpa run sebelumnya. Durasi pipeline mencakup pemrosesan dan penulisan output, bukan waktu ekspor engine atau penyalinan ke Drive. Waktu `vehicle_detect` juga mencakup transfer hasil dan ByteTrack, sehingga bukan latency inferensi GPU murni. Perbedaan teks OCR perlu diperiksa terhadap label manual sebelum dinilai sebagai perbedaan akurasi.

## Struktur kode

- `hauling/cli.py`: argumen dan pemanggilan pipeline.
- `hauling/config.py`: membaca serta memvalidasi parameter inferensi dan area.
- `hauling/detector.py`: memuat model kendaraan, deteksi, dan tracking.
- `hauling/plate.py` dan `hauling/state.py`: OCR nomor lambung dan state kendaraan.
- `hauling/pipeline.py`: pemrosesan video dan penulisan event.
- `hauling/summary.py`: penyusunan ringkasan hasil run.
- `hauling/comparison.py`: mencocokkan event dari dua runtime berdasarkan frame crossing.
