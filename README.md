# Big Data Challenge (Satria Data): Waste Image Classification

Eksperimen notebook untuk seleksi Big Data Challenge (Satria Data). Task: klasifikasi gambar sampah ke 3 kelas: `Recyclable`, `Electronic`, `Organic`.

> Repo ini berisi notebook eksperimen (bukan production pipeline). Fokus struktur: reproducibility, clean code, dan tracking eksperimen.

## Struktur Folder

```
.
├── data/                   # dataset (git-ignored, kecuali .gitkeep)
│   ├── train/               # raw training images per kelas (folder Indonesia: 0_Recyclable, dst.)
│   ├── processed/           # hasil stratified split
│   │   ├── train/{class}/
│   │   └── val/{class}/
│   ├── test/                 # gambar test untuk submission
│   ├── submission/           # file CSV hasil prediksi
│   └── ground_truth.csv
├── docs/                    # dokumen soal/panduan challenge
├── models/                  # artefak model terlatih (git-ignored, kecuali .gitkeep)
│   └── {run_name}/
│       ├── model.pt
│       ├── config.json
│       └── training_history.json
├── notebooks/
│   ├── 00_data_split.ipynb   # stratified train/val split
│   ├── 01_eda.ipynb          # exploratory data analysis
│   ├── experiments/          # notebook eksperimen aktif
│   └── archive/              # notebook lama, di-exclude dari lint
├── utils/                    # kode Python shared, dipakai lintas notebook
│   ├── config.py              # load & validasi YAML config, simpan run manifest
│   ├── dataset.py             # Dataset/DataLoader (albumentations transform)
│   ├── download_dataset.py    # download dataset dari Google Drive (gdown)
│   ├── engine.py               # training loop, EarlyStopping, run_phase
│   ├── seed.py                  # set_seed untuk reproducibility
│   ├── split.py                  # stratified split raw -> processed/{train,val}
│   └── submission.py             # preprocessing test image & generate submission CSV
├── requirements.txt
├── ruff.toml
└── .pre-commit-config.yaml
```

## Cara Pemakaian

### 1. Buat environment

```bash
conda create -n bdc python=3.12 -y
conda activate bdc
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Setup pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

Jalankan manual di semua file (opsional, biasanya buat cek awal):

```bash
pre-commit run --all-files
```

### 4. Download dataset

```bash
python utils/download_dataset.py
```

Dataset ditarik dari folder Google Drive (lihat `FOLDER_ID` di `utils/download_dataset.py`) ke `data/`.

Alternatif, download manual langsung dari Google Drive: https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7

### 5. Split data train/val

Jalankan `notebooks/00_data_split.ipynb`, pakai `utils/split.py` (`split_and_organize_dataset`) buat stratified split raw folder ke `data/processed/{train,val}/{class_name}`.

### 6. EDA

`notebooks/01_eda.ipynb`: cek distribusi kelas, ukuran gambar, sample visual, dll.

### 7. Eksperimen training

Notebook eksperimen ada di `notebooks/experiments/`. Tiap notebook:
- pakai `utils/seed.py` (`set_seed`) buat reproducibility,
- pakai `utils/dataset.py` buat DataLoader,
- pakai `utils/engine.py` buat training loop + early stopping,
- simpan config run & manifest hasil training lewat `utils/config.py`,
- artefak (`model.pt`, `config.json`, `training_history.json`) disimpan di `models/{run_name}/`.

### 8. Generate submission

Pakai `utils/submission.py` (`preprocess_for_inference`, `load_test_images`, dll.) buat load `data/test/`, jalankan inferensi, dan tulis CSV ke `data/submission/`, sesuai format `data/submission/template_submission.csv`.

## Konfigurasi Eksperimen

Tiap run training dikonfigurasi lewat YAML config dengan field wajib: `run_id`, `seed`, `model`, `training` (divalidasi oleh `load_config` di `utils/config.py`). Manifest hasil run (config + metrik + path artefak) disimpan otomatis via `save_run_manifest`.

## Requirements

Semua dependency ada di `requirements.txt` (pip freeze), termasuk:
- `torch`, `torchvision`, `torchaudio`, `timm`: training & model
- `albumentations`: augmentasi image
- `scikit-learn`, `torchmetrics`: evaluasi
- `pandas`, `numpy`, `matplotlib`, `seaborn`: analisis & visualisasi
- `jupyterlab`: notebook environment
- `ruff`, `pre-commit`: lint/format & git hooks

Install:

```bash
pip install -r requirements.txt
```

## Lint & Format

Pakai [Ruff](https://docs.astral.sh/ruff/) buat lint + format, dikonfigurasi di `ruff.toml`. Notebook lama (`notebooks/archive/`, `notebooks/01_eda.ipynb`) di-exclude dari lint karena sifatnya eksploratif/arsip.

```bash
ruff check .          # lint
ruff format .         # format
```

## Pre-commit Hooks

Dikonfigurasi di `.pre-commit-config.yaml`:

| Hook | Fungsi |
|---|---|
| `ruff` | lint check + auto-fix |
| `ruff-format` | format code |
| `trailing-whitespace` | hapus trailing whitespace |
| `end-of-file-fixer` | pastikan file diakhiri newline |
| `check-yaml` | validasi syntax YAML |
| `check-added-large-files` | cegah file > 10MB ke-commit gak sengaja (mis. `model.pt`) |
| `check-merge-conflict` | cegah marker conflict ke-commit |
| `conventional-pre-commit` | validasi commit message ikut [Conventional Commits](https://www.conventionalcommits.org/) |

## Data & Model Artifacts

`data/` dan `models/` di-git-ignore (kecuali `.gitkeep`): dataset dan bobot model tidak disimpan di git, harus di-download/di-generate ulang lewat langkah di atas. Ini menjaga repo tetap ringan dan menghindari commit file besar secara gak sengaja.
