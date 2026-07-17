# Penjelasan Notebook: `03_recyclable_error_diagnosis.ipynb`

## 1. Latar Belakang dan Tujuan

Notebook ini merupakan tahap **diagnosis error** (error analysis) dalam pipeline klasifikasi gambar sampah ke dalam tiga kelas: **Recyclable**, **Electronic**, dan **Organic**. Setelah empat model (MobileNetV2, ResNet, ConvNeXtV2, Swin-Tiny) dilatih dan dievaluasi secara umum di notebook `evaluation.ipynb`, notebook ini masuk lebih dalam untuk menjawab pertanyaan spesifik:

> **Mengapa model — khususnya Swin-Tiny yang performanya terbaik — masih salah mengklasifikasikan sampah *Recyclable* sebagai *Organic*?**

Kesalahan **Recyclable → Organic** dipilih sebagai fokus karena ini adalah jenis kesalahan yang paling sering muncul dan paling merugikan secara praktis (barang yang sebenarnya bisa didaur ulang malah dibuang ke kategori sampah organik). Notebook ini menguji tiga hipotesis (H1, H2, H3) untuk mencari akar penyebab kesalahan tersebut, bukan sekadar melaporkan angka akurasi.

Tiga hipotesis yang diuji:

- **H1 — Hipotesis Ketidakpastian Model (Confidence-based):** Apakah kesalahan Rec→Org terjadi saat model "ragu" (low confidence), atau justru model "yakin salah" (high confidence)? Ini membedakan antara *kesalahan wajar karena ambiguitas* vs *kesalahan sistematis yang berbahaya*.
- **H2 — Hipotesis *Shortcut* Warna:** Apakah model menggunakan warna (terutama hue/kehijauan-kecokelatan) sebagai sinyal utama untuk mengenali "Organic", sehingga sampah recyclable yang kebetulan berwarna serupa (misalnya kertas cokelat, kardus) ikut tertipu?
- **H3 — Hipotesis Subclass/Cluster Tersembunyi:** Apakah kelas "Recyclable" sebenarnya terdiri dari beberapa sub-kelompok visual (misalnya plastik, logam, kertas, kaca) dan hanya sub-kelompok tertentu yang rawan disalahklasifikasikan sebagai Organic?

## 2. Struktur dan Alur Kerja Notebook

### 2.1 Setup dan Konfigurasi (Sel 1–2)

Notebook mengimpor pustaka standar data science (`numpy`, `pandas`, `matplotlib`, `seaborn`), pustaka computer vision (`cv2`, `PIL`), pustaka machine learning (`torch`, `sklearn`), serta pustaka khusus untuk analisis lanjutan: `umap-learn` (reduksi dimensi non-linear) dan `scipy.stats.mannwhitneyu` (uji statistik non-parametrik).

Konfigurasi penting yang didefinisikan:
- **Device**: mendeteksi otomatis MPS (Apple Silicon GPU) atau CPU.
- **Label space kompetisi**: `{0: Recyclable, 1: Electronic, 2: Organic}` — penting dicatat karena tidak sama dengan urutan indeks kelas internal model (lihat 2.2).
- **MODEL_DIRS**: peta ke empat model yang sudah dilatih sebelumnya, disimpan di folder `../models/`.
- Notebook menggunakan sistem **caching artefak** (`utils/diagnostics.py` — `get_or_compute_logits`, `get_or_compute_features`) sehingga logits dan fitur embedding tidak perlu dihitung ulang setiap kali notebook dijalankan; hasil disimpan sebagai file `.npy` di `notebooks/artifacts/`.

### 2.2 Load & Konsolidasi Data (Sel 4–7)

Bagian ini menyatukan seluruh informasi menjadi satu `tidy_df` (data frame "rapi") — praktik data science yang baik agar analisis selanjutnya tidak perlu bongkar-pasang banyak struktur data terpisah.

Langkah pentingnya:
1. **Ground truth** dibaca dari `kuncijawaban.csv` dan dicocokkan dengan file gambar di `data/test/`.
2. **Validasi konsistensi label** — setiap model punya `config.json` sendiri dengan `class_indices` (urutan kelas internal saat training, misalnya `Electronic=0, Organic=1, Recyclable=2`). Notebook melakukan `assert` bahwa keempat model punya pemetaan indeks kelas yang **sama**, lalu membuat `MODEL_IDX_TO_COMP_LABEL` untuk menerjemahkan indeks internal model ke label kompetisi. Ini adalah langkah **defensif** yang krusial — kalau urutan kelas antar-model berbeda tapi tidak diperiksa, seluruh analisis berikutnya akan salah tanpa disadari.
3. **Inferensi**: keempat model dijalankan pada seluruh 1458 gambar test, hasilnya berupa logits → di-*softmax* menjadi probabilitas → direorder ke label space kompetisi.
4. Semua probabilitas dan prediksi digabung ke `tidy_df` dengan penamaan kolom yang konsisten (`prob_<Kelas>_<model>`, `pred_label_<model>`).

### 2.3 Tabel Metrik Baseline (Sel 9)

Sebelum menyelidiki error, notebook membuat ulang ringkasan performa per model (precision, recall, F1 per kelas, accuracy, macro-F1, weighted-F1) sebagai **titik referensi**. Hasilnya:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Swin-Tiny | 0.944 | 0.944 |
| ConvNeXt | 0.931 | 0.937 |
| MobileNetV2 | 0.857 | 0.844 |
| ResNet | 0.599 | 0.588 |

Swin-Tiny adalah model terbaik, sehingga dipilih sebagai **subjek utama** analisis error di seluruh notebook ini (meskipun model lain tetap digunakan untuk perbandingan silang).

### 2.4 Error Intersection — Uji Variansi Antar-Model (Sel 11–12)

Bagian ini mengekstrak **himpunan sampel Recyclable yang salah diprediksi sebagai Organic** (`error_sets`) untuk masing-masing dari empat model, lalu menghitung **Jaccard similarity** antar himpunan tersebut (rasio irisan terhadap gabungan).

Tujuannya: membedakan apakah error ini bersifat **acak/khusus per-arsitektur** (Jaccard rendah → tiap model salah pada sampel yang berbeda-beda, mengindikasikan noise model) atau **sistematis/melekat pada data** (Jaccard tinggi → semua model salah pada sampel yang sama, mengindikasikan gambar tersebut memang secara inheren ambigu).

Hasil menunjukkan Jaccard tertinggi antara ConvNeXt dan Swin-Tiny (0.324) — dua model paling kuat — sementara ResNet (model terlemah) punya overlap rendah dengan yang lain. Ini mengarah pada dugaan bahwa sebagian error memang disebabkan oleh **ambiguitas visual pada gambar itu sendiri**, bukan sekadar kelemahan acak satu arsitektur. Histogram "agreement count" divisualisasikan untuk melihat berapa sampel yang disepakati salah oleh 1, 2, 3, atau 4 model sekaligus.

### 2.5 Stratifikasi Confidence — Uji H1 (Sel 14)

Untuk sampel yang salah diprediksi Swin-Tiny, notebook memplot distribusi **max softmax probability** dan **p(Organic)** khusus pada kelompok error tersebut, lalu memisahkan menjadi:
- **High-confidence error** (p(Organic) > 0.9): model *sangat yakin* tapi salah — ini paling berbahaya karena tidak bisa dideteksi lewat threshold confidence biasa.
- **Low-confidence error** (p(Organic) < 0.6): model *ragu-ragu* — kesalahan jenis ini relatif "wajar" dan bisa ditangkap dengan mekanisme *reject/abstain* di produksi.

Bagian ini langsung menjawab H1: apakah error terjadi karena model bingung (low-confidence) atau karena model punya *false belief* yang kuat (high-confidence, lebih serius).

### 2.6 Audit Visual (Sel 16)

Fungsi `save_error_grid` membuat **grid montase gambar** — 40 gambar dengan prediksi p(Organic) tertinggi dan 40 gambar dengan p(Organic) terendah — disimpan sebagai PNG ke `reports/figures/`. Ini adalah langkah **qualitative inspection** khas dalam error analysis: statistik saja tidak cukup, seorang praktisi harus benar-benar *melihat* gambar yang salah untuk menangkap pola yang tidak tertangkap angka (misalnya pencahayaan, tekstur, latar belakang).

### 2.7 Uji *Color Shortcut* — H2 (Sel 18–19)

Notebook menghitung **rata-rata nilai HSV** (Hue, Saturation, Value) untuk tiga kelompok gambar:
- `rec_correct`: Recyclable yang diprediksi benar,
- `rec_to_org`: Recyclable yang salah diprediksi Organic,
- `org_correct`: Organic yang diprediksi benar.

Hasil (rata-rata Hue): rec_correct = 52.2, rec_to_org = **40.7**, org_correct = 32.2. Nilai Hue kelompok error berada **di antara** dua kelompok benar, tetapi lebih dekat ke arah Organic. Untuk menguji signifikansi statistik perbedaan ini digunakan **Mann-Whitney U test** (uji non-parametrik, dipilih karena tidak mengasumsikan distribusi normal pada data warna gambar) yang membandingkan `rec_correct` vs `rec_to_org` pada tiap kanal HSV, sekaligus dicek channel tersebut "lebih dekat" ke rata-rata kelompok mana (Organic atau Recyclable).

Ini menguji H2: apakah model menggunakan **warna sebagai *shortcut feature*** — pola belajar yang secara teknis "berhasil" pada data training tapi rentan gagal (*spurious correlation*) ketika ada objek recyclable yang warnanya menyerupai sampah organik (misal kardus cokelat, kertas kusam).

### 2.8 Analisis Embedding & Subclass — H3 (Sel 21–25)

Ini adalah bagian analisis paling dalam secara teknik:

1. **Ekstraksi fitur** — mengambil *penultimate layer* (fitur sebelum classifier head) dari Swin-Tiny untuk seluruh 1458 gambar, menghasilkan vektor 256-dimensi per gambar (representasi semantik yang dipelajari model, bukan hanya piksel mentah).
2. **UMAP** — mereduksi fitur 256-dimensi ke 2 dimensi untuk visualisasi, sambil menandai secara khusus titik-titik yang merupakan error Rec→Org (marker "x" merah). Tujuannya melihat apakah error-error itu **mengelompok** di satu wilayah tertentu dalam ruang fitur (menandakan sub-kategori Recyclable yang secara sistematis membingungkan model) atau tersebar acak (menandakan noise individual).
3. **K-Means + Silhouette Score** — mengelompokkan *hanya* sampel Recyclable (bukan semua kelas) ke dalam k=3 sampai k=6 cluster, memilih k terbaik berdasarkan **silhouette score** (metrik yang mengukur seberapa rapat & terpisah suatu cluster, tanpa perlu label). k=6 terpilih sebagai yang terbaik (meski scorenya rendah/moderat, ~0.14, mengindikasikan struktur cluster tidak terlalu tegas).
4. **Analisis per-cluster** — untuk tiap cluster dihitung `recall`, `rec_to_org_error_rate`, dan `share_of_total_rec_to_org_errors`. Temuan kunci: **Cluster 2** (65 sampel) memiliki error rate 44.6% dan menyumbang **51.8%** dari seluruh error Rec→Org Swin-Tiny — meskipun ukurannya hanya sekitar 8% dari total sampel Recyclable. Ini adalah temuan paling penting di notebook: sebagian besar masalah model terkonsentrasi pada satu sub-kelompok visual spesifik, bukan tersebar merata.
5. **Montase representatif per-cluster** — untuk tiap cluster, diambil 12 gambar yang paling dekat dengan centroid-nya (representasi visual "prototipe" cluster tersebut), disimpan sebagai PNG. Ini memungkinkan identifikasi *secara manual* jenis objek apa yang mendominasi cluster bermasalah (misalnya: kardus, kertas, atau bahan tertentu yang teksturnya mirip sampah organik).

## 3. Insight dan Kontribusi Metodologis

Notebook ini mencontohkan pendekatan **error analysis yang rigorus** dalam machine learning, dengan beberapa prinsip yang patut dicatat sebagai pembelajaran:

- **Triangulasi metode**: kesalahan yang sama diselidiki dari tiga sudut berbeda — konsistensi antar-model (variance), sinyal tingkat rendah seperti warna (bias data), dan struktur semantik tersembunyi (representasi fitur). Ini menghindari kesimpulan prematur dari satu jenis bukti saja.
- **Uji hipotesis eksplisit** (H1, H2, H3) alih-alih eksplorasi data tanpa arah — setiap sel punya pertanyaan spesifik yang ingin dijawab.
- **Kombinasi kuantitatif dan kualitatif**: statistik (Jaccard, Mann-Whitney, silhouette) dipadukan dengan inspeksi visual langsung (grid gambar, montase cluster) — keduanya saling melengkapi, karena angka bisa menyembunyikan pola yang mata manusia langsung tangkap, dan sebaliknya.
- **Reproducibility & efisiensi**: penggunaan `RANDOM_SEED` konsisten, serta caching logits/features/HSV stats ke disk sehingga re-run notebook tidak perlu menghitung ulang inferensi model yang mahal.
- **Actionable output**: hasil akhirnya bukan sekadar angka, tapi identifikasi **sub-kelompok data spesifik** (cluster 2) yang menjadi target perbaikan konkret — misalnya menambah data augmentasi/collection untuk jenis recyclable tersebut, atau memberi bobot loss lebih besar pada sub-kelompok itu saat retraining.

## 4. Kaitan dengan Notebook Lain

Notebook ini adalah kelanjutan dari notebook eksperimen model (`01_mobilenetv2`, `02_resnet`, `03_convnext`, `04_swin_tiny`) dan `evaluation.ipynb`. Alurnya:

```
Training per model  →  evaluation.ipynb (metrik agregat)  →  03_recyclable_error_diagnosis.ipynb (akar masalah error spesifik)
```

Output notebook ini (temuan cluster bermasalah, bukti *color shortcut*, karakteristik confidence pada error) idealnya menjadi dasar keputusan pada iterasi model berikutnya — apakah perlu data tambahan, augmentasi warna, re-labeling, atau perubahan arsitektur/loss function.
