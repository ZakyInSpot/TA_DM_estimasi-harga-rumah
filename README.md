# Estimasi Harga Rumah Jabodetabek (Streamlit)

Aplikasi web yang memperkirakan harga rumah dan segmen pasarnya dari data yang diisi pengguna.

- **Data:** Daftar Harga Rumah Jabodetabek (Kaggle, `nafisbarizki`), asal Rumah123.com
- **Model:** Random Forest pada log(harga); segmentasi K-Means (4 segmen)
- **Catatan:** harga di dataset adalah harga penawaran, bukan harga transaksi. Hasil hanya perkiraan awal.

## File
| File | Fungsi |
|---|---|
| `streamlit_app.py` | Aplikasi web |
| `model/house_model.joblib` | Model hasil latih |
| `requirements.txt` | Library dan versinya (cocok dengan Python 3.11 ke atas) |
| `train_model.py` | Melatih ulang model dari CSV: `python train_model.py jabodetabek_house_price.csv` |

## Menjalankan di komputer sendiri
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Hosting gratis di Streamlit Community Cloud
1. Buat repo **publik** baru di GitHub, lalu unggah `streamlit_app.py`, `requirements.txt`, dan folder `model/` ke folder utama repo (bukan di dalam subfolder).
2. Buka share.streamlit.io dan masuk dengan akun GitHub.
3. Klik **Create app** (atau **New app**), pilih repo, branch `main`, dan file utama `streamlit_app.py`.
4. Di **Advanced settings**, pilih Python **3.12** (versi library di `requirements.txt` butuh Python 3.11 ke atas).
5. Klik **Deploy**, tunggu beberapa menit, lalu salin link `.streamlit.app` yang muncul.

Aplikasi gratis akan tidur kalau lama tidak dikunjungi. Cukup buka linknya lagi dan klik tombol untuk membangunkannya.
Jika model gagal dimuat, penyebabnya hampir selalu beda versi scikit-learn: jalankan ulang `train_model.py` di lingkungan yang sama.
