import pandas as pd

# --- 1. Struktur Kolom dan Nama File ---
columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# --- 2. Baca File Saham Baru (skip header baris 1-3) ---
df_isat = pd.read_csv("saham/data_ISAT.csv", names=columns)
df_tins = pd.read_csv("saham/data_TINS.csv", names=columns)
df_mdka = pd.read_csv("saham/data_MDKA.csv", names=columns)

# --- 3. Tambahkan Kolom Ticker ---
df_isat['Ticker'] = 'ISAT.JK'
df_tins['Ticker'] = 'TINS.JK'
df_mdka['Ticker'] = 'MDKA.JK'

# --- 4. Gabungkan Ketiga Saham Baru ---
df_baru = pd.concat([df_isat, df_tins, df_mdka], ignore_index=True)

# --- 5. Atur Ulang Urutan Kolom ---
df_baru = df_baru[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# --- 6. Baca Dataset Saham Gabungan Lama ---
df_lama = pd.read_csv("saham_gabungan.csv")

# --- 7. Gabungkan dengan Data Lama ---
df_gabungan = pd.concat([df_lama, df_baru], ignore_index=True)

# --- 8. Konversi Tanggal dan Urutkan ---
df_gabungan['Date'] = pd.to_datetime(df_gabungan['Date'], errors='coerce')
df_gabungan = df_gabungan.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

# --- 9. Simpan ke File Baru ---
df_gabungan.to_csv("saham_gabungan_final.csv", index=False)
print("✅ Data saham berhasil digabung dan disimpan ke 'saham_gabungan_final.csv'")