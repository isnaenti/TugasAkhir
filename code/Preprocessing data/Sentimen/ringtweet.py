# === KODE LENGKAP PENGOLAHAN DATA LABELING MANUAL UNTUK ANALISIS HARIAN ===

import pandas as pd

# === Load Dataset Tweet yang Sudah Ada Label Manual ===
# Ganti dengan path file hasil labeling manual Anda
path_file = "hasilsentvader/perbaikan/set-tins.csv"
df = pd.read_csv(path_file)
import pandas as pd

# Pastikan kolom 'date' dalam format datetime
df['date'] = pd.to_datetime(df['date'])

# Agregasi harian
rekap_harian = df.groupby('date').agg(
    total_tweet=('vader_label', 'count'),
    positif_tweet=('vader_label', lambda x: (x == 2).sum()),
    negatif_tweet=('vader_label', lambda x: (x == 1).sum()),
    netral_tweet=('vader_label', lambda x: (x == 0).sum()),
    avg_compound=('vader_compound', 'mean')
).reset_index()

rekap_harian.to_csv("ringtweet/baru/ringtins.csv", index=False)
print("✅ Rekap harian berhasil disimpan ke 'rekap_sentimen_per_hari.csv'")