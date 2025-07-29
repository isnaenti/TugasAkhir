import pandas as pd

# Load semua data CSV
saham_gabungan = pd.read_csv("saham/saham_gabungan_final.csv")
hasil_sent_tins = pd.read_csv("ringtweet/baru/ringtins.csv")
hasil_sent_isat = pd.read_csv("ringtweet/baru/ringisat.csv")
hasil_sent_mdka = pd.read_csv("ringtweet/baru/ringmdka.csv")

# Pastikan format tanggal seragam dan hanya berisi tanggal (tanpa jam)
saham_gabungan['Date'] = pd.to_datetime(saham_gabungan['Date']).dt.date
hasil_sent_tins['date'] = pd.to_datetime(hasil_sent_tins['date']).dt.date
hasil_sent_isat['date'] = pd.to_datetime(hasil_sent_isat['date']).dt.date
hasil_sent_mdka['date'] = pd.to_datetime(hasil_sent_mdka['date']).dt.date

# Filter saham berdasarkan ticker dengan format yang sesuai
saham_tins = saham_gabungan[saham_gabungan['Ticker'] == 'TINS.JK']
saham_isat = saham_gabungan[saham_gabungan['Ticker'] == 'ISAT.JK']
saham_mdka = saham_gabungan[saham_gabungan['Ticker'] == 'MDKA.JK']

# Gabungkan saham dengan data sentimen berdasarkan tanggal
gabung_tins = pd.merge(saham_tins, hasil_sent_tins, left_on='Date', right_on='date', how='inner')
gabung_isat = pd.merge(saham_isat, hasil_sent_isat, left_on='Date', right_on='date', how='inner')
gabung_mdka = pd.merge(saham_mdka, hasil_sent_mdka, left_on='Date', right_on='date', how='inner')

# Gabungkan semua menjadi satu dataframe
gabungan_sentimen_saham = pd.concat([gabung_tins, gabung_isat, gabung_mdka], ignore_index=True)

# Hapus kolom 'date' dari sentimen karena sudah ada 'Date'
gabungan_sentimen_saham.drop(columns=['date'], inplace=True)

# Simpan ke file CSV jika diperlukan
gabungan_sentimen_saham.to_csv("sahamsent_baru.csv", index=False)

# Tampilkan beberapa baris awal
print(gabungan_sentimen_saham.head())

