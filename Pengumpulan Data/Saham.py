import yfinance as yf
import pandas as pd

# Pilih ticker saham
ticker = 'MDKA.JK'

# Tentukan rentang tanggal
start_date = '2019-01-01'
end_date = '2025-07-09'

# Unduh data dari Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Tampilkan 5 baris pertama
print(data.head())

# Simpan ke file CSV
data.to_csv(f'data_{ticker.replace(".JK", "")}.csv')
print(f"Data berhasil disimpan sebagai data_{ticker.replace('.JK', '')}.csv")
