import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import datetime

# Fungsi load model dan scaler
def load_model_and_scalers(ticker):
    model_path = f"modelXGBOOST/{ticker}/model4.pkl"
    scalerX_path = f"modelXGBOOST/{ticker}/X_scaler.pkl"
    scalerY_path = f"modelXGBOOST/{ticker}/y_scaler.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scalerX_path) and os.path.exists(scalerY_path)):
        raise Exception(f"Model atau scaler untuk {ticker} tidak ditemukan!")

    model = joblib.load(model_path)
    print(f"{ticker}: model type = {type(model)}")
    try:
        booster = model.get_booster()
        print(f"{ticker}: booster loaded successfully.")
    except Exception as e:
        raise Exception(f"{ticker}: model belum di-fit! ({str(e)})")
    scaler_X = joblib.load(scalerX_path)
    scaler_y = joblib.load(scalerY_path)
    return model, scaler_X, scaler_y

# Load data
DATA_PATH = 'data/saham_with_lag1.csv'
df = pd.read_csv(DATA_PATH)

st.title("Prediksi Harga Saham")

tickers = ['TINS.JK', 'MDKA.JK', 'ISAT.JK']
input_data = []

with st.form("form_prediksi"):
    st.subheader("Masukkan Data Saham")
    today = st.date_input("Tanggal Hari Ini", value=datetime.date.today())

    for ticker in tickers:
        st.markdown(f"### {ticker}")
        open_price = st.text_input(f"Open {ticker}", key=f"open_{ticker}")
        high_price = st.text_input(f"High {ticker}", key=f"high_{ticker}")
        neg_tweet = st.text_input(f"Negatif Tweet {ticker}", key=f"neg_{ticker}")
        input_data.append((ticker, open_price, high_price, neg_tweet))

    submitted = st.form_submit_button("Prediksi")

if submitted:
    results = []

    for ticker, open_price, high_price, neg_tweet in input_data:
        try:
            # Konversi input
            open_price = float(open_price.replace(",", "."))
            high_price = float(high_price.replace(",", "."))
            neg_tweet = float(neg_tweet.replace(",", "."))


            # Ambil close_lag otomatis
            df_ticker = df[df['Ticker'] == ticker]
            if len(df_ticker) < 5:
                results.append(f"{ticker}: Data tidak cukup.")
                continue
            
            # df_ticker harus diurutkan dulu berdasarkan tanggal
            df_ticker = df_ticker.sort_values("Date")

            close_lag = df_ticker['Close'].iloc[-1]

            open_log = np.log(open_price)
            high_log = np.log(high_price)

            # Ambil 5 baris terakhir untuk input window
            last5 = df_ticker[['negatif_tweet', 'Close_lag1']].tail(5).values
            input_window = [ [open_log, high_log, *row] for row in last5 ]
            input_window.append([open_log, high_log, neg_tweet, close_lag])

            input_array = np.array(input_window)

            model, scaler_X, scaler_y = load_model_and_scalers(ticker)

            input_scaled = scaler_X.transform(input_array)
            #input_reshaped = input_scaled.reshape(1, 6, 5)

            y_scaled_pred = model.predict(input_scaled[-1].reshape(1, -1))
            # Inverse transform dari log-skala ke skala normal
            y_pred_log = scaler_y.inverse_transform(y_scaled_pred.reshape(1, -1))
            y_pred_rupiah = np.exp(y_pred_log)[0][0]  # balikin dari log ke harga asli
            prediksi_harga = round(y_pred_rupiah)


            selisih = prediksi_harga - close_lag
            persentase = (selisih / close_lag) * 100

            if selisih > 0:
                arah = "⬆️ Naik"
            elif selisih < 0:
                arah = "⬇️ Turun"
            else:
                arah = "⏸️ Tidak berubah"

            formatted_price = f"Rp {prediksi_harga:,}".replace(",", ".")
            results.append(f"{ticker}: {formatted_price} ({arah} {abs(persentase):.2f}%)")

        except Exception as e:
            results.append(f"{ticker}: Error - {str(e)}")

    st.subheader("Hasil Prediksi")
    for r in results:
        st.write(r)

    # Grafik perbandingan
    st.subheader("Perbandingan Harga Saham (7 Data Terakhir)")
    colors = {'TINS.JK': 'red', 'MDKA.JK': 'blue', 'ISAT.JK': 'green'}
    merged_df = pd.DataFrame()

    for ticker in tickers:
        df_t = df[df['Ticker'] == ticker].copy()
        df_t['Date'] = pd.to_datetime(df_t['Date'])
        df_t = df_t.sort_values('Date').tail(7)
        df_t = df_t[['Date', 'Close']].rename(columns={'Close': ticker})
        if merged_df.empty:
            merged_df = df_t
        else:
            merged_df = pd.merge(merged_df, df_t, on='Date', how='outer')
            from datetime import timedelta

    # Simpan titik prediksi untuk plot marker
    prediction_points = {}

    for r in results:
        for ticker in tickers:
            if r.startswith(ticker):
                try:
                    pred_value = int(r.split("Rp")[1].split("(")[0].strip().replace(".", ""))
                    last_date = merged_df['Date'].max()
                    next_date = pd.to_datetime(today) + pd.Timedelta(days=1)

                    # Simpan koordinat prediksi
                    prediction_points[ticker] = (next_date, pred_value)

                    # Tambahkan baris prediksi ke data
                    new_row = {col: np.nan for col in merged_df.columns}
                    new_row['Date'] = next_date
                    new_row[ticker] = pred_value
                    merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)
                except:
                    pass
                

    # Tambahkan titik prediksi ke grafik (opsional)
    for ticker, open_price, high_price, neg_tweet in input_data:
        try:
            df_t = df[df['Ticker'] == ticker].copy()
            df_t['Date'] = pd.to_datetime(df_t['Date'])
            last_date = df_t['Date'].max()
            next_date = last_date + pd.Timedelta(days=1)

            # Ambil prediksi harga dari results
            for r in results:
                if r.startswith(ticker):
                    pred_value = int(r.split("Rp")[1].split("(")[0].strip().replace(".", ""))
                    merged_df.loc[len(merged_df.index)] = [next_date if col == 'Date' else (pred_value if col == ticker else np.nan) for col in merged_df.columns]
        except:
            continue

    merged_df = merged_df.sort_values('Date')
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for ticker in tickers:
        ax.plot(merged_df['Date'], merged_df[ticker], label=ticker, color=colors[ticker])
    # Tambahkan marker titik prediksi
    for ticker, (pred_date, pred_value) in prediction_points.items():
        ax.plot(pred_date, pred_value, 'o', markersize=10, color=colors[ticker], label=f'Prediksi {ticker}')

    # Tambahkan anotasi angka prediksi
    for ticker, (pred_date, pred_value) in prediction_points.items():
        ax.text(pred_date, pred_value + 40, f'{pred_value:,}', fontsize=9, ha='center', color=colors[ticker])
        
    # Optional: titik prediksi ditandai juga, tapi tidak perlu warna hitam
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Close (Rp)")
    ax.set_title("Perbandingan Harga Saham")
    ax.legend()
    st.pyplot(fig)
