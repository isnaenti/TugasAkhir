import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =============================
# Inisialisasi alat bantu
# =============================

# Stopword & Stemmer
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# Kamus slang dan istilah saham (pendek)
kamus_slang = {
    'gk': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'bgt': 'sangat', 'yg': 'yang','gua':'saya',
    'tdk': 'tidak', 'dr': 'dari', 'tp': 'tapi', 'skrng': 'sekarang', 'udh': 'sudah','gue':'saya',
    'kmrn': 'kemarin', 'utk': 'untuk', 'gw': 'saya', 'blm': 'belum', 'gue': 'saya','cuma':'hanya',
    'cuan': 'untung', 'goreng': 'spekulatif', 'terbang': 'naik', 'hold': 'tahan', 
    'manteebb':'mantap','ijo':'hijau', 'kmrin': 'kemarin', 'trims':'terima kasih', 
    'pgi':'pagi', 'rebu':'ribu','gede':'besar', 'boncos':'rugi','bgt':'banget',
    'sempet':'sempat', 'pen':'ingin','jg':'juga','lbh':'lebih','hmpr':'hampir',
    'bbrp':'beberapa', 'ap':'apa','klo':'kalau', 'btw': 'ngomong-ngomong',
    'maen': 'main','ampe': 'sampai','nanges': 'nangis','ude': 'sudah','cuan':'uang',
    'kagak': 'tidak','gak': 'tidak','lu': 'kamu','klau': 'kalau','jg': 'juga',
    'drpd': 'daripada','utk': 'untuk','dgn': 'dengan','sm': 'sama','tpi': 'tapi',
    'ndak': 'tidak','udh': 'sudah','lgsg': 'langsung', 'anchoor':'hancur','nanjak':'naik',
    'manteeb':'bagus', 'gokss':'bagus', 'duid':'uang','ambyar':'hancur','gampang':'mudah',
    'ngegas':'melaju','iseh':'masih', 'lempeng':'datar', 'mumet':'pusing','capex':'lelah',
    'naek':'naik', 'kanggo':'untuk', 'nanges':'nangis', 'nutup':'tutup', 'happy':'senang', 
    'egk':'tidak','kompak':'seirama', 'galau':'bingung', 'ngotot':'memaksa', 'trap':'jebakan',
    'shm':'saham', 'drop':'rugi','pengen':'ingin'
}

istilah_saham = {
    'ara': 'naik maksimal',
    'ath': 'harga tertinggi',
    'breakout': 'naik kuat',
    'breakdown': 'turun tajam',
    'hijau': 'naik',
    'greenday': 'semua saham naik',
    'melesat': 'naik cepat',
    'royo-royo': 'semua naik',
    'oversold': 'harga turun terlalu dalam',
    'spinning': 'candle ragu ragu',
    'mengecewakan': 'buruk',
    'rebound': 'berbalik naik',
    'to the moon': 'naik drastis',
    'bangkit':'naik',
    'lonjak':'naik',
    'swing': 'strategi jangka pendek',
    'floating': 'potensi rugi berjalan',
    'overprice': 'harga terlalu mahal',
    'resisten': 'batas atas harga',
    'neckline': 'batas teknikal',
    'hammer': 'pola candle pembalikan',
    'stochastic': 'indikator teknikal',
    'divergence': 'penyimpangan tren',
    'continuation': 'kelanjutan tren',
    'cutloss': 'jual rugi',
    'buyback': 'beli kembali',
    'mantul': 'memantul naik',
    'longsor': 'turun tajam',
    'sukuk': 'obligasi syariah'
}

# =============================
# Fungsi normalisasi
# =============================

def normalize_kata(text):
    for slang, normal in kamus_slang.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', normal, text)
    return text

def expand_istilah(text):
    for istilah, arti in istilah_saham.items():
        text = re.sub(r'\b' + re.escape(istilah) + r'\b', arti, text)
    return text

# =============================
# Fungsi utama preprocessing multi-langkah
# =============================

def preprocess_steps(text):
    step1 = text.lower()
    step2 = re.sub(r"http\S+|www\S+|pic\.twitter\.com/\S+", "", step1)
    step2 = re.sub(r"[^\w\s]", "", step2)
    step2 = re.sub(r"\s+", " ", step2).strip()
    
    step3 = normalize_kata(step2)
    step3 = expand_istilah(step3)
    
    step4 = stopword_remover.remove(step3)
    step5 = stemmer.stem(step4)
    step6 = step5.split()
    
    return {
        "lower": step1,
        "cleaned": step2,
        "normalized": step3,
        "stopword": step4,
        "steemed": step5,
        "tokens": step6
    }
# =============================
# 📥 Membaca Data Tweet
# =============================

# Ganti path sesuai kebutuhanmu
df = pd.read_csv("tweet-tambah/tweetTINS.csv")
# Ambil tanggal dari kolom created_at
df["date"] = pd.to_datetime(df["created_at"]).dt.date

# Hilangkan duplikat
df = df.drop_duplicates(subset="full_text")

# =============================
# 🧪 Jalankan Preprocessing ke Semua Teks
# =============================

preprocessed = df["full_text"].apply(preprocess_steps)
preprocessed_df = pd.DataFrame(preprocessed.tolist())

# =============================
# 💾 Simpan ke CSV
# =============================
output_df = pd.concat(
    [df[['date', 'full_text']].reset_index(drop=True), preprocessed_df.reset_index(drop=True)],
    axis=1
)
output_df.to_csv("hasil-preprocessing/hasiltins.csv", index=False)
print("✅ Preprocessing selesai. Kolom telah disimpan.")
