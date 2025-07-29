import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from tqdm import tqdm

tqdm.pandas()  # agar progress bar muncul saat apply

# 1. Load Dataset (pastikan kolom 'text' sudah ada)
df = pd.read_csv("hasil-preprocessing/hasiltins.csv")

# 2. Terjemahkan teks ke Bahasa Inggris
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return ""

df['text_en'] = df['tokens'].astype(str).progress_apply(translate_text)

# 3. Inisialisasi VADER
analyzer = SentimentIntensityAnalyzer()

# 4. Lexicon Tambahan (opsional, bisa di-skip jika sudah translate)
custom_lexicon = {
    'nice': 1.5,'good': 1.5,'hold': 0.5,'sell': -0.7,'rise': 1.6,'maximum': 1.0,
    'drop': -1.6,'drastic': -1.0,'buy': 1.2,'recommended': 1.5,'shaking head': -0.8,
    'dribble': -0.4,'great': 1.5,'cool': 1.4,'average': 0.0,'down': -1.5,
    'stuck': -1.2,'avoid': -0.8,'better': 1.0,'decent': 0.6,'donation': -0.6,
    'update': 0.1,'no': -0.5,'risk': -1.0,'out': -0.5,'a lot': 0.3,'fall': -1.5,
    'buy a lot': 1.2,'slump': -1.7,'ma': 0.0,'return': 0.4,'highest': 1.2,
    'heaven': 1.8,'inflation': -1.5,'destroyed': -2.0,'signal': 0.2,'against': -0.4,
    'cut': -0.5,'loss': -1.8,'stop loss': -1.2,'potential': 0.7,'big': 0.4,'negative': -1.5,
    'profit': 1.7,'grow': 1.4,'slow': -0.8,'throw down': -1.3,'earnings': 1.3,
    'net': 0.5,'term': 0.0,'long': 0.1,'strong': 1.3,'sharp': 0.8,'fail': -1.5,
    'lack': -1.0,'target': 0.5,'money': 0.2,'rocket': 1.8,'don’t': -0.6,
    'lose': -1.4,'release': -0.2,'enter': 0.2,'junk stock': -1.6,'pace': 0.4,
    'let’s go': 0.6,'issue': -0.3,'happy': 1.4,'victorious': 1.5,'fortune': 1.2,
    'trust': 1.0,'uptrend': 1.8,'weak': -1.3,'catch up': 0.5,'no risk': 0.3,
    'bottom': -1.0,'expectation': 0.3,'high': 1.2,'level': 0.1,'opportunity': 1.0,
    'explode': 1.8,'dividend': 1.3,'fry': -1.4,'break': -0.8,'mess': -1.5,'grateful': 1.5,
    'floating': 0.1,'overbought': -1.2,'consistent': 1.2,'breakthrough': 1.0,
    'pullback': -0.5,'negative': -1.5,'current': 0.0,'positive': 1.5,'red': -0.8,'fold': -1.0,
    'multiple': 0.4,'crossover': 0.7,'bearish': -1.8,'marubozu': 0.2,
    'bullish': 1.8,'bad': -1.5,'gain': 1.4,'capital': 0.6,'strategic': 1.0,
    'liquid': 0.5,'fast': 0.6,'strong push': 1.3,'weak': -1.3,'line': 0.0,'blue': 0.2,
    'turn': 0.3,'blessing': 1.5,'treasure': 1.4,'thin': -0.4,'auto': 0.1,
    'rejected': -1.0,'recovered': 1.3,'lazy': -1.2,'growth': 1.5,'buythedip': 1.6,
    'stable': 1.0, 'breakthrough': 1.0,'support': 0.6,'pick': 0.3,'up': 0.4,
    'interesting': 1.2,'awesome': 1.6,'movement': 0.5,'fun': 1.2,'flat': -0.5,
    'blow': -0.3,'thank you': 0.8,'jump': 1.3,'win': 1.5,'breakout': 1.8,
    'broker': 0.2,'limit': 0.0,'patience': 0.4,'take': 0.2,'luck': 1.0,
    'reforestation': 0.0, 'lob': 0.2,'snack': 0.4,'congrats': 1.4,'happy': 1.4,
    'keep': 0.3,'stop': -0.8,'enough': 0.0,'greedy': -1.4,'maybe': 0.0,
    'necklineresistance': -0.3,'broken': -1.2,'plunge': -1.6,'admire': 1.3,
    'mercy': -0.6,'already': 0.0,'retest': 0.3,'touch': 0.2,'record': 1.3,
    'decline': -1.5,'entry': 0.5,'haka': -0.3,'wow': 1.0,'expose': -0.5,
    'dive': -1.2,'again': 0.0,'smooth': 0.8,'sting': -0.6,'mid-term': 0.0,
    'medium-term': 0.0,'short-term': 0.0,'little': -0.3,'reflection': 0.4,
    'fly': 1.6,'good job': 1.8 
}

analyzer.lexicon.update(custom_lexicon)

# 5. Fungsi Sentimen VADER
def get_vader_label(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 2  # Positif
    elif compound <= -0.05:
        return 1  # Negatif
    else:
        return 0  # Netral

# 6. Proses Sentimen dari Teks yang Sudah Diterjemahkan
df['vader_compound'] = df['text_en'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['vader_label'] = df['text_en'].apply(get_vader_label)

# 7. Simpan ke File
df.to_csv("hasilsentvader/perbaikan/set-tins.csv", index=False)
print("✅ Selesai! File disimpan sebagai 'hasil_sentimen_vader_translate.csv'")
