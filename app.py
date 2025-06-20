import streamlit as st
import joblib
import re
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Muat model dan vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Setup stemmer dan stopwords
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())
stopwords.update({'saya', 'kami', 'aku', 'kita', 'dia', 'anda', 'dan','aplikasi'})
stopword_tidak_hapus = {'tidak', 'bisa'}

# Kamus normalisasi
kamus_normalisasi = {
    'gk': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak', 'ngga': 'tidak',
    'udah': 'sudah', 'udh': 'sudah', 'blm': 'belum', 'bgt': 'banget', 'bnyk': 'banyak',
    'lwt': 'lewat', 'dgn': 'dengan', 'sm': 'sama', 'aja': 'saja', 'bpj': 'bpjs',
    'ap': 'apa', 'tp': 'tapi', 'dr': 'dari', 'kalo': 'kalau', 'bnget': 'banget',
    'no': 'nomor', 'sangaaatttt': 'sangat', 'lamaaaaaa': 'lama', 'burukkk': 'buruk',
    'bisaa': 'bisa', 'app': 'aplikasi', 'jgn': 'jangan', 'smpe': 'sampai', 'ngk': 'tidak',
    'bkn': 'bukan', 'n': 'dan', 'applikasi': 'aplikasi', 'telf': 'telepon', 'kluar': 'keluar',
    'tak': 'tidak', 'dg': 'dengan', 'apk': 'aplikasi', 'dpt': 'dapat', 'kenaap': 'kenapa',
    'downlod': 'download', 'bs': 'bisa', 'kl': 'kalau', 'dng': 'dengan', 'klo': 'kalau',
    'gmn': 'gimana', 'yg': 'yang'
}

# Fungsi preprocessing
def preprocess(text):
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stopwords or word in stopword_tidak_hapus])
    tokens = text.split()
    tokens = [kamus_normalisasi.get(word, word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# UI Streamlit
st.title("📊 Analisis Sentimen Review Aplikasi 🇮🇩")
st.markdown("Masukkan review aplikasi dan dapatkan **prediksi sentimen otomatis**.")

user_input = st.text_area("📝 Masukkan teks review:", "")

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        preprocessed = preprocess(user_input)
        vectorized = vectorizer.transform([preprocessed])
        prediction = model.predict(vectorized)[0]

        if prediction == "Positif":
            st.success(f"✅ Prediksi Sentimen: **{prediction}**")
        elif prediction == "Negatif":
            st.error(f"❌ Prediksi Sentimen: **{prediction}**")
        else:
            st.info(f"ℹ️ Prediksi Sentimen: **{prediction}**")
