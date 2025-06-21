import streamlit as st
import joblib
import re
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# === Load Model & Vectorizer ===
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# === Setup Stemmer & Stopwords ===
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())
stopwords.update({'saya', 'kami', 'aku', 'kita', 'dia', 'anda', 'dan', 'aplikasi'})
stopword_tidak_hapus = {'tidak', 'bisa'}

# === Kamus Normalisasi ===
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

# === Daftar Kata Kunci ===
positif_words = {
    'bagus', 'baik', 'cepat', 'memudahkan', 'membantu', 'puas', 'mantap', 'nyaman',
    'semoga', 'mudah', 'ok', 'terima kasih', 'enak', 'efisien', 'informasi',
    'ramah', 'berfungsi', 'bermanfaat', 'profesional', 'akurat', 'menyenangkan',
    'berharga', 'menyelesaikan', 'hebat', 'canggih', 'solid', 'keren', 'sukses',
    'aman', 'terjangkau', 'alhamdulillah', 'terbantu', 'top'
}

negatif_words = {
    'susah', 'error', 'gagal', 'lama', 'buruk', 'mengecewakan', 'kenapa',
    'ribet', 'macet', 'lambat', 'sulit', 'antri', 'kacau', 'masalah',
    'kecewa', 'menghambat', 'nyebelin', 'berat', 'mengganggu', 'terhambat',
    'jelek', 'sampah', 'lemot', 'menyusahkan', 'bohong', 'parah', 'zonk',
    'rusak', 'hilang', 'hang', 'ngecrash', 'menyesal', 'menipu', 'trouble',
    'emosi', 'dicopot', 'dihapus'
}

kata_negasi = {'tidak', 'tak', 'gak', 'ga', 'gk', 'belum', 'kurang'}
whitelist_kata_positif = positif_words

# === Preprocessing Function ===
def preprocess(text):
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [kamus_normalisasi.get(word, word) for word in tokens]
    filtered_tokens = [word for word in tokens if (word not in stopwords or word in stopword_tidak_hapus or word in whitelist_kata_positif)]
    tokens_final = [word if word in whitelist_kata_positif else stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(tokens_final), tokens_final

# === Highlight Function ===
def highlight_tokens(tokens):
    highlighted = []
    for token in tokens:
        if token in positif_words:
            highlighted.append(f"<span style='color:green;font-weight:bold'>{token}</span>")
        elif token in negatif_words:
            highlighted.append(f"<span style='color:red;font-weight:bold'>{token}</span>")
        elif token in kata_negasi:
            highlighted.append(f"<span style='color:orange;font-weight:bold'>{token}</span>")
        else:
            highlighted.append(token)
    return ' '.join(highlighted)

# === Streamlit UI ===
st.title("🔍 Analisis Sentimen Aplikasi PORSI - Siti Hajar")
st.markdown("Masukkan review aplikasi, dan dapatkan **prediksi sentimen otomatis** berdasarkan model Naive Bayes.")

user_input = st.text_area("📝 Masukkan review Anda:", "")

if st.button("🚀 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        clean_text, tokens = preprocess(user_input)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]

        if prediction == "Positif":
            st.success(f"✅ Prediksi Sentimen: **{prediction}**")
        elif prediction == "Negatif":
            st.error(f"❌ Prediksi Sentimen: **{prediction}**")
        else:
            st.info(f"ℹ️ Prediksi Sentimen: **{prediction}**")

        # Tampilkan hasil highlight
        highlighted_text = highlight_tokens(tokens)
        st.markdown("### 🧠 Highlight Kata-Kata Terkait:")
        st.markdown(highlighted_text, unsafe_allow_html=True)

