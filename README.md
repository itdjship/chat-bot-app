# 🤖 RAG Chatbot dengan Gemini & HuggingFace

Project ini berisi dua implementasi chatbot RAG (Retrieval-Augmented Generation) untuk tanya jawab dengan dokumen PDF:

1. **app1.py** - RAG dengan Google Gemini Embeddings (Sering kena error 429 Rate/Limit)
2. **app2.py** - Hybrid RAG dengan HuggingFace Embeddings (Lokal) + Gemini LLM

---

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Arsitektur](#-arsitektur)
- [Instalasi](#-instalasi)
- [Konfigurasi](#-konfigurasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Troubleshooting](#-troubleshooting)
- [Perbedaan app.py vs app2.py](#-perbedaan-apppy-vs-app2py)

---

## ✨ Fitur

### app1.py (Google Gemini Full)
- ✅ Embedding menggunakan Google Generative AI (`models/embedding-001`)
- ✅ LLM menggunakan Gemini 1.5 Flash
- ✅ Upload dan proses dokumen PDF
- ✅ Chat interface dengan history
- ✅ Vector database menggunakan ChromaDB

### app2.py (Hybrid: Lokal + Cloud)
- ✅ **Embedding lokal** menggunakan HuggingFace (`all-MiniLM-L6-v2`)
- ✅ **LLM cloud** menggunakan Gemini 2.0 Flash
- ✅ **Tanpa batasan API** untuk embedding (tidak ada error 429)
- ✅ Model embedding didownload otomatis (~80MB, sekali saja)
- ✅ Proses embedding lebih cepat karena lokal
- ✅ Chat interface dengan history

---

## 🏗️ Arsitektur

### Alur Kerja RAG

```
┌─────────────────┐
│   Upload PDF    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Split   │  ← PyPDFLoader + RecursiveCharacterTextSplitter
│   (Chunks)      │     (chunk_size=1000, overlap=200)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  ← app1.py: Google Gemini
│                 │     app2.py: HuggingFace (Lokal)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  ← ChromaDB (In-Memory)
│   (ChromaDB)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  ← Similarity Search (k=5)
│  (Top 5 Docs)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Answer    │  ← Gemini 1.5/2.0 Flash
│  (Gemini API)   │
└─────────────────┘
```

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd chat-bot-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies yang diinstall:**
- `streamlit` - UI framework
- `langchain`, `langchain-core`, `langchain-community`, `langchain-classic` - RAG framework
- `langchain-google-genai` - Integrasi Gemini
- `chromadb` - Vector database
- `pypdf` - PDF loader
- `sentence-transformers` - HuggingFace embeddings (untuk app2.py)
- `python-dotenv` - Environment variables

---

## ⚙️ Konfigurasi

### 1. Buat File `.env`

Rename file `.env.example` menjadi `.env` 
atau
Buat file `.env` di root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Dapatkan Google API Key

1. Kunjungi [Google AI Studio](https://aistudio.google.com/app/api-keys)
2. Buat API key baru
3. Copy dan paste ke file `.env`

---

## 🎮 Cara Menjalankan

### Menjalankan app.py (Google Gemini Full)

```bash
streamlit run app1.py
```

Aplikasi akan berjalan di: `http://localhost:8501`

### Menjalankan app2.py (Hybrid RAG)

```bash
streamlit run app2.py
```

Aplikasi akan berjalan di: `http://localhost:8502` (atau port lain jika 8501 sudah digunakan)

### Cara Menggunakan

1. **Upload PDF**: Klik tombol "Browse files" di sidebar
2. **Proses Dokumen**: Klik tombol "Proses Dokumen"
3. **Tunggu**: Aplikasi akan memproses PDF (pertama kali agak lama untuk download model di app2.py)
4. **Tanya**: Ketik pertanyaan di chat input
5. **Dapatkan Jawaban**: AI akan menjawab berdasarkan konten PDF

---

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'langchain.chains'`

**Solusi:** Sudah diperbaiki dengan menggunakan import dari `langchain_classic`:

```python
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```

### Error: `API Key tidak ditemukan`

**Solusi:**
1. Pastikan file `.env` ada di root directory
2. Pastikan format: `GOOGLE_API_KEY=your_key_here` (tanpa spasi atau quotes)
3. Restart aplikasi setelah membuat `.env`

### Error: `429 Too Many Requests` (app1.py)

**Solusi:** Gunakan `app2.py` yang menggunakan embedding lokal, sehingga tidak ada batasan API.

### Model Download Lambat (app2.py - Pertama Kali)

**Normal:** Model `all-MiniLM-L6-v2` (~80MB) akan didownload otomatis pertama kali. Setelah itu akan menggunakan cache lokal.

**Lokasi Cache:**
- Windows: `C:\Users\<username>\.cache\huggingface\hub`
- Linux/Mac: `~/.cache/huggingface/hub`

---

## 📊 Perbedaan app.py vs app2.py

| Aspek | app.py | app2.py |
|-------|--------|---------|
| **Embedding** | Google Gemini API | HuggingFace (Lokal) |
| **LLM** | Gemini 1.5 Flash | Gemini 2.0 Flash |
| **API Limit** | Ada (429 error possible) | Tidak ada (embedding lokal) |
| **Kecepatan Embedding** | Tergantung internet | Cepat (lokal) |
| **Setup Awal** | Cepat | Agak lama (download model) |
| **Ukuran Model** | - | ~80MB (sekali download) |
| **Biaya** | Gratis (quota API) | Gratis (LLM saja pakai API) |
| **Offline Capability** | Tidak | Partial (embedding bisa offline) |

### Kapan Menggunakan app.py?
- ✅ Koneksi internet stabil
- ✅ Tidak banyak dokumen yang diproses
- ✅ Ingin setup cepat tanpa download model

### Kapan Menggunakan app2.py?
- ✅ Banyak dokumen yang perlu diproses
- ✅ Ingin menghindari API limit
- ✅ Koneksi internet lambat (setelah model terdownload)
- ✅ Butuh kecepatan embedding yang konsisten

---

## 📝 Struktur File

```
chat-bot-app/
├── .env                   # Tempat simpan credentials (API keys) jangan sampai ke commit
├── app1.py                # RAG dengan Google Gemini Full
├── app2.py                # Hybrid RAG (HuggingFace + Gemini)
├── requirements.txt       # Python dependencies
└── README.md              # Dokumentasi ini
```

---

## 🔐 Keamanan

⚠️ **PENTING:**
- Jangan commit file `.env` ke repository
- Tambahkan `.env` ke `.gitignore`
- Jangan share API key di public

---

## 📚 Teknologi yang Digunakan

- **Streamlit** - Web framework untuk Python
- **LangChain** - Framework untuk aplikasi LLM
- **Google Gemini** - Large Language Model
- **HuggingFace** - Model embeddings lokal
- **ChromaDB** - Vector database
- **PyPDF** - PDF processing
- **Sentence Transformers** - Embedding models

---

## 🤝 Kontribusi

Silakan buat issue atau pull request untuk improvement!

---

## 📄 Lisensi

MIT License

---