# 🚀 RAG Chatbot dengan TiDB Vector Database

Dokumentasi untuk **app4.py** - Implementasi RAG Chatbot menggunakan TiDB sebagai vector database dengan HuggingFace embeddings lokal dan Gemini LLM.

---

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Arsitektur](#-arsitektur)
- [Keunggulan TiDB Vector](#-keunggulan-tidb-vector)
- [Instalasi](#-instalasi)
- [Konfigurasi TiDB](#-konfigurasi-tidb)
- [Cara Menjalankan](#-cara-menjalankan)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Fitur

### app4.py (TiDB Vector + HuggingFace + Gemini)
- ✅ **Vector Database Persistent** menggunakan TiDB Cloud
- ✅ **Embedding Lokal** dengan HuggingFace (`all-MiniLM-L6-v2`)
- ✅ **LLM Cloud** menggunakan Gemini 2.0 Flash
- ✅ **Upload History** dengan tracking lengkap
- ✅ **Metadata Tracking** untuk setiap dokumen
- ✅ **Scalable** - Data tersimpan di cloud database
- ✅ **Multi-session** - Data tidak hilang saat restart aplikasi
- ✅ **Source Attribution** - Menampilkan sumber dokumen dalam jawaban

---

## 🏗️ Arsitektur

### Alur Kerja RAG dengan TiDB

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
│   Embedding     │  ← HuggingFace (Lokal)
│  (all-MiniLM)   │     all-MiniLM-L6-v2
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TiDB Vector    │  ← TiDB Cloud Vector Store
│   Store (SQL)   │     Persistent Storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  ← Cosine Similarity Search (k=5)
│  from TiDB      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Answer    │  ← Gemini 2.0 Flash
│  (Gemini API)   │
└─────────────────┘
```

---

## 🌟 Keunggulan TiDB Vector

### Dibanding ChromaDB (In-Memory)

| Aspek | ChromaDB (app1-3) | TiDB Vector (app4) |
|-------|-------------------|-------------------|
| **Persistence** | In-memory (hilang saat restart) | Persistent (tersimpan di database) |
| **Scalability** | Terbatas memory lokal | Unlimited (cloud database) |
| **Multi-user** | Single session | Multi-session support |
| **Backup** | Manual | Automatic (database backup) |
| **Query Speed** | Sangat cepat (RAM) | Cepat (optimized SQL) |
| **Cost** | Gratis | Gratis (TiDB Serverless tier) |
| **Production Ready** | Development only | Production ready |

### Fitur TiDB Vector
- ✅ **Vector Search** dengan cosine similarity
- ✅ **SQL Compatible** - Bisa query dengan SQL biasa
- ✅ **ACID Transactions** - Data consistency terjamin
- ✅ **Horizontal Scaling** - Auto-scale sesuai kebutuhan
- ✅ **Built-in Backup** - Point-in-time recovery
- ✅ **Monitoring Dashboard** - Real-time metrics

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

**Dependencies tambahan untuk TiDB:**
- `tidb-vector` - TiDB Vector Store connector
- `pymysql` - MySQL driver untuk koneksi TiDB
- `cryptography` - Untuk SSL connection

---

## ⚙️ Konfigurasi TiDB

### 1. Buat TiDB Cluster (Gratis)

1. Kunjungi [TiDB Cloud](https://tidbcloud.com/)
2. Sign up / Login
3. Buat **Serverless Cluster** (Gratis)
4. Tunggu cluster aktif (~2 menit)

### 2. Dapatkan Connection String

Di TiDB Cloud Dashboard:
1. Klik cluster Anda
2. Pilih **Connect**
3. Copy informasi berikut:
   - Host: `gateway01.ap-southeast-1.prod.aws.tidbcloud.com`
   - Port: `4000`
   - Username: `<your-username>`
   - Password: `<your-password>`
   - Database: `test` (default)

### 3. Setup Environment Variables

Edit file `.env`:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# TiDB Configuration
TIDB_HOST=gateway01.ap-southeast-1.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your_tidb_username
TIDB_PASSWORD=your_tidb_password
TIDB_DATABASE=test
```

### 4. Verifikasi Koneksi (Optional)

Test koneksi dengan MySQL client:

```bash
mysql -h gateway01.ap-southeast-1.prod.aws.tidbcloud.com -P 4000 -u <username> -p<password> -D test --ssl-mode=VERIFY_IDENTITY
```

---

## 🎮 Cara Menjalankan

### 1. Jalankan Aplikasi

```bash
streamlit run app4.py
```

Aplikasi akan berjalan di: `http://localhost:8501`

### 2. Cara Menggunakan

#### Step 1: Koneksi ke TiDB
1. Isi form konfigurasi TiDB di sidebar:
   - Host
   - Port
   - Username
   - Password
   - Database
   - Table Name (default: `rag_documents`)
2. Klik tombol **"🔗 Connect to TiDB"**
3. Tunggu hingga status berubah menjadi **"🟢 TiDB Connected"**

#### Step 2: Upload Dokumen
1. Klik **"Browse files"** di section Upload Dokumen
2. Pilih file PDF
3. Klik **"Proses Dokumen"**
4. Tunggu proses embedding dan upload ke TiDB
5. Dokumen akan muncul di **History Upload**

#### Step 3: Tanya Jawab
1. Ketik pertanyaan di chat input
2. AI akan mencari jawaban dari dokumen di TiDB
3. Jawaban akan ditampilkan beserta sumber dokumennya
4. Klik **"📚 Sumber Dokumen"** untuk melihat detail

---

## 🔧 Troubleshooting

### Error: `Connection refused` atau `Can't connect to TiDB`

**Penyebab:**
- Kredensial salah
- Cluster belum aktif
- Firewall blocking

**Solusi:**
1. Verifikasi username dan password di TiDB Cloud
2. Pastikan cluster status **"Available"**
3. Cek koneksi internet
4. Pastikan SSL certificate tersedia

### Error: `SSL connection error`

**Solusi:**
```bash
# Download CA certificate
curl --create-dirs -o /etc/ssl/cert.pem https://letsencrypt.org/certs/isrgrootx1.pem
```

Atau ubah connection string di app4.py:
```python
connection_string = f"mysql+pymysql://{tidb_user}:{tidb_password}@{tidb_host}:{tidb_port}/{tidb_database}?ssl_mode=DISABLED"
```

### Error: `Table already exists`

**Normal:** Tabel akan dibuat otomatis pertama kali. Jika sudah ada, akan menggunakan tabel yang sama.

**Reset tabel:**
```sql
-- Login ke TiDB
DROP TABLE rag_documents;
```

### Model Download Lambat (Pertama Kali)

**Normal:** Model `all-MiniLM-L6-v2` (~80MB) akan didownload otomatis pertama kali.

**Lokasi Cache:**
- Windows: `C:\Users\<username>\.cache\huggingface\hub`
- Linux/Mac: `~/.cache/huggingface/hub`

### Data Tidak Muncul Setelah Upload

**Solusi:**
1. Cek koneksi TiDB masih aktif (status hijau)
2. Refresh halaman
3. Cek di TiDB Cloud Console apakah data masuk:
   ```sql
   SELECT COUNT(*) FROM rag_documents;
   ```

---

## 📊 Perbandingan Semua Versi

| Fitur | app1.py | app2.py | app3.py | app4.py |
|-------|---------|---------|---------|---------|
| **Embedding** | Google API | HuggingFace | HuggingFace | HuggingFace |
| **Vector DB** | ChromaDB | ChromaDB | ChromaDB | TiDB Vector |
| **Persistence** | ❌ | ❌ | ❌ | ✅ |
| **Upload History** | ❌ | ❌ | ✅ | ✅ |
| **API Limit** | ⚠️ Ada | ✅ Tidak | ✅ Tidak | ✅ Tidak |
| **Production Ready** | ❌ | ❌ | ❌ | ✅ |
| **Multi-session** | ❌ | ❌ | ❌ | ✅ |
| **Scalability** | Low | Low | Low | High |

---

## 📝 Struktur Database TiDB

### Tabel: `rag_documents`

```sql
CREATE TABLE rag_documents (
    id VARCHAR(255) PRIMARY KEY,
    document TEXT,
    embedding VECTOR(384),  -- Dimensi sesuai model all-MiniLM-L6-v2
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index untuk vector search
CREATE INDEX idx_embedding ON rag_documents (embedding);
```

### Metadata Structure

```json
{
    "source_file": "document.pdf",
    "chunk_id": 0,
    "upload_time": "2024-01-01T12:00:00",
    "page": 1
}
```

---

## 🔐 Keamanan

⚠️ **PENTING:**
- Jangan commit file `.env` ke repository
- Gunakan `.env.example` sebagai template
- Jangan share credentials TiDB di public
- Aktifkan IP whitelist di TiDB Cloud (production)
- Gunakan SSL connection (sudah default)

---

## 📚 Teknologi yang Digunakan

- **Streamlit** - Web framework
- **LangChain** - RAG framework
- **Google Gemini** - Large Language Model
- **HuggingFace** - Embedding models (lokal)
- **TiDB Cloud** - Vector database (MySQL-compatible)
- **PyPDF** - PDF processing
- **Sentence Transformers** - Embedding models

---

## 🎯 Use Cases

### Kapan Menggunakan app4.py?

✅ **Production Environment**
- Aplikasi yang digunakan banyak user
- Data perlu persistent
- Butuh backup dan recovery

✅ **Large Scale**
- Banyak dokumen (>100 files)
- Dokumen besar (>10MB)
- Butuh scalability

✅ **Multi-session**
- User bisa logout dan data tetap ada
- Sharing knowledge base antar user
- Team collaboration

❌ **Jangan Gunakan Jika:**
- Hanya testing/development
- Data sensitif yang tidak boleh di cloud
- Tidak ada koneksi internet

---

## 🤝 Kontribusi

Silakan buat issue atau pull request untuk improvement!

---

## 📄 Lisensi

MIT License

---

## 📞 Support

- TiDB Documentation: https://docs.pingcap.com/tidb/stable
- TiDB Cloud: https://tidbcloud.com/
- LangChain TiDB: https://python.langchain.com/docs/integrations/vectorstores/tidb_vector

---

**Happy Coding! 🚀**