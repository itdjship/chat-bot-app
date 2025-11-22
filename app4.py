import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from datetime import datetime

# --- Import Library ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import TiDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Load API Key & Database Config
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    if "GOOGLE_API_KEY" in st.secrets:
        # LangChain mencari key di os.environ
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("API Key tidak ditemukan. Pastikan file .env sudah dibuat!")
        st.stop()

# 2. Setup Halaman
st.set_page_config(
    page_title="CyberSec Buddy - Your Security Awareness Partner",
    page_icon="🛡️",
    layout="wide"
)

# Header dengan persona
st.title("🛡️ CyberSec Buddy")
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>👋 Halo! Aku Senior Cybersecurity Expert</h3>
    <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>
        Siap bantu kamu aware soal keamanan digital dengan gaya santai tapi tetap informatif!
        Upload dokumen cybersecurity atau tanya apa aja tentang security awareness. Let's make the internet safer! 🚀
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Setup Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
if "tidb_connected" not in st.session_state:
    st.session_state.tidb_connected = False
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# --- FUNGSI KONEKSI TIDB ---
@st.cache_resource
def init_tidb_connection():
    """
    Inisialisasi koneksi ke Database Vector Store dari environment variables
    """
    try:
        # Baca konfigurasi dari .env
        tidb_host = os.getenv("TIDB_HOST") or st.secrets["TIDB_HOST"]
        tidb_port = os.getenv("TIDB_PORT", "4000") or st.secrets["TIDB_PORT"]
        tidb_user = os.getenv("TIDB_USER") or st.secrets["TIDB_USER"]
        tidb_password = os.getenv("TIDB_PASSWORD") or st.secrets["TIDB_PASSWORD"]
        tidb_database = os.getenv("TIDB_DATABASE", "test") or st.secrets["TIDB_DATABASE"]
        tidb_table = os.getenv("TIDB_TABLE", "rag_documents") or st.secrets["TIDB_TABLE"]
        
        # Validasi kredensial
        if not all([tidb_host, tidb_user, tidb_password]):
            return None, None, "Database Kredensial tidak lengkap di file .env atau secrets"
        
        # Buat connection string
        connection_string = f"mysql+pymysql://{tidb_user}:{tidb_password}@{tidb_host}:{tidb_port}/{tidb_database}?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
        
        # Inisialisasi embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Inisialisasi vector store
        vector_store = TiDBVectorStore(
            connection_string=connection_string,
            embedding_function=embeddings,
            table_name=tidb_table,
            distance_strategy="cosine"
        )
        
        return vector_store, embeddings, None
        
    except Exception as e:
        return None, None, str(e)

# Auto-connect saat aplikasi dimulai
if not st.session_state.tidb_connected and st.session_state.vector_store is None:
    with st.spinner("🔄 Menghubungkan ke Database Vector Store..."):
        vector_store, embeddings, error = init_tidb_connection()
        
        if vector_store and embeddings:
            st.session_state.vector_store = vector_store
            st.session_state.embeddings = embeddings
            st.session_state.tidb_connected = True
            st.session_state.connection_error = None
        else:
            st.session_state.tidb_connected = False
            st.session_state.connection_error = error

# --- FUNGSI PROSES DOKUMEN ---
def process_pdf(uploaded_file, vector_store, embeddings):
    try:
        # a. Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # b. Baca PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path) 

        # c. Pecah teks (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    
            chunk_overlap=200   
        )
        splits = text_splitter.split_documents(docs)
        st.write(f"Memproses {len(splits)} potongan data...")

        # d. Tambahkan metadata untuk tracking
        for i, doc in enumerate(splits):
            doc.metadata["source_file"] = uploaded_file.name
            doc.metadata["chunk_id"] = i
            doc.metadata["upload_time"] = datetime.now().isoformat()

        # e. Simpan ke TiDB Vector Store
        vector_store.add_documents(splits)
        
        # f. Simpan ke upload history
        upload_info = {
            "filename": uploaded_file.name,
            "size": f"{uploaded_file.size / 1024:.2f} KB",
            "chunks": len(splits),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.upload_history.append(upload_info)
        
        return True

    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return False

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔌 Database Status")
    
    # Tampilkan status koneksi
    if st.session_state.tidb_connected:
        st.success("🟢 **Connected to Database Vector Store**")
        
        # Informasi koneksi
        with st.expander("ℹ️ Connection Info"):
            st.write(f"**Host:** {os.getenv('TIDB_HOST', 'N/A') or st.secrets['TIDB_HOST']}")
            st.write(f"**Port:** {os.getenv('TIDB_PORT', '4000') or st.secrets['TIDB_PORT']}")
            st.write(f"**Database:** {os.getenv('TIDB_DATABASE', 'test') or st.secrets['TIDB_DATABASE']}")
            st.write(f"**Table:** {os.getenv('TIDB_TABLE', 'rag_documents') or st.secrets['TIDB_TABLE']}")
            # st.write(f"**User:** {os.getenv('TIDB_USER', 'N/A')}")
    else:
        st.error("🔴 **Not Connected to Database Vector Store**")
        
        if st.session_state.connection_error:
            st.error(f"**Error:** {st.session_state.connection_error}")
        
        st.info("💡 **Cara Mengatasi:**\n1. Pastikan file `.env` sudah dibuat\n2. Isi kredensial Database Vector Store di `.env`\n3. Restart aplikasi")
        
        # Tombol retry
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    st.divider()
    
    # --- UPLOAD DOKUMEN ---
    st.header("📂 Upload Dokumen")
    
    if not st.session_state.tidb_connected:
        st.warning("Hubungkan ke Database Vector Store terlebih dahulu!")
    else:
        uploaded_file = st.file_uploader("Upload file PDF kamu", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Proses Dokumen"):
                with st.spinner("Memproses dan upload..."):
                    success = process_pdf(
                        uploaded_file, 
                        st.session_state.vector_store,
                        st.session_state.embeddings
                    )
                    if success:
                        st.success("✅ Dokumen berhasil disimpan!")
    
    st.divider()
    
    # --- UPLOAD HISTORY ---
    st.header("📜 History Upload")
    
    if len(st.session_state.upload_history) == 0:
        st.info("Belum ada file yang diupload")
    else:
        st.write(f"**Total file diproses:** {len(st.session_state.upload_history)}")
        
        for idx, file_info in enumerate(reversed(st.session_state.upload_history), 1):
            with st.expander(f"📄 {file_info['filename']}", expanded=(idx==1)):
                st.write(f"**Ukuran:** {file_info['size']}")
                st.write(f"**Chunks:** {file_info['chunks']}")
                st.write(f"**Waktu:** {file_info['timestamp']}")
        
        # Tombol clear history
        if st.button("🗑️ Hapus History", type="secondary"):
            st.session_state.upload_history = []
            st.rerun()

# --- AREA CHAT ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_query = st.chat_input("💬 Tanya apa aja tentang cybersecurity... aku siap jawab!")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    if not st.session_state.tidb_connected:
        with st.chat_message("assistant"):
            st.write("Eh, database belum connect nih! Upload dokumen dulu ya di sidebar sebelah kiri 👈 Biar aku bisa bantu kamu dengan info yang akurat!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Lagi nyari info terbaik buat kamu..."):
                try:
                    # LLM menggunakan Gemini dengan temperature lebih tinggi untuk gaya santai
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

                    prompt = ChatPromptTemplate.from_template("""
                    Kamu adalah Senior Cybersecurity Expert yang berpengalaman puluhan tahun di bidang keamanan siber.
                    Kamu punya gaya komunikasi yang santai, friendly, dan mudah dipahami, tapi tetap profesional dan informatif.
                    
                    PERSONALITY & STYLE:
                    - Gunakan bahasa Indonesia yang santai tapi tetap sopan (seperti ngobrol sama teman)
                    - Sesekali pakai kata "aku/kamu" untuk kesan lebih personal
                    - Tambahkan emoji yang relevan untuk membuat penjelasan lebih menarik
                    - Berikan analogi atau contoh real-world yang mudah dipahami
                    - Selalu tekankan pentingnya security awareness
                    - Jika ada ancaman/risiko, jelaskan dengan cara yang tidak menakut-nakuti tapi tetap serius
                    
                    RESPONSE STRUCTURE:
                    1. Mulai dengan greeting singkat atau acknowledgment pertanyaan
                    2. Berikan jawaban utama yang jelas dan to-the-point
                    3. Tambahkan tips praktis atau best practices jika relevan
                    4. Akhiri dengan motivasi atau reminder tentang pentingnya security awareness
                    
                    IMPORTANT RULES:
                    - HANYA jawab berdasarkan konteks dokumen yang diberikan
                    - Jika info tidak ada di dokumen, bilang dengan jujur: "Nah, untuk yang ini aku belum punya info lengkap di dokumen yang kamu upload. Tapi secara umum..."
                    - Jangan mengarang atau membuat informasi palsu
                    - Selalu prioritaskan keakuratan informasi
                    
                    CONTEXT dari dokumen:
                    <context>
                    {context}
                    </context>
                    
                    PERTANYAAN USER: {input}
                    
                    Jawab dengan gaya kamu yang khas sebagai Senior Cybersecurity Expert yang friendly tapi tetap expert!
                    """)

                    # Retriever dari TiDB
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 5}
                    )
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({"input": user_query})
                    answer = response['answer']
                    
                    st.write(answer)
                    st.session_state.chat_history.append(("assistant", answer))
                    
                    # Tampilkan sumber dokumen dengan style yang lebih menarik
                    with st.expander("📚 Referensi Dokumen (Dari mana aku dapet info ini)"):
                        st.markdown("*Info di atas aku ambil dari dokumen-dokumen ini:*")
                        sources = set()
                        for doc in response.get('context', []):
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            if source_file not in sources:
                                st.write(f"✅ **{source_file}**")
                                sources.add(source_file)
                            
                except Exception as e:
                    st.error(f"❌ Oops, ada error nih: {e}")
                    st.write("Kayaknya belum ada dokumen yang kamu upload deh. Coba upload dokumen cybersecurity dulu ya di sidebar! 📄")