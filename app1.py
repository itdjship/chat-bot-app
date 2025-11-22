import time
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Library untuk RAG
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Key
load_dotenv()
# Pastikan GOOGLE_API_KEY terbaca
if "GOOGLE_API_KEY" not in os.environ:
    st.error("API Key tidak ditemukan. Pastikan file .env sudah dibuat!")

# 2. Setup Halaman
st.set_page_config(page_title="RAG Chatbot Gemini", layout="wide")
st.title("🤖 Chatbot RAG: Tanya Jawab dengan PDF")

# 3. Setup Session State (Agar chat history tidak hilang saat reload)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- FUNGSI PROSES DOKUMEN ---
def process_pdf(uploaded_file):
    """
    Fungsi untuk mengubah PDF menjadi Vector Database
    """
    try:
        # a. Simpan file sementara (karena Loader butuh path file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # b. Baca PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path) # Hapus file temporary

        # c. Pecah teks menjadi potongan kecil (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,    # Ukuran per potongan
            chunk_overlap=2   # Overlap agar konteks tidak putus
        )
        splits = text_splitter.split_documents(docs)

        st.write(f"Jumlah potongan teks (chunks): {len(splits)}")

        # d. Buat Embedding & Simpan ke Vector Store (Chroma)
        # Menggunakan model embedding khusus Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            # persist_directory="./chroma_db" # Opsional: jika ingin disimpan ke disk
        )

        batch_size = 1
        total_batches = len(splits) // batch_size + 1
        
        progress_bar = st.progress(0)
        for i in range(0, len(splits), batch_size):
            # Ambil potongan data (slicing)
            batch = splits[i:i + batch_size]
            
            # Masukkan ke Chroma
            if batch: # Pastikan batch tidak kosong
                vectorstore.add_documents(batch)
            
            # Update progress bar
            current_progress = min((i + batch_size) / len(splits), 1.0)
            progress_bar.progress(current_progress)
            
            # Tunda 2 detik agar tidak kena limit RPM (PENTING!)
            time.sleep(2)

        return vectorstore

    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return None

# --- SIDEBAR: UPLOAD FILE ---
with st.sidebar:
    st.header("📂 Upload Dokumen")
    uploaded_file = st.file_uploader("Upload file PDF kamu", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Proses Dokumen"):
            with st.spinner("Sedang membaca & menyusun data..."):
                # Simpan vector db ke session state
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("Dokumen berhasil diproses! Silakan bertanya.")

# --- AREA CHAT UTAMA ---

# Tampilkan chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Input User
user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...")

if user_query:
    # 1. Tampilkan pertanyaan user
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Cek apakah dokumen sudah diproses
    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.write("Mohon upload dan proses dokumen PDF terlebih dahulu di sidebar sebelah kiri ya. 👈")
    else:
        # 3. Proses Jawaban dengan RAG
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis dokumen..."):
                # Setup Model Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

                # Setup Prompt RAG (Instruksi khusus)
                prompt = ChatPromptTemplate.from_template("""
                Jawablah pertanyaan berikut HANYA berdasarkan konteks yang diberikan.
                Jika jawabannya tidak ada di dalam konteks, katakan "Maaf, informasi tersebut tidak ditemukan dalam dokumen."
                Jangan mengarang jawaban.

                <context>
                {context}
                </context>

                Pertanyaan: {input}
                """)

                # Setup Retriever (Pencari data di vector db)
                retriever = st.session_state.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                # Gabungkan Dokumen -> Prompt -> LLM
                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Eksekusi
                response = retrieval_chain.invoke({"input": user_query})
                answer = response['answer']
                
                st.write(answer)
                
                # Simpan jawaban ke history
                st.session_state.chat_history.append(("assistant", answer))