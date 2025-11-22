import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Import Library ---
from langchain_google_genai import ChatGoogleGenerativeAI
# [MODIFIKASI 1] Ganti Embedding Google jadi HuggingFace (Lokal)
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Menggunakan import dari langchain_classic (LangChain 1.0+)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("API Key tidak ditemukan. Pastikan file .env sudah dibuat!")

# 2. Setup Halaman
st.set_page_config(page_title="RAG Hybrid (Local + Gemini)", layout="wide")
st.title("🤖 Chatbot RAG: Embedding Lokal (Anti-Limit)")

# 3. Setup Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- FUNGSI PROSES DOKUMEN ---
def process_pdf(uploaded_file):
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
        # [MODIFIKASI 2] Kembalikan ukuran chunk ke normal (1000)
        # Kalau 10 terlalu kecil, AI tidak akan mengerti konteks kalimat.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    
            chunk_overlap=200   
        )
        splits = text_splitter.split_documents(docs)
        st.write(f"Memproses {len(splits)} potongan data secara lokal...")

        # d. Buat Embedding (LOKAL)
        # Model ini akan didownload otomatis sekali saja (sekitar 80MB)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # [MODIFIKASI 3] Langsung proses sekaligus (Tanpa Batching/Sleep)
        # Karena lokal, tidak ada limit 429. Bisa langsung hajar semua.
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        return vectorstore

    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload Dokumen")
    uploaded_file = st.file_uploader("Upload file PDF kamu", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Proses Dokumen"):
            with st.spinner("Sedang download model & embedding (pertama kali agak lama)..."):
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.success("Selesai! Vector Database tersimpan di memory lokal.")

# --- AREA CHAT ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.write("Upload dokumen dulu bosku! 👈")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                # LLM TETAP PAKAI GEMINI (Agar jawabannya pintar)
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

                prompt = ChatPromptTemplate.from_template("""
                Jawab pertanyaan berdasarkan konteks berikut:
                <context>
                {context}
                </context>
                Pertanyaan: {input}
                """)

                retriever = st.session_state.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({"input": user_query})
                answer = response['answer']
                
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))