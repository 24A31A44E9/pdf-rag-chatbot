import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Smart PDF RAG Chatbot", page_icon="📖", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #09090b, #18181b, #0f172a, #030712);
        color: #f4f4f5;
    }
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #f4f4f5;
        margin-bottom: 5px;
    }
    .sub-text {
        color: #a1a1aa;
        font-size: 14px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📖 Smart Personal Document RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload your PDF study notes or research paper and chat with it instantly!</p>', unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF file", type=[".pdf"])
    
    if uploaded_file is not None:
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process PDF 🚀", type="primary"):
            with st.spinner("Processing document and building vector database..."):
                try:
                    loader = PyPDFLoader(temp_pdf_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = text_splitter.split_documents(pages)
                    
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vector_store = Chroma.from_documents(docs, embeddings)
                    
                    st.success(f"✅ Success! Loaded {len(pages)} pages & {len(docs)} chunks.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF file in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                    relevant_docs = retriever.invoke(prompt)
                    
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    response = f"**🤖 AI Answer:**\n\n{context}"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
