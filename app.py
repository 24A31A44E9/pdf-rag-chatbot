import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# Page Config
st.set_page_config(page_title="Smart PDF RAG Chatbot", page_icon="📖", layout="centered")

# Custom CSS for Sleek Dark Theme
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
st.markdown('<p class="sub-text">Upload any PDF (normal or text-heavy) and chat with intelligent AI answers!</p>', unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Config (Using Streamlit Secrets automatically)
with st.sidebar:
    st.header("📁 Document Upload")
    
    # Safely load Groq API key from Streamlit Secrets
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None
        st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets!")

    uploaded_file = st.file_uploader("Upload your PDF file", type=[".pdf"])
    
    if uploaded_file is not None:
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process PDF 🚀", type="primary"):
            if not groq_api_key:
                st.error("⚠️ Please configure your API key in Streamlit Secrets.")
            else:
                with st.spinner("Processing document chunks & building embeddings..."):
                    try:
                        try:
                            loader = PyPDFLoader(temp_pdf_path)
                            pages = loader.load()
                        except Exception:
                            loader = UnstructuredPDFLoader(temp_pdf_path)
                            pages = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = text_splitter.split_documents(pages)
                        
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.vector_store = Chroma.from_documents(docs, embeddings)
                        
                        st.success(f"✅ Success! Processed {len(pages)} pages.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY is missing in Streamlit Secrets.")
    elif st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF file first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching document..."):
                try:
                    # Retrieve relevant document chunks
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(prompt)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Call Groq LLM for accurate synthesis
                    client = Groq(api_key=groq_api_key)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful AI assistant. Answer the user's question accurately using ONLY the provided context from the document. If the answer is not in the context, politely state that you cannot find it in the document."
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"
                            }
                        ],
                        temperature=0.3,
                    )
                    
                    response = completion.choices[0].message.content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
