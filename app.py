# Sidebar Config (Using Streamlit Secrets automatically)
with st.sidebar:
    st.header("📁 Document Upload")
    
    # Automatically load Groq API key from Streamlit Secrets
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except:
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
                        except:
                            loader = UnstructuredPDFLoader(temp_pdf_path)
                            pages = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = text_splitter.split_documents(pages)
                        
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.vector_store = Chroma.from_documents(docs, embeddings)
                        
                        st.success(f"✅ Success! Processed {len(pages)} pages.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
