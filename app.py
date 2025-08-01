# app.py

import streamlit as st
import tempfile
import os
from rag_backend import process_and_store, get_qa_chain

st.set_page_config(page_title="RAG Chatbot with TinyLLaMA", page_icon="🧠")
st.title("🧠 RAG Chatbot (TinyLLaMA + Ollama)")
st.write("Upload a file once and ask multiple questions about it.")

# Upload once and save temp file path
if "file_path" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.file_path = tmp_file.name
        st.success("File uploaded! Now processing...")

        # Build vector store and chain once
        process_and_store(st.session_state.file_path)
        st.session_state.qa_chain = get_qa_chain()
        st.success("Document indexed! Ask your questions below 👇")

# Ask questions in multi-turn style
if "qa_chain" in st.session_state:
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain(query)

        st.write("### 💬 Answer")
        st.write(result["result"])

        st.write("### 📄 Source(s):")
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")