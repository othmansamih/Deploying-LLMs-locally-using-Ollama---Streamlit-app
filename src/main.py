import os
import streamlit as st
from main_utilities import save_uploaded_file, get_answer


working_dir = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(
    page_title="Chat with your document",
    page_icon="🖹",
    layout="centered"
)

st.title("Document Q&A - Gemma2 - Ollama")

uploaded_file = st.file_uploader(label="Upload your file here",
                                 accept_multiple_files=False,
                                 type="pdf")

if uploaded_file:
    uploaded_file_path = save_uploaded_file(uploaded_file)
    query = st.text_input("Ask me anything about your document")
    if st.button("Run") and query:
        answer = get_answer(uploaded_file_path, query)
        st.success(answer)