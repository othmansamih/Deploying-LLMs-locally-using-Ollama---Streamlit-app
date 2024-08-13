import os
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import time
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["OPENAI_API_KEY"] = json.load(open(f"{working_dir}/config.json"))["OPENAI_API_KEY"]
system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )


def save_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    bytes_data = uploaded_file.read()
    file_path = f"{working_dir}/{file_name}"
    with open(file_path, "wb") as f:
        f.write(bytes_data)
        f.close()
    return file_path

def get_answer(uploaded_file_path, query):
    loader = UnstructuredPDFLoader(uploaded_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    start_time = time.time()
    chunks = text_splitter.split_documents(documents)
    end_time = time.time()
    st.write(f"The process of splitting the document is: {end_time-start_time}")
    start_time = time.time()
    embeddings_model = HuggingFaceEmbeddings()
    end_time = time.time()
    st.write(f"The process of loading the embeddings model is: {end_time - start_time}")
    start_time = time.time()
    db = FAISS.from_documents(chunks, embeddings_model)
    end_time = time.time()
    st.write(f"The process of embedding the chunks is: {end_time - start_time}")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = OllamaLLM(model="gemma2:2b")
    #llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    start_time = time.time()
    response = rag_chain.invoke({"input": query})
    end_time = time.time()
    st.write(f"The process of returning the response from the model is: {end_time - start_time}")
    return response["answer"]

