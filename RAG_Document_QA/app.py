import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader


import streamlit as st

st.write("Groq key exists:", "GROQ_API_KEY" in st.secrets)

import os
from dotenv import  load_dotenv 
load_dotenv()

key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=st.secrets["GROQ_API_KEY"]
)

prompt = ChatPromptTemplate.from_template(
    '''provide the answer based on context only.
    please provide the answer based on question
    <context>
    {context}
    <context>
    question:{input}
    '''
)


def create_vector_embedding():
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings

    if "vectors" in st.session_state:
        st.write("Vector database already exists.")
        return

    st.write("Step 3: Initializing embedding model")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    st.session_state.embeddings = embeddings

    pdf_folder = Path(__file__).parent / "research_papers"
    st.write("PDF folder:", pdf_folder)

    loader = PyPDFDirectoryLoader(str(pdf_folder))
    docs = loader.load()

    st.write("Documents loaded:", len(docs))

    if not docs:
        raise ValueError("No PDF documents were loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_documents = splitter.split_documents(docs)

    st.write("Chunks created:", len(final_documents))

    if not final_documents:
        raise ValueError(
            "No text chunks were extracted from the PDFs."
        )

    st.write("Step 4: Creating FAISS vector database")

    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    st.session_state.vectors = vectors

    st.write("Step 5: Vector database saved in session.")


st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")


if st.button("Document Embedding"):
    st.write("Step 1: Button clicked")

    try:
        create_vector_embedding()
        st.write("Step 2: Embedding function finished")

        if "vectors" in st.session_state:
            st.success("Vector Database is ready!")
        else:
            st.error("Embedding function finished, but vectors were not created.")

    except Exception as e:
        st.error(f"Embedding failed: {e}")


import time
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first.")
    else:
            document_chain = create_stuff_documents_chain(
                model, prompt
            )

            retriever = st.session_state.vectors.as_retriever()

            retrieval_chain = create_retrieval_chain(
                retriever, document_chain
            )

            response = retrieval_chain.invoke({
                "input": user_prompt
            })

            st.write(response["answer"])

            with st.expander("Document Similarity Search"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("------------------------")
