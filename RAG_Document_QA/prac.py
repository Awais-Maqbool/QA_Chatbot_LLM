import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chain import create_retriever_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import  load_dotenv
load_dotenv()

key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model = ChatGroq(
    model='openai/gpt-oss-120b',
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

def create_database():
    if 'vector' not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_states.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.database = FAISS.from_documents(st.session_state.final_document,
        st.session_state.embeddings)
    

