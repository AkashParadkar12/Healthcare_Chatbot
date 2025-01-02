import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os
load_dotenv()


# Create a .env file and add your API keys
with open(".env", "w") as f:
    f.write('GROQ_API_KEY="Enter Groq Api Key")
    f.write('GOOGLE_API_KEY="Enter Google Api Key")

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the API keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print("GROQ API Key:", groq_api_key)
print("Google API Key:", google_api_key)

## load the GROQ And OpenAI API KEY
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Healthcare Chatbot Using Rag")

llm =ChatGroq(groq_api_key=groq_api_key,model_name ="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader('C:/Users/Dell/Desktop/Rag_APP/Dataset')## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

prompt1=st.text_input("Enter Your Question From Doduments")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

