
import os
import boto3
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2", client=bedrock)

def handle_file_upload():
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "csv", "docx"])
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    return None

def data_ingestion(uploaded_file=None):
    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file)[1].lower()
        if file_ext == '.pdf':
            loader = PyPDFLoader(uploaded_file)
        elif file_ext == '.txt':

            loader = TextLoader(uploaded_file)
        elif file_ext == '.csv':

            loader = CSVLoader(uploaded_file)
        elif file_ext == '.docx':

            loader = Docx2txtLoader(uploaded_file)
    else:
        loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_v2():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock,
                model_kwargs={'max_tokens_to_sample': 512})
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1", client=bedrock,
                model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Analyze the following context carefully and provide a detailed response to the question.
Include:
1. Key points from the context
2. Relevant examples
3. Practical applications

Context: {context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Doc Chat")
    
    st.header("Document Q&A using AWS Bedrock 💬")

    user_question = st.text_input("Ask a question about the files uploaded")

    with st.sidebar:
        st.title("Files Processing")
        
        uploaded_file = handle_file_upload()
        
        if st.button("Create Vector Store"):
            with st.spinner("Processing..."):
                docs = data_ingestion(uploaded_file)
                get_vector_store(docs)
                st.success("Vector store created successfully!")

    model_option = st.radio(
        "Select Model:",
        ("Claude v2", "Llama3")
    )

    if st.button("Get Answer"):
        with st.spinner("Generating response..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            
            if model_option == "Claude v2":
                llm = get_claude_v2()
            elif model_option == "Llama3":
                llm = get_llama3_llm()
            else:
                llm = get_claude_v2()
            
            st.write(get_response(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()