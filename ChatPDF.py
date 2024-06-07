import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

def main():
    st.title("Chat with PDF using RAG with LangChain")

    # Get OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and openai_api_key:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create vector store
        db = Chroma.from_documents(texts, embeddings)

        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key), chain_type="stuff", retriever=db.as_retriever())

        # Get user question
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            # Get answer
            answer = qa.run(user_question)
            st.write(answer)

        # Remove temporary file
        os.remove(temp_path)

if __name__ == "__main__":
    main()
