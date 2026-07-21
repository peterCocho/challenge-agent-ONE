import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()

@st.cache_resource
def initialize_vector_store(directory_path: str):
    # Fetch the Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is missing.")

    # Initialize Cohere's embedding model 
    # Using v3 multilingual model for Spanish documents
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=cohere_api_key
    )
    
    # Define local path for FAISS index storage
    index_path = "./faiss_index"
    
    # Check if the index already exists on disk
    if os.path.exists(index_path):
        print("Loading existing FAISS index from disk...")
        # Load index bypassing the serialization safety warning (safe for local files)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Verify if the directory exists before attempting to load
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} was not found.")

    # Use DirectoryLoader to load all .txt files in the specified folder
    # Pass TextLoader and utf-8 encoding to prevent parsing errors
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    # Verify that documents were successfully loaded
    if not docs:
        raise ValueError("No documents were loaded. Check the directory content.")

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = None
    batch_size = 5 

    print(f"Total chunks to process: {len(splits)}")

    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Embedding batch {(i // batch_size) + 1}...")
        
        try:
            if vectorstore is None:
                # Create the FAISS index
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add documents to the existing index
                vectorstore.add_documents(batch)
            
            # Pause to avoid rate limiting
            time.sleep(5) 
            
        except Exception as e:
            raise RuntimeError(f"API collapse at batch {(i // batch_size) + 1}: {e}")

    # Save the finalized index to disk before returning
    print("Saving new FAISS index to disk...")
    vectorstore.save_local(index_path)

    return vectorstore


# Temporary execution block to verify the directory ingestion
if __name__ == "__main__":
    try:
        # Point to the data directory instead of a single file
        v_store = initialize_vector_store("./data")
        print("Vector store initialized successfully with DirectoryLoader.")
    except Exception as e:
        print(f"Error initializing vector store: {e}")