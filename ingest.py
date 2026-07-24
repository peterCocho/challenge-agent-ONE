import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()


def build_vector_database():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is missing.")

    # 1. Load raw text files from the 'docs' directory
    print("Loading documents from ./docs...")
    # Force UTF-8 encoding to prevent Windows cp1252 decoding errors
    loader = DirectoryLoader(
        './docs',
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    if not documents:
        raise RuntimeError("No text documents found. Ensure your .txt files are inside the 'docs' folder.")

    # 2. Split documents into manageable chunks to respect token limits
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    document_chunks = text_splitter.split_documents(documents)

    # 3. Generate embeddings and build the FAISS index
    print(f"Generating embeddings for {len(document_chunks)} chunks...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=cohere_api_key
    )
    vector_store = FAISS.from_documents(document_chunks, embeddings)

    # 4. Persist the index to disk
    index_path = "./faiss_index"
    print(f"Saving vector store to {index_path}...")
    vector_store.save_local(index_path)

    print("Vector database built successfully. You can now launch app.py.")


if __name__ == "__main__":
    build_vector_database()