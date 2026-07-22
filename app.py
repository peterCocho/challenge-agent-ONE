import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_vector_store(directory_path: str):
    # Fetch the Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is missing.")

    # Initialize Cohere's embedding model 
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=cohere_api_key
    )
    
    # Define local path for FAISS index storage
    index_path = "./faiss_index"
    
    # Check if the index already exists on disk
    if os.path.exists(index_path):
        print("Loading existing FAISS index from disk...")
        # Load index bypassing the serialization safety warning
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Verify if the directory exists before attempting to load
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} was not found.")

    # Use DirectoryLoader to load all .txt files
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

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
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            
            time.sleep(5) 
            
        except Exception as e:
            raise RuntimeError(f"API collapse at batch {(i // batch_size) + 1}: {e}")

    # Save the finalized index to disk
    print("Saving new FAISS index to disk...")
    vectorstore.save_local(index_path)

    return vectorstore


@st.cache_resource
def setup_rag_chain(_vectorstore):
    # Fetch configurations from environment variables
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is missing.")
        
    model_name = os.getenv("COHERE_MODEL_NAME")
    if not model_name:
        raise ValueError("COHERE_MODEL_NAME environment variable is missing in .env file.")

    # Initialize the LLM dynamically
    # Temperature 0 is strictly required to prevent hallucinations
    llm = ChatCohere(
        model=model_name,
        temperature=0,
        cohere_api_key=cohere_api_key
    )

    # Define strict boundaries for the agent
    system_prompt = (
        "You are a strict customer service assistant for BimBam Buy. "
        "Use ONLY the following pieces of retrieved context to answer the user's question. "
        "If you do not know the answer based on the context, say exactly: "
        "'Lo siento, no tengo información sobre esa política en mis manuales actuales.' "
        "Do not invent policies, shipping costs, or return timeframes. "
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Transform the vector store into a retriever
    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain using modern LCEL syntax
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Temporary execution block to verify the RAG chain
if __name__ == "__main__":
    try:
        # 1. Load the vector store
        v_store = initialize_vector_store("./data")
        
        # 2. Setup the RAG chain
        rag_chain = setup_rag_chain(v_store)
        
        # 3. Test a query directly
        test_question = "¿Cuál es el tiempo máximo para hacer una devolución?"
        print(f"\nPregunta: {test_question}")
        
        # Invoke returns a string directly due to StrOutputParser
        response = rag_chain.invoke(test_question)
        print(f"Respuesta del Agente: {response}")
        
    except Exception as e:
        print(f"Error executing RAG chain: {e}")