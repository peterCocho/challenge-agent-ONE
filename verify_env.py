# Import the retrieval chain from the base chains module
from langchain.chains import create_retrieval_chain

# Import the document combination chain from its specific submodule
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

print("env validation imports OK")
