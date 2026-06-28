from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sys


def get_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

def load_docs(file_path):
    loader = DirectoryLoader(file_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def create_vector_db(file_path, chroma_path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    
    docs = load_docs(file_path)
    chunks = splitter.split_documents(docs)
    embedding_model = get_embedding_model()
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_path,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"BDD {chroma_path} creada a partir de {file_path}")

def get_db(chroma_path):
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model
    )
    return vector_store

if __name__ == "__main__": 
    if len(sys.argv) == 1:
        print("to execute: src/create_database.py file_path chroma_path")
    elif len(sys.argv) == 3:
        file_path = sys.argv[1]
        chroma_path = sys.argv[2]
        create_vector_db(file_path, chroma_path)
