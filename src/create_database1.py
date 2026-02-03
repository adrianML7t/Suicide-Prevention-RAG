from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

FILE_PATH = "Files/3rd"
CHROMA_PATH = "chroma3/"

def get_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

def load_docs():
    loader = DirectoryLoader(FILE_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def create_vector_db():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    
    docs = load_docs()
    chunks = splitter.split_documents(docs)
    embedding_model = get_embedding_model()
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print("BDD creada")

def get_db():
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    return vector_store

if __name__ == "__main__": #Nota: los imports retrasan bastante la ejecucion del programa..
    create_vector_db()