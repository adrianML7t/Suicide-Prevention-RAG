from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Paths
FILE_PATH = "Files"
CHROMA_PATH="chroma"

#Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True,
)


#Embedding model
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}

embedding_model= HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

#Loader function
def load_docs():
    loader = DirectoryLoader(FILE_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def create_vector_DB(): #solo se ejecuta directamente
    #Ejecucion
    docs = load_docs()
    chunks = splitter.split_documents(docs)

    #Database
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH)

    print("Base de datos creada correctamente")
    

def get_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    return db

if __name__ == "__main__":
    create_vector_DB()
