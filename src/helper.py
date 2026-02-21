from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document


#To extract data from PDF file
def load_knowledge_files(data):
    loder = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loder.load()
    return documents


#To filter required data from extracted documents
def filter_required_data_from_Knowledge_files(document):

    filtered_document = []

    for doc in document:
        metadata_src = {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
            }
        filtered_document.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source":metadata_src}
            )
        )
    
    return filtered_document


#To perform text spliting on the extratced document
def perform_text_spliting(filtered_document):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 50
    )
    text = text_splitter.split_documents(filtered_document)
    return text


#Download the Opsource Embeded Model
def download_opensource_embeded_model():
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    return embedding_model

#Create Index
def create_vector_database(text, embeding_model):

    vector_db = FAISS.from_documents(
    documents=text,
    embedding=embeding_model
    )
    return vector_db
    
#Save the Index
def save_vertor_database(vector_db):
    
    vector_db.save_local("vector_database\\oncology_faiss_index")

#Load the locally stored Index
def load_local_vector_database(database_name, embeding_model):

    vector_db = FAISS.load_local(
    database_name,
    embeding_model,
    allow_dangerous_deserialization=True
    )

    return vector_db