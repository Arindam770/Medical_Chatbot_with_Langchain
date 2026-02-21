from dotenv import load_dotenv
from src.helper import download_opensource_embeded_model, load_local_vector_database, retrive_relevent_document


load_dotenv()

def initialize_rag(database_name):

    embeding_model = download_opensource_embeded_model()

    # load existing DB (fast)
    vector_db = load_local_vector_database(database_name,embeding_model)

    retriever = retrive_relevent_document(vector_db)

    return retriever