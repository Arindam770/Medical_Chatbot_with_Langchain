from dotenv import load_dotenv 
from src.helper import download_opensource_embeded_model, load_knowledge_files, filter_required_data_from_Knowledge_files, perform_text_spliting, create_vector_database, save_vertor_database, load_local_vector_database

load_dotenv()


loded_data = load_knowledge_files("data")
filtered_document = filter_required_data_from_Knowledge_files(loded_data)
text= perform_text_spliting(filtered_document)

embeding_model = download_opensource_embeded_model()
vector_database= create_vector_database(text, embeding_model)
save_vertor_database(vector_database)
vector_db = load_local_vector_database("vector_database\\oncology_faiss_index", embeding_model)