import os
import streamlit as st
from rag_pipeline import initialize_rag
from chatbot_backend import generate_response

st.set_page_config(page_title="Oncology Assistant", page_icon="🧬")
st.title("🧬 Oncology RAG Chatbot")

# Load retriever only once
@st.cache_resource
def load_rag():
    return initialize_rag(os.path.join("vector_database", "oncology_faiss_index"))

retriever = load_rag()

# session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
user_input = st.chat_input("Ask medical question...")

if user_input:

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    response, updated_history = generate_response(
        user_input,
        st.session_state.chat_history,
        retriever
    )

    st.session_state.chat_history = updated_history

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )