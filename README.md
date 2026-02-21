# 🧬 Medical Oncology RAG Chatbot

An AI-powered **Oncology Assistant** built using **Retrieval Augmented Generation (RAG)**, **Streamlit**, and a **FAISS vector database**.
The system retrieves knowledge from oncology documents and generates context-grounded responses using an LLM.

This project demonstrates a **real-world AI application architecture**.

---

## 🚀 Project Overview

The chatbot allows users to ask oncology-related questions through a conversational interface. Instead of relying only on LLM knowledge, it:

* Retrieves relevant medical context from indexed documents
* Grounds answers using retrieved evidence
* Maintains conversational memory
* Provides an interactive Streamlit chat experience

> ⚠️ Educational use only — not a substitute for professional medical advice.

---

## 🏗️ Architecture

```text
User (Streamlit UI)
        ↓
app.py
        ↓
chatbot_backend.py
        ↓
RAG Pipeline (rag_pipeline.py)
        ↓
FAISS Vector Database
        ↓
LLM + Prompt Engineering
        ↓
Grounded Response
```

---

## 📂 Folder Structure

```text
Medical_Chatbot_with_LangChain/
│
├── app.py                     # Streamlit UI (chat interface)
├── chatbot_backend.py         # Response generation logic
├── rag_pipeline.py            # Retriever + vector DB initialization
│
├── src/
│   ├── __init__.py
│   ├── helper.py              # Utility/helper functions
│   └── prompt.py              # Prompt templates
│
├── data/
│   └── Medical_Oncology_Handbook.pdf   # Source medical document
│
├── vector_database/           # FAISS index storage
│
├── research/
│   └── trials.ipynb           # Experimentation & testing notebooks
│
├── .env                       # API keys & environment variables
├── .gitignore
├── requirements.txt
├── setup.py
├── template.sh
├── LICENSE
└── README.md
```

---

## ⚙️ Key Features

* 💬 Chat-style medical assistant UI
* 🔎 Semantic search using FAISS
* 🧠 Conversational memory support
* ⚡ Cached retriever loading for performance
* 📚 Context-aware grounded responses
* 🧩 Modular architecture (UI / RAG / Prompt separation)

---

## 🧪 Tech Stack

| Layer           | Technology           |
| --------------- | -------------------- |
| UI              | Streamlit            |
| Backend         | Python               |
| Vector Database | FAISS                |
| LLM Framework   | LangChain            |
| Embeddings      | HuggingFace / OpenAI |
| Prompting       | Custom templates     |

---

## 🛠️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
cd Medical_Chatbot_with_LangChain
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create `.env` file:

```env
OPENAI_API_KEY=your_api_key
HUGGINGFACE_API_KEY=your_key
```

---

### 5️⃣ Build Vector Database (if not already created)

Ensure the oncology PDF exists inside `/data`.

Then run your ingestion/indexing script (if implemented inside helper/setup pipeline).

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 💬 Example Questions

* What are common treatments for lung cancer?
* Explain immunotherapy in oncology.
* What are chemotherapy side effects?
* What is targeted therapy?

---

## 🧠 How RAG Works

1. Oncology handbook is converted into embeddings.
2. FAISS stores semantic vectors.
3. User question is embedded.
4. Relevant chunks are retrieved.
5. Context is injected into prompt template.
6. LLM generates grounded answer.

---

## ⚠️ Medical Disclaimer

This chatbot provides **educational information only**.

* Not intended for diagnosis
* Not a replacement for clinicians
* Always consult a licensed oncologist for treatment decisions

---

## 🔮 Future Improvements

* ✅ Source citations in answers
* ✅ Streaming responses
* ✅ Hallucination guardrails
* ✅ RAGAS / DeepEval evaluation
* ✅ Agentic tool-calling workflows
* ✅ Feedback scoring system

---

## 📜 License

MIT License.

---

**🧬 Built with AI, Retrieval Engineering, and curiosity.**
