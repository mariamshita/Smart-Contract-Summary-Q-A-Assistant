# 📄 Smart Contract Summary & Q&A Assistant

A sophisticated RAG (Retrieval-Augmented Generation) system designed to analyze and query smart contracts with high precision. This project combines **Local Hybrid Search**, **Cross-Encoder Reranking**, and **Multi-turn Conversation Memory** to provide an expert-level analysis experience through a modern, ChatGPT-like interface.

---

## ✨ Features

- **Multi-Format Support**: Upload and analyze smart contracts in **PDF** or **DOCX** formats.
- **Local Hybrid Search**: Combines the precision of **BM25** (keyword search) with the contextual understanding of **Vector Search** (ChromaDB).
- **Cross-Encoder Reranking**: Utilizes `ms-marco-MiniLM-L-6-v2` to re-score retrieved documents for maximum relevance.
- **Conversational Memory**: Remembers previous context in the chat (last 5 exchanges) for natural, follow-up questions.
- **Modern Chat UI**: Premium dark-themed, responsive dashboard with glassmorphism aesthetics and real-time feedback.
- **Source Citation**: Clearly displays exactly which parts of the contract were used to generate each answer.
- **LangServe Enabled**: Exposes the core RAG chain as a standard LangChain API.
- **Rate Limited**: Integrated protection against API abuse using `slowapi`.

---

## 🛠️ Tech Stack

- **Backend**: Python, [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://www.langchain.com/)
- **LLM**: Google **Gemini 1.5 Flash** (via `langchain-google-genai`)
- **Embeddings**: Google **text-embedding-004**
- **Vector Store**: [ChromaDB](https://www.trychroma.com/) (Local)
- **Reranker**: Sentence-Transformers (`ms-marco-MiniLM-L-6-v2`)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3 (Custom Glassmorphism Design)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google AI (Gemini) API Key. Get one at [Google AI Studio](https://aistudio.google.com/).

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Clone the repository (or extract files)
# cd to the project directory

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Running the Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```

Access the dashboard at: **[http://localhost:8000](http://localhost:8000)**

---

## 📂 Project Structure

- `main.py`: Core FastAPI application and API routing.
- `retrieval.py`: Advanced hybrid search and cross-encoder reranking logic.
- `utils.py`: Document parsing and text chunking utilities.
- `static/`: Frontend assets (modern chat UI).
- `chroma_db/`: Local persistent vector database.
- `.env`: Environment variables (API keys).
- `requirements.txt`: Python dependencies.

---

## 🛡️ Security & Analysis Logic

The assistant is prompted as an expert smart contract analyst. It follows strict instructions to:
1. Provide accurate, detailed answers based **only** on the uploaded context.
2. Highlight security concerns (like reentrancy, access control issues, etc.).
3. State clearly if information is missing from the provided documents.

---

## 📜 License
Internal Project / Academic Use.
