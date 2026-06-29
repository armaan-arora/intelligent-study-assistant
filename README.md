# 📚 Intelligent Study Assistant

An AI-powered **Multi-Document Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDFs, ask questions across documents, and receive accurate, citation-backed answers. The application combines **keyword search** and **semantic search** using **Reciprocal Rank Fusion (RRF)** for improved retrieval quality and uses a **LangGraph ReAct agent** for intelligent tool routing.

---

## ✨ Features

- 📄 Upload and query multiple PDF documents
- 🔍 Hybrid Retrieval
  - BM25 Keyword Search
  - FAISS Semantic Vector Search
  - Reciprocal Rank Fusion (RRF)
- 🤖 LangGraph ReAct Agent for intelligent reasoning and tool selection
- 📑 Citation-aware responses with source document and page number
- 🌐 Web Search fallback using Tavily when local documents lack sufficient information
- 📝 Exam Mode
  - AI-generated quizzes
  - Notes generation
  - Chapter summaries
- 💬 Persistent conversation memory using SQLite
- 🗂️ Conversation thread management
  - Create
  - Rename
  - Delete
- 📋 Structured outputs using Pydantic
- 🎨 Interactive Streamlit interface

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Framework | LangGraph, LangChain |
| LLM | OpenAI GPT-4o-mini |
| Retrieval | FAISS, BM25 |
| Ranking | Reciprocal Rank Fusion (RRF) |
| UI | Streamlit |
| Database | SQLite |
| Validation | Pydantic |
| Web Search | Tavily API |
| Language | Python |

---

## 🏗️ System Architecture

```
                User
                  │
                  ▼
        Streamlit Frontend
                  │
                  ▼
          LangGraph ReAct Agent
         ┌────────┴─────────┐
         │                  │
         ▼                  ▼
   Hybrid Retriever     Tavily Search
         │
         ▼
  BM25 + FAISS Retrieval
         │
         ▼
Reciprocal Rank Fusion
         │
         ▼
 Relevant Context
         │
         ▼
 OpenAI GPT-4o-mini
         │
         ▼
 Citation-backed Response
```

---

## 📂 Project Structure

```
Study-A/
│
├── streamlit_frontend_multi_hybrid_chroma.py
├── langgraph_backend_multi_hybrid_chroma.py
├── exam_backend.py
├── exam_ui_pdf.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/intelligent-study-assistant.git

cd intelligent-study-assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Run the application

```bash
streamlit run streamlit_frontend_multi_hybrid_chroma.py
```

---

## 📸 Demo

Add screenshots or a GIF here.

Example:

```
screenshots/demo.png
```

---

## 🎯 Key Highlights

- Hybrid Retrieval with BM25 + FAISS
- Reciprocal Rank Fusion for improved ranking
- LangGraph ReAct Agent
- Multi-document RAG
- Source-grounded answers with citations
- Web search fallback
- Persistent chat memory
- AI-powered exam preparation

---

## 🔮 Future Improvements

- OCR support for scanned PDFs
- Image understanding
- Multi-modal document retrieval
- Support for DOCX and PPTX
- User authentication
- Cloud vector database integration
- Docker deployment

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**

LinkedIn: https://linkedin.com/in/your-linkedin

GitHub: https://github.com/yourusername
