from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever




BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "chatbot2.db"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

print("DB PATH:", DB_PATH)


load_dotenv()

#********************************** LLM**********************
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_CHROMA: Dict[str, Chroma] = {}
_THREAD_DOCUMENTS: Dict[str, list] = {}  # ← stores raw chunks for BM25
#******************************* utilities *******************

def _get_retriever(thread_id : str):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def save_file_for_thread(thread_id: str, filename: str):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_files (
            thread_id TEXT,
            filename TEXT,
            PRIMARY KEY (thread_id, filename)
        )
    """)
    conn.execute(
        "INSERT OR IGNORE INTO thread_files (thread_id, filename) VALUES (?, ?)",
        (str(thread_id), filename)
    )
    conn.commit()

def get_files_for_thread(thread_id: str) -> list:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_files (
            thread_id TEXT,
            filename TEXT,
            PRIMARY KEY (thread_id, filename)
        )
    """)
    rows = conn.execute(
        "SELECT filename FROM thread_files WHERE thread_id = ?",
        (str(thread_id),)
    ).fetchall()
    return [row[0] for row in rows]

def save_thread_title(thread_id: str, title: str, overwrite: bool = False):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)
    if overwrite:
        conn.execute(
            "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (str(thread_id), title)
        )
    else:
        conn.execute(
            "INSERT OR IGNORE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (str(thread_id), title)
        )
    conn.commit()

def get_thread_title(thread_id: str) -> str:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
        """)
    row = conn.execute(
        "SELECT title FROM thread_titles WHERE thread_id = ?",
        (str(thread_id),)
    ).fetchone()
    return row[0] if row else "New Chat"  # fallback for old threads

def _restore_thread(thread_id: str) -> None:
    """Reload chunks from Chroma after a server restart and rebuild BM25 + hybrid retriever."""
    key = str(thread_id)
    if key in _THREAD_RETRIEVERS:
        return  # already warm

    # Load Chroma collection from disk
    vs = Chroma(
        collection_name=f"t-{key[:60]}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    _THREAD_CHROMA[key] = vs

    # Pull stored docs back out to rebuild BM25
    raw = vs.get(include=["documents", "metadatas"])
    if not raw["documents"]:
        return

    from langchain_core.documents import Document
    docs = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]
    _THREAD_DOCUMENTS[key] = docs

    # Rebuild hybrid retriever
    vector_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4
    _THREAD_RETRIEVERS[key] = EnsembleRetriever(
        retrievers=[bm25, vector_retriever], weights=[0.4, 0.6]
    )
    print(f"[restore] thread={key} docs={len(docs)}")

def ingest(file_bytes : bytes , thread_id : str, filename: str = "document.pdf") -> dict:

    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes :
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"
    ) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    # ✅ override AFTER splitting — this is the fix
    for chunk in chunks:
        chunk.metadata["source"] = filename
        # page is already set by PyPDFLoader, keep it

    thread_key = str(thread_id)
    # ── vector store (FAISS) ──
    if thread_key not in _THREAD_CHROMA:
        _THREAD_CHROMA[thread_key] = Chroma(
            collection_name=f"t-{thread_key[:60]}",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    _THREAD_CHROMA[thread_key].add_documents(chunks)
    
    # ── BM25 document store ──
    if thread_key in _THREAD_DOCUMENTS:
        _THREAD_DOCUMENTS[thread_key].extend(chunks)  # ← merge chunks
    else:
        _THREAD_DOCUMENTS[thread_key] = chunks



    vector_retriever = _THREAD_CHROMA[thread_key].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    bm25_retriever = BM25Retriever.from_documents(
        _THREAD_DOCUMENTS[thread_key]
    )
    bm25_retriever.k = 4

    hybrid_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever,vector_retriever],
        weights = [0.4,0.6] # 40% BM25, 60% vector

    ) #uses Reciprocal Rank Fusion (RRF)
    _THREAD_RETRIEVERS[thread_key] = hybrid_retriever
    
    
    return {
        "status": "success",
        "pages": len(docs),
        "chunks": len(chunks),
        "total_docs": len(_THREAD_CHROMA[thread_key].get(include=[])["ids"])
    }

#******************************** Tools ************************************


@tool
def rag_tool(
    query: str,
    config: RunnableConfig
) -> dict:
    """
    Retrieve relevant information from the uploaded PDF
    for the current chat thread.
    """
    

    thread_id = str(
    config["configurable"]["thread_id"]
    )
   
    
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)


    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
    }

tavily_search = TavilySearch(
    max_results=5,
    search_depth="basic"
)

@tool
def tavily_tool(query: str) -> dict:
    """
    Web search tool using Tavily via LangChain.
    Use this when answer is not in PDF.
    """

    results = tavily_search.invoke(query)

    return {
        "query": query,
        "results": results
    }

tools = [rag_tool, tavily_tool]
llm_with_tools = llm.bind_tools(tools)


#************************** State********************************

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


#*********************************** Nodes **************************
from langchain_core.messages import SystemMessage

def chat_node(state: ChatState, config: RunnableConfig):

    system_message = SystemMessage(
     content="""
You are an Intelligent Study Assistant.

You have access to two tools:
1. rag_tool → Use for questions based on uploaded PDF documents.
2. tavily_tool → Use for general knowledge or when information is not available in the PDF.

---

RULES:

1. ALWAYS check if the question can be answered from the uploaded PDF first.
   - If yes → use rag_tool ONLY.

2. If the PDF does NOT contain enough information:
   - Use tavily_tool to search the web.

3. If the question is general (not related to uploaded PDF):
   - Use tavily_tool.

4. Never guess or hallucinate answers.
   If you are unsure, always use a tool.

5. Always use tool output as the source of truth.

6. Keep answers concise, clear, and study-friendly.

7. If multiple sources are used (PDF + web), clearly separate them in the answer.

8. When answering from the PDF, ALWAYS cite your source at the end of each point like:
   [Source: filename.pdf, Page 3]
   This information is available in the metadata returned by rag_tool.

---
"""
)
    

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config
    )
    
    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


#********************************** checkpointer *********************************

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)



#************************** Graph ***********************************
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


for _tid in retrieve_all_threads():
    _restore_thread(_tid)
    
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# response = chatbot.invoke(
#     {'messages':[HumanMessage(content='what is my name')]},
#     config=CONFIG
# )

# print(response)
