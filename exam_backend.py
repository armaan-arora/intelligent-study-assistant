from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph_backend_multi_hybrid_chroma import _THREAD_DOCUMENTS, llm , _restore_thread

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Option(BaseModel):
    a: str = Field(description="Option A")
    b: str = Field(description="Option B")
    c: str = Field(description="Option C")
    d: str = Field(description="Option D")

class Question(BaseModel):
    question: str = Field(description="The question text")
    options: Option = Field(description="Four options for the question")
    correct_answer: str = Field(description="Correct option key: a, b, c or d")
    explanation: str = Field(description="Why the correct answer is correct")

class Quiz(BaseModel):
    topic: str = Field(description="Main topic of the quiz")
    questions: List[Question] = Field(description="List of MCQ questions")

class BulletPoint(BaseModel):
    point: str = Field(description="A single study note bullet point")

class TopicNotes(BaseModel):
    topic: str = Field(description="Topic heading")
    bullets: List[BulletPoint] = Field(description="Bullet points under this topic")

class Notes(BaseModel):
    title: str = Field(description="Title of the notes")
    topics: List[TopicNotes] = Field(description="List of topics with bullet points")

class ChapterSummary(BaseModel):
    source: str = Field(description="PDF filename")
    summary: str = Field(description="2-3 line summary of this source")
    key_points: List[str] = Field(description="3-5 key points from this source")

class DocumentSummary(BaseModel):
    summaries: List[ChapterSummary] = Field(description="Summary per PDF source")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_chunks(thread_id: str) -> list:
    """Fetch all stored chunks for a thread."""
    # return _THREAD_DOCUMENTS.get(str(thread_id), [])
    key = str(thread_id)
    if key not in _THREAD_DOCUMENTS:
        _restore_thread(key)   # rebuild from Chroma if not in memory
    return _THREAD_DOCUMENTS.get(key, [])

def _chunks_to_context(chunks: list, max_chunks: int = 150) -> str:
    """Convert chunks to a single context string for the LLM."""
    selected = chunks[:max_chunks]
    return "\n\n".join([
        f"[Source: {c.metadata.get('source','unknown')}, Page: {c.metadata.get('page','?')}]\n{c.page_content}"
        for c in selected
    ])

DIFFICULTY_INSTRUCTIONS = {
    "Easy": "Ask direct factual questions. Answers should be explicitly stated in the text.",
    "Medium": "Ask questions that require understanding concepts, not just recalling facts.",
    "Hard": "Ask analytical questions that require comparing, contrasting, or applying concepts."
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Quiz Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_quiz(
    thread_id: str,
    num_questions: int = 5,
    difficulty: str = "Medium"
) -> Quiz | None:
    """
    Generate an MCQ quiz from uploaded PDFs for the given thread.
    Returns a Quiz Pydantic object or None if no documents found.
    """
    chunks = _get_chunks(thread_id)
    if not chunks:
        return None

    context = _chunks_to_context(chunks)
    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["Medium"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert quiz generator for students.
         
Generate exactly {num_questions} MCQ questions from the provided study material.

Difficulty level: {difficulty}
Difficulty instruction: {difficulty_instruction}

Rules:
- Each question must have exactly 4 options (a, b, c, d)
- correct_answer must be exactly one of: a, b, c, d
- Explanation must clearly justify why the answer is correct
- Questions must be based ONLY on the provided content
- Do not repeat questions
- Make options plausible — avoid obviously wrong choices
"""),
        ("human", "Study material:\n\n{context}")
    ])

    structured_llm = llm.with_structured_output(Quiz)
    chain = prompt | structured_llm

    try:
        quiz = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "difficulty_instruction": difficulty_instruction,
            "context": context
        })
        return quiz
    except Exception as e:
        print(f"Quiz generation error: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Notes Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_notes(thread_id: str) -> Notes | None:
    """
    Generate structured topic-wise study notes from uploaded PDFs.
    Returns a Notes Pydantic object or None if no documents found.
    """
    chunks = _get_chunks(thread_id)
    if not chunks:
        return None

    context = _chunks_to_context(chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert study notes creator.

Generate clear, structured study notes from the provided material.

Rules:
- Organize notes by topic/theme
- Each topic should have 3-6 bullet points
- Bullet points should be concise and study-friendly
- Cover all major concepts from the material
- Use simple, clear language
"""),
        ("human", "Study material:\n\n{context}")
    ])

    structured_llm = llm.with_structured_output(Notes)
    chain = prompt | structured_llm

    try:
        notes = chain.invoke({"context": context})
        return notes
    except Exception as e:
        print(f"Notes generation error: {e}")
        return None
    
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Chapter Summarizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def summarize_chapters(thread_id: str) -> DocumentSummary | None:
    """
    Summarize each uploaded PDF separately.
    Returns a DocumentSummary Pydantic object or None if no documents found.
    """
    chunks = _get_chunks(thread_id)
    if not chunks:
        return None

    # group chunks by source PDF
    source_chunks: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in source_chunks:
            source_chunks[source] = []
        source_chunks[source].append(chunk.page_content)

    # build context per source
    source_contexts = {}
    for source, contents in source_chunks.items():
        source_contexts[source] = "\n\n".join(contents[:50])  # max 50 chunks per PDF

    combined_context = ""
    for source, context in source_contexts.items():
        combined_context += f"\n\n=== PDF: {source} ===\n{context}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document summarizer.

Summarize each PDF document separately.

Rules:
- Write a 2-3 line summary for each PDF
- Extract 3-5 key points per PDF
- Keep summaries concise and study-friendly
- Each summary must reference its PDF source name exactly as provided
"""),
        ("human", "Documents to summarize:\n\n{context}")
    ])

    structured_llm = llm.with_structured_output(DocumentSummary)
    chain = prompt | structured_llm

    try:
        summary = chain.invoke({"context": combined_context})
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return None