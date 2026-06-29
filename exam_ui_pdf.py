import streamlit as st
from exam_backend import generate_quiz, generate_notes, summarize_chapters
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
import io

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _no_docs_warning():
    st.warning("⚠️ No PDFs uploaded yet. Go to the Chat tab and upload PDFs first.")

def _notes_to_pdf(notes) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('title', fontSize=20, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold')
    heading_style = ParagraphStyle('heading', fontSize=14, spaceAfter=6, fontName='Helvetica-Bold')
    bullet_style = ParagraphStyle('bullet', fontSize=11, spaceAfter=4, leftIndent=20, fontName='Helvetica')

    story = []
    story.append(Paragraph(notes.title, title_style))
    story.append(Spacer(1, 0.5*cm))

    for topic in notes.topics:
        story.append(Paragraph(topic.topic, heading_style))
        for bullet in topic.bullets:
            story.append(Paragraph(f"- {bullet.point}", bullet_style))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return buffer.getvalue()


def _summary_to_pdf(doc_summary) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('title', fontSize=20, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold')
    heading_style = ParagraphStyle('heading', fontSize=14, spaceAfter=6, fontName='Helvetica-Bold')
    label_style = ParagraphStyle('label', fontSize=11, spaceAfter=4, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('body', fontSize=11, spaceAfter=4, fontName='Helvetica')
    bullet_style = ParagraphStyle('bullet', fontSize=11, spaceAfter=4, leftIndent=20, fontName='Helvetica')

    story = []
    story.append(Paragraph("Document Summaries", title_style))
    story.append(Spacer(1, 0.5*cm))

    for chapter in doc_summary.summaries:
        story.append(Paragraph(chapter.source, heading_style))
        story.append(Paragraph("Summary:", label_style))
        story.append(Paragraph(chapter.summary, body_style))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Key Points:", label_style))
        for point in chapter.key_points:
            story.append(Paragraph(f"- {point}", bullet_style))
        story.append(Spacer(1, 0.4*cm))

    doc.build(story)
    return buffer.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quiz UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_quiz(thread_id: str):
    st.subheader("Generate Quiz")

    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.selectbox("Number of questions", [5, 10, 15], index=0)
    with col2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    if st.button("Generate Quiz ✨", use_container_width=True):
        with st.spinner("Generating quiz from your PDFs..."):
            quiz = generate_quiz(thread_id, num_questions, difficulty)

        if quiz is None:
            _no_docs_warning()
            return

        st.session_state[f'quiz_{thread_id}'] = quiz
        st.session_state[f'quiz_answers_{thread_id}'] = {}
        st.session_state[f'quiz_submitted_{thread_id}'] = False

    if f'quiz_{thread_id}' in st.session_state and st.session_state[f'quiz_{thread_id}']:
        quiz = st.session_state[f'quiz_{thread_id}']

        st.markdown(f"### 📚 {quiz.topic}")
        st.markdown(f"**{len(quiz.questions)} questions — {difficulty} difficulty**")
        st.divider()

        for i, q in enumerate(quiz.questions):
            st.markdown(f"**Q{i+1}. {q.question}**")

            options = {
                "a": q.options.a,
                "b": q.options.b,
                "c": q.options.c,
                "d": q.options.d,
            }

            selected = st.radio(
                label=f"Select answer for Q{i+1}",
                options=list(options.keys()),
                format_func=lambda x, opts=options: f"{x.upper()}. {opts[x]}",
                key=f"q_{i}_{thread_id}",
                label_visibility="collapsed",
                index=None
            )
            st.session_state[f'quiz_answers_{thread_id}'][i] = selected
            st.markdown("")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Quiz ✅", use_container_width=True):
                st.session_state[f'quiz_submitted_{thread_id}'] = True
        with col2:
            if st.button("Retake Quiz 🔄", use_container_width=True):
                st.session_state[f'quiz_{thread_id}'] = None
                st.session_state[f'quiz_answers_{thread_id}'] = {}
                st.session_state[f'quiz_submitted_{thread_id}'] = False
                st.rerun()

        if st.session_state.get(f'quiz_submitted_{thread_id}'):
            st.divider()
            correct = 0
            total = len(quiz.questions)

            for i, q in enumerate(quiz.questions):
                user_ans = st.session_state[f'quiz_answers_{thread_id}'].get(i)
                is_correct = user_ans == q.correct_answer

                if is_correct:
                    correct += 1
                    st.success(f"✅ Q{i+1} — Correct!")
                else:
                    st.error(
                        f"❌ Q{i+1} — You answered **{user_ans.upper()}**, "
                        f"correct was **{q.correct_answer.upper()}**"
                    )
                    st.info(f"💡 {q.explanation}")

            st.divider()
            score_pct = int((correct / total) * 100)
            st.markdown(f"## 🎯 Score: {correct}/{total} ({score_pct}%)")

            if score_pct == 100:
                st.balloons()
                st.success("Perfect score! Outstanding! 🏆")
            elif score_pct >= 70:
                st.success("Good job! Keep it up! 💪")
            elif score_pct >= 50:
                st.warning("Not bad! Review the explanations above. 📖")
            else:
                st.error("Keep studying! You'll get there. 📚")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Notes UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_notes(thread_id: str):
    st.subheader("Generate Study Notes")

    if st.button("Generate Notes 📝", use_container_width=True):
        with st.spinner("Generating notes from your PDFs..."):
            notes = generate_notes(thread_id)

        if notes is None:
            _no_docs_warning()
            return

        st.session_state[f'notes_{thread_id}'] = notes

    if f'notes_{thread_id}' in st.session_state and st.session_state[f'notes_{thread_id}']:
        notes = st.session_state[f'notes_{thread_id}']

        st.markdown(f"## 📖 {notes.title}")
        st.divider()

        for topic in notes.topics:
            st.markdown(f"### {topic.topic}")
            for bullet in topic.bullets:
                st.markdown(f"- {bullet.point}")
            st.markdown("")

        st.divider()

        pdf_bytes = _notes_to_pdf(notes)
        st.download_button(
            label="⬇️ Download Notes as PDF",
            data=pdf_bytes,
            file_name="study_notes.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter Summary UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_summaries(thread_id: str):
    st.subheader("Summarize Documents")

    if st.button("Summarize PDFs 📄", use_container_width=True):
        with st.spinner("Summarizing your PDFs..."):
            doc_summary = summarize_chapters(thread_id)

        if doc_summary is None:
            _no_docs_warning()
            return

        st.session_state[f'doc_summary_{thread_id}'] = doc_summary

    if f'doc_summary_{thread_id}' in st.session_state and st.session_state[f'doc_summary_{thread_id}']:
        doc_summary = st.session_state[f'doc_summary_{thread_id}']

        for chapter in doc_summary.summaries:
            with st.expander(f"📄 {chapter.source}", expanded=True):
                st.markdown("**Summary:**")
                st.markdown(chapter.summary)
                st.markdown("**Key Points:**")
                for point in chapter.key_points:
                    st.markdown(f"- {point}")

        st.divider()

        pdf_bytes = _summary_to_pdf(doc_summary)
        st.download_button(
            label="⬇️ Download Summaries as PDF",
            data=pdf_bytes,
            file_name="document_summaries.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main render function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render(thread_id: str):
    st.title("📝 Exam Mode")

    if not thread_id:
        _no_docs_warning()
        return

    exam_tab1, exam_tab2, exam_tab3 = st.tabs([
        "🎯 Quiz",
        "📝 Notes",
        "📄 Summaries"
    ])

    with exam_tab1:
        render_quiz(thread_id)

    with exam_tab2:
        render_notes(thread_id)

    with exam_tab3:
        render_summaries(thread_id)