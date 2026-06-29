import streamlit as st
import exam_ui_pdf
from langgraph_backend_multi_hybrid_chroma import (
    chatbot, retrieve_all_threads, ingest,
    save_file_for_thread, get_files_for_thread,
    save_thread_title, get_thread_title
)

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
import uuid



#******************************************* Utility Functions ********************************
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_threads(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['uploaded_files'] = []
    st.session_state['last_seen_files'] = set() 
    save_thread_title(str(thread_id), "New Chat")

def add_threads(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    
def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )

    return state.values.get('messages', [])

#****************************************** Session State **************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = get_files_for_thread(
        str(st.session_state['thread_id']) 
    )

if 'last_seen_files' not in st.session_state:
    st.session_state['last_seen_files'] = set()


add_threads(st.session_state['thread_id'])

#****************************************** SideBar UI **************************************

st.sidebar.title("Study Assistant")

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

file_uploader = st.sidebar.file_uploader(
    "Upload file",
    type=["pdf"],
    accept_multiple_files=True
)

current_files = {f.name for f in file_uploader} if file_uploader else set()
newly_added = current_files - st.session_state['last_seen_files']
st.session_state['last_seen_files'] = current_files

if file_uploader:
    seen_in_this_upload = set()
    for uploaded_file in file_uploader:
        
        # first check — duplicate in same batch
        if uploaded_file.name in seen_in_this_upload:
            st.sidebar.info(f"ℹ️ '{uploaded_file.name}' is already uploaded and indexed.")
            continue
        seen_in_this_upload.add(uploaded_file.name)

        # second check — already processed in previous rerun
        if uploaded_file.name not in newly_added:
            continue

        # third check — already indexed
        if uploaded_file.name not in st.session_state['uploaded_files']:
            result = ingest(
                uploaded_file.getvalue(),
                str(st.session_state["thread_id"]),
                filename=uploaded_file.name
            )
            save_file_for_thread(st.session_state['thread_id'], uploaded_file.name)
            st.session_state['uploaded_files'].append(uploaded_file.name)
            st.sidebar.success(
                f"✅ Indexed {result['pages']} pages, {result['chunks']} new chunks "
                f"(total: {result['total_docs']} chunks across all PDFs)"
            )
        else:
            st.sidebar.info(f"ℹ️ '{uploaded_file.name}' is already uploaded and indexed.")
            
# show loaded PDFs

if st.session_state['uploaded_files']:
    st.sidebar.markdown("**Loaded PDFs:**")
    for name in st.session_state['uploaded_files']:
        st.sidebar.caption(f"📄 {name}")


st.sidebar.title('My Conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    title = get_thread_title(str(thread_id))
    if st.sidebar.button(title, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['uploaded_files'] = get_files_for_thread(thread_id)
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
                content = msg.content
            elif isinstance(msg, AIMessage) and msg.content:
                role = 'assistant'
                content = msg.content
            else:
                continue  # skip ToolMessage, empty AIMessageChunks, etc.

            temp_messages.append({'role': role, 'content': content})
        st.session_state['message_history'] = temp_messages
        st.rerun()

#************************************* Main UI **************************************

# ── tabs at the end ──
user_input = st.chat_input('Type here')
tab1, tab2 = st.tabs(["💬 Chat", "📝 Exam Mode"])


#Load converstion history
with tab1:
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content']) 

    if user_input:

        st.session_state['message_history'].append({'role' : 'user', 'content' : user_input})

        if len(st.session_state['message_history']) == 1:
            title = user_input[:50] + "..." if len(user_input) > 50 else user_input
            save_thread_title(str(st.session_state['thread_id']), title, overwrite=True)

        with st.chat_message('user'):
            st.markdown(user_input)

        CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
        status_holder = {"box": None}
        def ai_stream():
            for msg, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                elif isinstance(msg, AIMessageChunk) and msg.content:  # <-- fix here
                    yield msg.content

        with st.chat_message("assistant"):
            ai_message = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )
        
        st.session_state['message_history'].append({'role' : 'assistant', 'content' : ai_message})
        st.rerun()
    

with tab2:  # ← now correctly at same level as tab1
    exam_ui_pdf.render(str(st.session_state['thread_id']))