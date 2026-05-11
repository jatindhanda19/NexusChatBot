import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from Langgraph_chatbot import chatbot, retrieve_all_threads, save_thread_name, ingest_pdf, thread_document_metadata

# Utility Fuctions

def generate_thread_id():
    """Generate a unique thread ID."""
    return str(uuid.uuid4())

def reset_chat():
    """Create a new chat session."""
    thread_id = generate_thread_id()

    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

    st.session_state.pop('uploaded_pdf_name', None)

    st.rerun()

def add_thread(thread_id):
    """Add a thread to session stae if it does not exist"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    """Load all messages for a specific conversation"""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', []) if state.values else []

# Session state Initilazation

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'processed_pdfs' not in st.session_state:
    st.session_state['processed_pdfs'] = {}

# Current thread Information

tid = st.session_state["thread_id"]
thread_docs = thread_document_metadata(tid)

# Main UI

st.title("🤖 NexusChat")

# sidebar

st.sidebar.title("NexusChat")

if st.sidebar.button('+ New Chat'):
    reset_chat()

# PDF status
if thread_docs:
    st.sidebar.success(
        f"Using '{thread_docs.get('filename')}'"
        f"({thread_docs.get('chunk')} chunks from {thread_docs.get('documents')} pages)"
    )
else:
    st.sidebar.info("NO PDF indexed yet.")

# PDF upload

Uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if Uploaded_pdf:
    tid = st.session_state['thread_id']
    already_processed = st.session_state.get('processed_pdfs').get(tid) == Uploaded_pdf.name
    
    if already_processed:
        st.sidebar.info(f"'{Uploaded_pdf.name}' already processed for this chat.")
    else:
        with st.sidebar.spinner("Processing PDF...."):
            ingest_pdf(Uploaded_pdf, tid)
        st.session_state['processed_pdfs'][tid] = Uploaded_pdf.name
        st.sidebar.success(f"'{Uploaded_pdf.name}' indexed sucessfully.")
        st.rerun()

# Conversation History sidebar

st.sidebar.header("My Conversations")

for tid, name in reversed(list(st.session_state['chat_threads'].items())):
    if st.sidebar.button(name, key=tid):
        st.session_state['thread_id'] = tid
        messages = load_conversation(tid)

        st.session_state['message_history'] = [
            {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.content}
            for m in messages if m.content
        ]
        st.rerun()

# Reset current thread Id after thread switch
tid = st.session_state['thread_id']

# Display Chat messages

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
user_input =  st.chat_input('Type here')

# Handle user messages

if user_input:
    tid = st.session_state["thread_id"]
    
    # Save new thread name on first message

    if tid not in st.session_state['chat_threads']:
        name = " ".join(user_input.split()[:6])
        save_thread_name(tid, name)
        st.session_state['chat_threads'][tid] = name

    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Dispaly user message
    with st.chat_message('user'):
        st.text(user_input)
    
    # LangGraph configuration
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # Generate assistant response
    with st.chat_message('assistant'):
        full_response = ""
        placeholder= st.empty()

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content = user_input)],'thread_id': st.session_state['thread_id']},
            config = CONFIG,
            stream_mode= 'messages'
        ):
                node = metadata.get("langgraph_node")
                
                # show tool usage indicator
                if node == 'tools':
                    placeholder.markdown(f"🔧 *Using {message_chunk.name}...*")
                
                # stream model response
                if isinstance(message_chunk, AIMessage) and message_chunk.content and node == 'chat_node':
                    full_response += message_chunk.content
                    placeholder.markdown(full_response + " ")
        
        # Final response          
        placeholder.markdown(full_response)
        ai_message = full_response
    
    # Save assistant response to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

