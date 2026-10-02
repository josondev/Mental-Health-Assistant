import streamlit as st
import os
import uuid
import sys
import time
import logging
from pathlib import Path

# Configure logging once at entrypoint — database.py and other modules
# use logging.getLogger so their messages are now visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Ensure src/ is on the path so database.py is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
import database as db
from input_guard import validate_user_input, InputTooLongError, EmptyInputError

# Page config
st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  /* App background */
  .stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }

  section.main > div {
    max-width: 100%;
    padding: 1rem;
  }

  /* Base chat message box styling (common) */
  div[data-testid="stChatMessage"]{
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12);
  }

  /* DARK MODE */
  @media (prefers-color-scheme: dark) {
    div[data-testid="stChatMessage"]{
      background: #0b0b0b !important;
    }

    /* Target the actual rendered text inside chat messages */
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #f7fafc !important;
    }
  }

  /* LIGHT MODE */
  @media (prefers-color-scheme: light) {
    div[data-testid="stChatMessage"]{
      background: #ffffff !important;
    }

    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #111827 !important;
    }
  }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot and database (cached to avoid reloading)"""
    try:
        # Get API keys from Streamlit secrets
        pinecone_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
        google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        pinecone_cloud = st.secrets.get("PINECONE_CLOUD", os.getenv("PINECONE_CLOUD", "aws"))
        pinecone_region = st.secrets.get("PINECONE_REGION", os.getenv("PINECONE_REGION", "us-east-1"))
        database_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
        gemini_model = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

        if not pinecone_key or not google_key:
            st.error("⚠️ API keys not found. Please add them to Streamlit secrets.")
            st.stop()

        # Initialise DB — non-blocking, failure just means no persistence
        db_ready = db.init_db(database_url)

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_key)
        spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Pinecone index
        index_name = "medical-chatbot"
        try:
            pc.Index(index_name).describe_index_stats()
        except PineconeApiException:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=spec
            )

        # Vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0.1,
            max_tokens=2048,        # raised from 1024 — prevents mid-sentence cutoff
            request_timeout=60,     # Streamlit Cloud proxy timeout is 30s by default; give Gemini room
            api_key=google_key,
        )

        # Contextualize question prompt
        contextualize_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualized_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=base_retriever,
            prompt=contextualized_q_prompt,
        )

        # QA prompt
        system_prompt = (
            "You are a compassionate mental health assistant. "
            "Use the following context to answer the question. "
            "Focus ONLY on mental health topics. "
            "If you don't know, say so. Be empathetic and supportive."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Final RAG chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

        return rag_chain, db_ready

    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None, False

# Initialize chatbot
if st.session_state.rag_chain is None:
    with st.spinner("🔄 Loading AI assistant..."):
        rag_chain, db_ready = initialize_chatbot()
        st.session_state.rag_chain = rag_chain
        st.session_state.db_ready = db_ready

# Restore conversation history from DB (runs once per browser session)
if not st.session_state.history_loaded and st.session_state.db_ready:
    session_id = st.session_state.session_id
    db.get_or_create_session(session_id)
    past_messages = db.load_session_messages(session_id)

    if past_messages:
        for msg in past_messages:
            # Rebuild the Streamlit display list
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
            # Rebuild the LangChain chat_history list for the RAG chain
            if msg["role"] == "user":
                st.session_state.chat_history.append(
                    HumanMessage(content=msg["content"])
                )
            else:
                st.session_state.chat_history.append(
                    AIMessage(content=msg["content"])
                )

    st.session_state.history_loaded = True

# Header
st.title("🧠 Mental Health Assistant")
st.caption("Your compassionate AI companion for mental health support")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI assistant provides mental health support using:
    - **RAG** (Retrieval Augmented Generation)
    - **Google Gemini** for natural language understanding
    - **Pinecone** for knowledge retrieval

    💡 **Tips:**
    - Ask questions about stress, anxiety, depression
    - The AI remembers your conversation context
    - This is not a replacement for professional help
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        # Wipe DB messages for this session and assign a fresh session_id
        if st.session_state.db_ready:
            db.delete_session_messages(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history_loaded = False
        st.rerun()

    st.divider()

    # DB health indicator
    if st.session_state.db_ready:
        st.caption("💾 **History:** saved to database")
    else:
        st.caption("⚠️ **History:** local only (no DATABASE_URL)")

    st.caption("⚠️ **Disclaimer:** This is an AI assistant. For emergencies, contact professional help immediately.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Check if chatbot is initialized
    if st.session_state.rag_chain is None:
        st.error("⚠️ Chatbot not initialized. Please check your API keys.")
        st.stop()

    # Validate input — rejects empty and over-length messages before
    # they reach the LLM or DB
    try:
        prompt = validate_user_input(prompt)
    except EmptyInputError:
        st.stop()
    except InputTooLongError as exc:
        st.warning(f"⚠️ {exc}")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        try:
            start_time = time.perf_counter()

            # Stream tokens as they arrive — no more long "Thinking..." wait
            response_placeholder = st.empty()
            full_response = ""
            retrieved_doc_ids = []

            for chunk in st.session_state.rag_chain.stream({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            }):
                # Accumulate answer tokens and re-render with blinking cursor
                if chunk.get("answer"):
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")

                # Capture doc IDs from the context chunk (arrives before answer)
                if chunk.get("context") and not retrieved_doc_ids:
                    retrieved_doc_ids = [
                        doc.metadata.get("doc_id")
                        or doc.metadata.get("id")
                        or doc.metadata.get("source")
                        or ""
                        for doc in chunk["context"]
                    ]

            # Final render — remove blinking cursor
            response_placeholder.markdown(full_response)

            latency = round(time.perf_counter() - start_time, 4)
            response = full_response or "I'm sorry, I couldn't generate a response."

            # Persist both turns to DB
            if st.session_state.db_ready:
                try:
                    db.save_message(
                        st.session_state.session_id,
                        "user",
                        prompt,
                    )
                    db.save_message(
                        st.session_state.session_id,
                        "assistant",
                        response,
                        retrieved_doc_ids=retrieved_doc_ids,
                        latency_seconds=latency,
                    )
                except Exception:
                    pass  # DB failure must never crash the chat

            # Update in-memory chat history for the RAG chain
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))

            # Add to display messages
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
