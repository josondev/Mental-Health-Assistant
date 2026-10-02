"""
tests/test_app_db_integration.py

Integration tests for the DB wiring inside src/app.py.

What we test:
    - ask_question_for_evaluation writes user + assistant rows when session_id given
    - ask_question_for_evaluation returns latency_seconds in its result dict
    - ask_question_for_evaluation does NOT write to DB when session_id is None
    - ask_question_streamed writes user + assistant rows when session_id given
    - ask_question_streamed does NOT write to DB when session_id is None
    - a DB failure inside save_message never raises out to the caller

Strategy:
    app.py imports Pinecone, LangChain, and Gemini at module level.
    We patch all of those with lightweight fakes before importing the module
    so no real API credentials are needed.  The RAG chain itself is replaced
    with a simple stub that returns a fixed answer and a fake context doc.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import database

# ---------------------------------------------------------------------------
# SQLite URL used for all tests in this file
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite:///:memory:"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db_for_each_test():
    """Reset the database module to a clean in-memory state before every test."""
    importlib.reload(database)
    database.init_db(SQLITE_URL)
    yield
    importlib.reload(database)


@pytest.fixture()
def app_module():
    """
    Import src/app.py with all external dependencies stubbed out.

    Returns the imported module so tests can call its functions directly.
    """
    # --- build fake modules that satisfy app.py's top-level imports ---

    # langchain_huggingface
    lc_hf = types.ModuleType("langchain_huggingface")
    mock_embeddings_cls = MagicMock()
    mock_embeddings_cls.return_value.model_name = "all-MiniLM-L6-v2"
    lc_hf.HuggingFaceEmbeddings = mock_embeddings_cls
    sys.modules["langchain_huggingface"] = lc_hf

    # pinecone
    pinecone_mod = types.ModuleType("pinecone")
    pinecone_mod.Pinecone = MagicMock()
    pinecone_mod.ServerlessSpec = MagicMock()
    pinecone_mod.PineconeApiException = Exception
    sys.modules["pinecone"] = pinecone_mod

    # langchain_google_genai
    lc_google = types.ModuleType("langchain_google_genai")
    lc_google.ChatGoogleGenerativeAI = MagicMock()
    sys.modules["langchain_google_genai"] = lc_google

    # langchain.chains.combine_documents
    lc_cbd = types.ModuleType("langchain.chains.combine_documents")
    lc_cbd.create_stuff_documents_chain = MagicMock()
    sys.modules["langchain"] = types.ModuleType("langchain")
    sys.modules["langchain.chains"] = types.ModuleType("langchain.chains")
    sys.modules["langchain.chains.combine_documents"] = lc_cbd

    # langchain_core.prompts / messages
    lc_core = types.ModuleType("langchain_core")
    lc_prompts = types.ModuleType("langchain_core.prompts")
    lc_prompts.ChatPromptTemplate = MagicMock()
    lc_prompts.MessagesPlaceholder = MagicMock()
    lc_msgs = types.ModuleType("langchain_core.messages")
    lc_msgs.HumanMessage = MagicMock(side_effect=lambda content: {"role": "user", "content": content})
    lc_msgs.AIMessage = MagicMock(side_effect=lambda content: {"role": "assistant", "content": content})
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.prompts"] = lc_prompts
    sys.modules["langchain_core.messages"] = lc_msgs

    # langchain.chains (create_history_aware_retriever etc.)
    lc_chains = sys.modules["langchain.chains"]
    lc_chains.create_history_aware_retriever = MagicMock()
    lc_chains.create_retrieval_chain = MagicMock()

    # langchain_pinecone
    lc_pinecone = types.ModuleType("langchain_pinecone")
    lc_pinecone.PineconeVectorStore = MagicMock()
    sys.modules["langchain_pinecone"] = lc_pinecone

    # dotenv
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = MagicMock()
    sys.modules["dotenv"] = dotenv_mod

    # Remove cached app module so we get a fresh import
    sys.modules.pop("app", None)

    # Set dummy env vars so Pinecone/Gemini constructors don't raise
    with patch.dict("os.environ", {
        "PINECONE_API_KEY": "fake-key",
        "PINECONE_CLOUD": "aws",
        "PINECONE_REGION": "us-east-1",
        "GOOGLE_API_KEY": "fake-google-key",
        "DATABASE_URL": SQLITE_URL,
    }):
        app = importlib.import_module("app")

    # --- replace rag_chain with a stub that returns known values ---
    fake_doc = MagicMock()
    fake_doc.page_content = "Grounding techniques help with anxiety."
    fake_doc.metadata = {"doc_id": "stub-doc-001"}

    def fake_invoke(payload):
        return {
            "answer": "Try deep breathing.",
            "context": [fake_doc],
        }

    def fake_stream(payload):
        # Yield two chunks then done
        yield {"answer": "Try "}
        yield {"answer": "deep breathing."}

    stub_chain = MagicMock()
    stub_chain.invoke.side_effect = fake_invoke
    stub_chain.stream.side_effect = fake_stream
    app.rag_chain = stub_chain

    return app


# ---------------------------------------------------------------------------
# ask_question_for_evaluation — DB wiring
# ---------------------------------------------------------------------------

class TestAskQuestionForEvaluationDb:

    def test_writes_two_rows_when_session_id_given(self, app_module):
        sid = "eval-sess-001"
        database.get_or_create_session(sid)

        app_module.ask_question_for_evaluation(
            "What helps with anxiety?",
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assert len(msgs) == 2

    def test_user_row_saved_correctly(self, app_module):
        sid = "eval-sess-002"
        database.get_or_create_session(sid)

        app_module.ask_question_for_evaluation(
            "I feel anxious.",
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        user_msg = msgs[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "I feel anxious."

    def test_assistant_row_saved_correctly(self, app_module):
        sid = "eval-sess-003"
        database.get_or_create_session(sid)

        app_module.ask_question_for_evaluation(
            "What is grounding?",
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Try deep breathing."

    def test_retrieved_doc_ids_persisted(self, app_module):
        sid = "eval-sess-004"
        database.get_or_create_session(sid)

        app_module.ask_question_for_evaluation(
            "Grounding techniques?",
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assistant_msg = msgs[1]
        assert "stub-doc-001" in assistant_msg["retrieved_doc_ids"]

    def test_latency_seconds_persisted(self, app_module):
        sid = "eval-sess-005"
        database.get_or_create_session(sid)

        app_module.ask_question_for_evaluation(
            "How do I manage stress?",
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assistant_msg = msgs[1]
        assert assistant_msg["latency_seconds"] is not None
        assert assistant_msg["latency_seconds"] >= 0.0

    def test_result_contains_latency_seconds_key(self, app_module):
        result = app_module.ask_question_for_evaluation("Test question")
        assert "latency_seconds" in result
        assert result["latency_seconds"] >= 0.0

    def test_no_db_write_when_session_id_is_none(self, app_module):
        """Without a session_id nothing should be written."""
        sid = "eval-sess-no-write"
        # Don't create session — just confirm no rows appear
        app_module.ask_question_for_evaluation(
            "Silent question",
            session_id=None,
        )
        msgs = database.load_session_messages(sid)
        assert msgs == []

    def test_result_structure_is_complete(self, app_module):
        result = app_module.ask_question_for_evaluation("Check keys")
        for key in ("answer", "documents", "retrieved_doc_ids", "latency_seconds"):
            assert key in result, f"missing key: {key}"

    def test_db_failure_does_not_raise(self, app_module, monkeypatch):
        """If save_message raises, the caller must not see the exception."""
        monkeypatch.setattr(database, "save_message", MagicMock(side_effect=RuntimeError("DB down")))
        sid = "eval-sess-failure"
        # Should complete without raising
        result = app_module.ask_question_for_evaluation(
            "Will this crash?",
            session_id=sid,
        )
        assert result["answer"] == "Try deep breathing."


# ---------------------------------------------------------------------------
# ask_question_streamed — DB wiring
# ---------------------------------------------------------------------------

class TestAskQuestionStreamedDb:

    def test_writes_two_rows_when_session_id_given(self, app_module):
        sid = "stream-sess-001"
        database.get_or_create_session(sid)

        app_module.ask_question_streamed(
            "I feel overwhelmed.",
            current_chat_history=[],
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assert len(msgs) == 2

    def test_user_row_role_and_content(self, app_module):
        sid = "stream-sess-002"
        database.get_or_create_session(sid)

        app_module.ask_question_streamed(
            "My question here.",
            current_chat_history=[],
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "My question here."

    def test_assistant_row_role_and_content(self, app_module):
        sid = "stream-sess-003"
        database.get_or_create_session(sid)

        app_module.ask_question_streamed(
            "Any question.",
            current_chat_history=[],
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assert msgs[1]["role"] == "assistant"
        # stub yields "Try " + "deep breathing."
        assert msgs[1]["content"] == "Try deep breathing."

    def test_latency_persisted_for_streamed(self, app_module):
        sid = "stream-sess-004"
        database.get_or_create_session(sid)

        app_module.ask_question_streamed(
            "Latency check.",
            current_chat_history=[],
            session_id=sid,
        )

        msgs = database.load_session_messages(sid)
        assert msgs[1]["latency_seconds"] is not None
        assert msgs[1]["latency_seconds"] >= 0.0

    def test_no_db_write_without_session_id(self, app_module):
        sid = "stream-sess-no-write"
        app_module.ask_question_streamed(
            "Silent.",
            current_chat_history=[],
            session_id=None,
        )
        msgs = database.load_session_messages(sid)
        assert msgs == []

    def test_returns_full_assembled_response(self, app_module):
        response = app_module.ask_question_streamed(
            "Assemble check.",
            current_chat_history=[],
        )
        assert response == "Try deep breathing."

    def test_db_failure_does_not_raise(self, app_module, monkeypatch):
        monkeypatch.setattr(database, "save_message", MagicMock(side_effect=RuntimeError("DB down")))
        sid = "stream-sess-failure"
        # Must not raise
        response = app_module.ask_question_streamed(
            "Will this crash?",
            current_chat_history=[],
            session_id=sid,
        )
        assert response == "Try deep breathing."
