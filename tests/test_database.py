"""
tests/test_database.py

Unit tests for database.py — all run against an in-memory SQLite database
so no real PostgreSQL connection is needed.

Covered:
    init_db             — happy path, missing URL, bad URL
    get_or_create_session — create, idempotent second call, updated_at touched
    save_message        — user turn, assistant turn with audit fields
    load_session_messages — ordering, doc_id round-trip, empty session
    delete_session_messages — wipes only target session, leaves others intact
    is_db_available     — reflects engine state
    cascade delete      — deleting a session row removes its messages
"""

import importlib
import pytest
import database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite:///:memory:"


def fresh_db():
    """
    Reset the module-level _engine to None, then call init_db with a fresh
    in-memory SQLite URL.  Returns True on success.

    We reload the module so each test group gets its own clean engine —
    SQLAlchemy's StaticPool keeps the in-memory DB alive for the engine's
    lifetime, but a new engine == a new empty DB.
    """
    importlib.reload(database)
    return database.init_db(SQLITE_URL)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:

    def test_happy_path_returns_true(self):
        importlib.reload(database)
        assert database.init_db(SQLITE_URL) is True

    def test_engine_is_available_after_init(self):
        importlib.reload(database)
        database.init_db(SQLITE_URL)
        assert database.is_db_available() is True

    def test_missing_url_returns_false(self, monkeypatch):
        importlib.reload(database)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = database.init_db(database_url=None)
        assert result is False

    def test_bad_url_returns_false(self):
        importlib.reload(database)
        result = database.init_db("postgresql://bad:bad@localhost:1/no_db")
        assert result is False

    def test_engine_not_available_after_bad_url(self):
        importlib.reload(database)
        database.init_db("postgresql://bad:bad@localhost:1/no_db")
        assert database.is_db_available() is False

    def test_idempotent_double_call(self):
        """Calling init_db twice must not raise or corrupt schema."""
        importlib.reload(database)
        assert database.init_db(SQLITE_URL) is True
        assert database.init_db(SQLITE_URL) is True
        assert database.is_db_available() is True


# ---------------------------------------------------------------------------
# get_or_create_session
# ---------------------------------------------------------------------------

class TestGetOrCreateSession:

    def setup_method(self):
        fresh_db()

    def test_creates_new_session(self):
        assert database.get_or_create_session("sess-001") is True

    def test_idempotent_second_call(self):
        database.get_or_create_session("sess-002")
        # Second call must not raise or return False
        assert database.get_or_create_session("sess-002") is True

    def test_updated_at_is_touched_on_second_call(self):
        database.get_or_create_session("sess-003")

        with database._engine.connect() as conn:
            row1 = conn.execute(
                database.sessions_table.select().where(
                    database.sessions_table.c.session_id == "sess-003"
                )
            ).fetchone()
            first_updated = row1._mapping["updated_at"]

        database.get_or_create_session("sess-003")

        with database._engine.connect() as conn:
            row2 = conn.execute(
                database.sessions_table.select().where(
                    database.sessions_table.c.session_id == "sess-003"
                )
            ).fetchone()
            second_updated = row2._mapping["updated_at"]

        assert second_updated >= first_updated

    def test_returns_false_when_db_unavailable(self):
        importlib.reload(database)   # engine is None after reload
        assert database.get_or_create_session("sess-x") is False


# ---------------------------------------------------------------------------
# save_message
# ---------------------------------------------------------------------------

class TestSaveMessage:

    def setup_method(self):
        fresh_db()
        database.get_or_create_session("save-sess")

    def test_save_user_message(self):
        result = database.save_message("save-sess", "user", "Hello there")
        assert result is True

    def test_save_assistant_message_with_audit_fields(self):
        result = database.save_message(
            "save-sess",
            "assistant",
            "Deep breathing helps.",
            retrieved_doc_ids=["abc", "def"],
            latency_seconds=1.23,
        )
        assert result is True

    def test_save_without_optional_fields(self):
        result = database.save_message("save-sess", "user", "Just a message")
        assert result is True

    def test_returns_false_when_db_unavailable(self):
        importlib.reload(database)
        assert database.save_message("save-sess", "user", "x") is False


# ---------------------------------------------------------------------------
# load_session_messages
# ---------------------------------------------------------------------------

class TestLoadSessionMessages:

    def setup_method(self):
        fresh_db()
        database.get_or_create_session("load-sess")

    def test_returns_empty_list_for_new_session(self):
        msgs = database.load_session_messages("load-sess")
        assert msgs == []

    def test_returns_messages_in_insertion_order(self):
        database.save_message("load-sess", "user", "first")
        database.save_message("load-sess", "assistant", "second")
        database.save_message("load-sess", "user", "third")

        msgs = database.load_session_messages("load-sess")
        assert len(msgs) == 3
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        assert msgs[2]["content"] == "third"

    def test_roles_are_preserved(self):
        database.save_message("load-sess", "user", "q")
        database.save_message("load-sess", "assistant", "a")

        msgs = database.load_session_messages("load-sess")
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_retrieved_doc_ids_round_trip(self):
        doc_ids = ["id-1", "id-2", "id-3"]
        database.save_message(
            "load-sess", "assistant", "answer",
            retrieved_doc_ids=doc_ids,
        )
        msgs = database.load_session_messages("load-sess")
        assert msgs[0]["retrieved_doc_ids"] == doc_ids

    def test_none_doc_ids_returns_empty_list(self):
        database.save_message("load-sess", "user", "q", retrieved_doc_ids=None)
        msgs = database.load_session_messages("load-sess")
        assert msgs[0]["retrieved_doc_ids"] == []

    def test_latency_seconds_preserved(self):
        database.save_message(
            "load-sess", "assistant", "a", latency_seconds=0.77
        )
        msgs = database.load_session_messages("load-sess")
        assert msgs[0]["latency_seconds"] == pytest.approx(0.77)

    def test_does_not_return_other_session_messages(self):
        database.get_or_create_session("other-sess")
        database.save_message("other-sess", "user", "not mine")
        database.save_message("load-sess", "user", "mine")

        msgs = database.load_session_messages("load-sess")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "mine"

    def test_returns_empty_list_when_db_unavailable(self):
        importlib.reload(database)
        assert database.load_session_messages("load-sess") == []

    def test_result_contains_expected_keys(self):
        database.save_message("load-sess", "user", "check keys")
        msg = database.load_session_messages("load-sess")[0]
        for key in ("id", "session_id", "role", "content",
                    "retrieved_doc_ids", "latency_seconds", "created_at"):
            assert key in msg, f"missing key: {key}"


# ---------------------------------------------------------------------------
# delete_session_messages
# ---------------------------------------------------------------------------

class TestDeleteSessionMessages:

    def setup_method(self):
        fresh_db()
        database.get_or_create_session("del-sess")
        database.get_or_create_session("keep-sess")

    def test_deletes_messages_for_target_session(self):
        database.save_message("del-sess", "user", "bye")
        database.delete_session_messages("del-sess")
        assert database.load_session_messages("del-sess") == []

    def test_does_not_touch_other_sessions(self):
        database.save_message("del-sess", "user", "bye")
        database.save_message("keep-sess", "user", "stay")
        database.delete_session_messages("del-sess")

        kept = database.load_session_messages("keep-sess")
        assert len(kept) == 1
        assert kept[0]["content"] == "stay"

    def test_safe_on_empty_session(self):
        """Deleting from a session with no messages must not raise."""
        result = database.delete_session_messages("del-sess")
        assert result is True

    def test_returns_false_when_db_unavailable(self):
        importlib.reload(database)
        assert database.delete_session_messages("del-sess") is False


# ---------------------------------------------------------------------------
# is_db_available
# ---------------------------------------------------------------------------

class TestIsDbAvailable:

    def test_false_before_init(self):
        importlib.reload(database)
        assert database.is_db_available() is False

    def test_true_after_successful_init(self):
        importlib.reload(database)
        database.init_db(SQLITE_URL)
        assert database.is_db_available() is True

    def test_false_after_failed_init(self):
        importlib.reload(database)
        database.init_db("postgresql://bad:bad@localhost:1/no")
        assert database.is_db_available() is False


# ---------------------------------------------------------------------------
# Cascade delete — deleting session row removes its messages
# ---------------------------------------------------------------------------

class TestCascadeDelete:

    def setup_method(self):
        fresh_db()

    def test_cascade_on_session_delete(self):
        """
        When the sessions row is deleted, its messages must be removed too
        (ON DELETE CASCADE on the FK).
        """
        database.get_or_create_session("cascade-sess")
        database.save_message("cascade-sess", "user", "will be gone")

        with database._engine.begin() as conn:
            conn.execute(
                database.sessions_table.delete().where(
                    database.sessions_table.c.session_id == "cascade-sess"
                )
            )

        # After cascade, loading messages must return empty
        msgs = database.load_session_messages("cascade-sess")
        assert msgs == []
