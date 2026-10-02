"""
database.py — Persistent storage layer for the Mental Health RAG Assistant.

Schema:
    sessions  — one row per browser/CLI session (keyed by UUID)
    messages  — one row per conversation turn, linked to a session

Supports PostgreSQL (Supabase in production) and SQLite (local dev / tests).
All public functions are safe to call even when DATABASE_URL is not set;
they return gracefully so a missing DB never crashes the chat.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level engine (initialised once via init_db)
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None
_metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

sessions_table = Table(
    "sessions",
    _metadata,
    Column("session_id", String(36), primary_key=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    ),
)

messages_table = Table(
    "messages",
    _metadata,
    Column("id", String(36), primary_key=True),
    Column(
        "session_id",
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,                     # fast lookup by session
    ),
    Column("role", String(16), nullable=False),          # "user" | "assistant"
    Column("content", Text, nullable=False),
    # JSON on Postgres → JSONB (queryable); falls back to TEXT on SQLite
    Column("retrieved_doc_ids", JSON, nullable=True),
    Column("latency_seconds", Float, nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    ),
)

# Explicit named index — also used by load_session_messages ORDER BY
_idx_messages_session_created = Index(
    "ix_messages_session_created",
    messages_table.c.session_id,
    messages_table.c.created_at,
)

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _create_engine_from_url(database_url: str) -> Engine:
    """
    Build a SQLAlchemy engine.

    pool_pre_ping=True   — handles Supabase idle-connection drops (cheap
                           SELECT 1 before reusing a pooled connection).
    pool_size=5          — Supabase free tier allows ~20 connections total;
                           5 keeps us well inside that limit.
    max_overflow=5       — allow up to 10 total connections under burst load.
    pool_timeout=30      — raise after 30 s if no connection is available
                           rather than hanging forever.
    pool_recycle=1800    — recycle connections every 30 min to avoid
                           server-side idle timeouts.
    """
    connect_args = {}

    # SQLite (used in tests / local fallback) does not support connection
    # pooling parameters — use StaticPool to keep in-memory DB alive.
    # Also enable FK enforcement, which SQLite disables by default.
    if database_url.startswith("sqlite"):
        from sqlalchemy import event
        from sqlalchemy.pool import StaticPool
        connect_args["check_same_thread"] = False
        engine = create_engine(
            database_url,
            connect_args=connect_args,
            poolclass=StaticPool,
        )

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=30,
        pool_recycle=1800,
        connect_args=connect_args,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(database_url: Optional[str] = None) -> bool:
    """
    Initialise the database engine and create tables if they don't exist.

    Call this once at application startup.  Safe to call multiple times.

    Args:
        database_url: SQLAlchemy connection string.  Falls back to the
                      DATABASE_URL environment variable when omitted.

    Returns:
        True  — DB is ready.
        False — DATABASE_URL not set or connection failed (app keeps running).
    """
    global _engine

    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        logger.warning(
            "DATABASE_URL not set — conversation history will not be persisted."
        )
        return False

    try:
        _engine = _create_engine_from_url(url)

        # Verify connectivity before touching schema
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # CREATE TABLE IF NOT EXISTS — idempotent
        _metadata.create_all(_engine)

        logger.info("Database initialised successfully.")
        return True

    except SQLAlchemyError as exc:
        logger.error("Database initialisation failed: %s", exc)
        _engine = None
        return False


def get_or_create_session(session_id: str) -> bool:
    """
    Ensure a row for `session_id` exists in the sessions table.

    Args:
        session_id: UUID string identifying this browser/CLI session.

    Returns:
        True  — row exists (or was just created).
        False — DB not available.
    """
    if _engine is None:
        return False

    try:
        with _engine.begin() as conn:
            existing = conn.execute(
                sessions_table.select().where(
                    sessions_table.c.session_id == session_id
                )
            ).fetchone()

            if existing is None:
                now = datetime.now(timezone.utc)
                conn.execute(
                    sessions_table.insert().values(
                        session_id=session_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                logger.debug("Created new session: %s", session_id)
            else:
                # Touch updated_at so we can order sessions by activity
                conn.execute(
                    sessions_table.update()
                    .where(sessions_table.c.session_id == session_id)
                    .values(updated_at=datetime.now(timezone.utc))
                )

        return True

    except SQLAlchemyError as exc:
        logger.error("get_or_create_session failed: %s", exc)
        return False


def save_message(
    session_id: str,
    role: str,
    content: str,
    retrieved_doc_ids: Optional[list] = None,
    latency_seconds: Optional[float] = None,
) -> bool:
    """
    Persist a single conversation turn to the messages table.

    Args:
        session_id:        UUID of the owning session.
        role:              "user" or "assistant".
        content:           The message text.
        retrieved_doc_ids: List of Pinecone doc IDs returned by the retriever
                           (only meaningful for assistant turns).
        latency_seconds:   Wall-clock time for the RAG chain call.

    Returns:
        True  — row written.
        False — DB not available or write failed.
    """
    if _engine is None:
        return False

    try:
        with _engine.begin() as conn:
            conn.execute(
                messages_table.insert().values(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role=role,
                    content=content,
                    retrieved_doc_ids=retrieved_doc_ids,   # JSON type handles serialisation
                    latency_seconds=latency_seconds,
                    created_at=datetime.now(timezone.utc),
                )
            )
        return True

    except SQLAlchemyError as exc:
        logger.error("save_message failed: %s", exc)
        return False


def load_session_messages(session_id: str) -> list[dict]:
    """
    Fetch all messages for a session, ordered oldest-first.

    Returns:
        List of dicts with keys: id, session_id, role, content,
        retrieved_doc_ids (parsed back to list), latency_seconds, created_at.
        Returns an empty list when the DB is unavailable or session has no history.
    """
    if _engine is None:
        return []

    try:
        with _engine.connect() as conn:
            rows = conn.execute(
                messages_table.select()
                .where(messages_table.c.session_id == session_id)
                .order_by(messages_table.c.created_at)
            ).fetchall()

        result = []
        for row in rows:
            row_dict = row._mapping  # SQLAlchemy 2.x row access
            result.append({
                "id": row_dict["id"],
                "session_id": row_dict["session_id"],
                "role": row_dict["role"],
                "content": row_dict["content"],
                "retrieved_doc_ids": row_dict["retrieved_doc_ids"] or [],  # JSON type: already a list
                "latency_seconds": row_dict["latency_seconds"],
                "created_at": row_dict["created_at"],
            })
        return result

    except SQLAlchemyError as exc:
        logger.error("load_session_messages failed: %s", exc)
        return []


def delete_session_messages(session_id: str) -> bool:
    """
    Delete all messages for a session (used by the 'Clear Chat' button).

    Does NOT delete the session row itself — only the message history.

    Returns:
        True  — rows deleted (or nothing to delete).
        False — DB not available.
    """
    if _engine is None:
        return False

    try:
        with _engine.begin() as conn:
            conn.execute(
                messages_table.delete().where(
                    messages_table.c.session_id == session_id
                )
            )
        return True

    except SQLAlchemyError as exc:
        logger.error("delete_session_messages failed: %s", exc)
        return False


def is_db_available() -> bool:
    """Lightweight check — returns True only if the engine is ready."""
    return _engine is not None
