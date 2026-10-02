"""
input_guard.py — Pure validation helpers for user input.

Kept framework-agnostic so they can be unit-tested without Streamlit.
streamlit_app.py imports and calls these before passing input to the LLM or DB.
"""

MAX_INPUT_CHARS = 2000


class InputTooLongError(ValueError):
    """Raised when user input exceeds MAX_INPUT_CHARS."""
    pass


class EmptyInputError(ValueError):
    """Raised when user input is empty or whitespace-only."""
    pass


def validate_user_input(text: str) -> str:
    """
    Validate and sanitise a user message before it reaches the RAG chain.

    Rules:
        1. Strip leading/trailing whitespace.
        2. Reject empty / whitespace-only strings.
        3. Reject strings longer than MAX_INPUT_CHARS.

    Args:
        text: Raw input string from the chat widget.

    Returns:
        The stripped input string if valid.

    Raises:
        EmptyInputError:    Input is blank after stripping.
        InputTooLongError:  Input exceeds MAX_INPUT_CHARS characters.
    """
    stripped = text.strip()

    if not stripped:
        raise EmptyInputError("Input must not be empty.")

    if len(stripped) > MAX_INPUT_CHARS:
        raise InputTooLongError(
            f"Input is {len(stripped)} characters — "
            f"maximum allowed is {MAX_INPUT_CHARS}."
        )

    return stripped
