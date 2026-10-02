"""
tests/test_input_guard.py

Unit tests for src/input_guard.py — validate_user_input().

Covers:
    happy path          — normal valid messages pass through
    stripping           — leading/trailing whitespace is removed
    empty input         — blank and whitespace-only strings raise EmptyInputError
    boundary lengths    — exactly MAX_INPUT_CHARS is accepted,
                          MAX_INPUT_CHARS + 1 is rejected
    over-length         — messages above the limit raise InputTooLongError
    error messages      — exceptions carry informative text
"""

import pytest
from input_guard import (
    MAX_INPUT_CHARS,
    EmptyInputError,
    InputTooLongError,
    validate_user_input,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestValidInput:

    def test_normal_message_passes(self):
        msg = "I have been feeling anxious lately."
        assert validate_user_input(msg) == msg

    def test_single_word_passes(self):
        assert validate_user_input("anxiety") == "anxiety"

    def test_returns_stripped_string(self):
        result = validate_user_input("  hello  ")
        assert result == "hello"

    def test_strips_leading_whitespace(self):
        result = validate_user_input("   leading")
        assert result == "leading"

    def test_strips_trailing_whitespace(self):
        result = validate_user_input("trailing   ")
        assert result == "trailing"

    def test_strips_newlines(self):
        result = validate_user_input("\n\nsome text\n")
        assert result == "some text"

    def test_exactly_max_chars_is_accepted(self):
        msg = "a" * MAX_INPUT_CHARS
        result = validate_user_input(msg)
        assert result == msg
        assert len(result) == MAX_INPUT_CHARS

    def test_one_char_message_passes(self):
        assert validate_user_input("?") == "?"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:

    def test_empty_string_raises(self):
        with pytest.raises(EmptyInputError):
            validate_user_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyInputError):
            validate_user_input("   ")

    def test_newlines_only_raises(self):
        with pytest.raises(EmptyInputError):
            validate_user_input("\n\n\n")

    def test_tabs_only_raises(self):
        with pytest.raises(EmptyInputError):
            validate_user_input("\t\t")

    def test_empty_error_is_value_error_subclass(self):
        with pytest.raises(ValueError):
            validate_user_input("")


# ---------------------------------------------------------------------------
# Over-length input
# ---------------------------------------------------------------------------

class TestOverLengthInput:

    def test_one_over_max_raises(self):
        msg = "a" * (MAX_INPUT_CHARS + 1)
        with pytest.raises(InputTooLongError):
            validate_user_input(msg)

    def test_far_over_max_raises(self):
        msg = "x" * 100_000
        with pytest.raises(InputTooLongError):
            validate_user_input(msg)

    def test_error_message_contains_char_count(self):
        msg = "a" * (MAX_INPUT_CHARS + 50)
        with pytest.raises(InputTooLongError) as exc_info:
            validate_user_input(msg)
        assert str(MAX_INPUT_CHARS + 50) in str(exc_info.value)

    def test_error_message_contains_limit(self):
        msg = "a" * (MAX_INPUT_CHARS + 1)
        with pytest.raises(InputTooLongError) as exc_info:
            validate_user_input(msg)
        assert str(MAX_INPUT_CHARS) in str(exc_info.value)

    def test_too_long_error_is_value_error_subclass(self):
        msg = "z" * (MAX_INPUT_CHARS + 1)
        with pytest.raises(ValueError):
            validate_user_input(msg)

    def test_whitespace_pad_does_not_bypass_limit(self):
        """
        Stripping happens before length check, so padding with spaces
        cannot shrink a genuinely long message below the limit.
        """
        core = "a" * (MAX_INPUT_CHARS + 10)
        padded = "   " + core + "   "
        with pytest.raises(InputTooLongError):
            validate_user_input(padded)

    def test_whitespace_pad_can_bring_valid_message_in(self):
        """
        Conversely, if core content is within limit, surrounding whitespace
        must not cause a false rejection.
        """
        core = "a" * MAX_INPUT_CHARS
        padded = "  " + core + "  "
        result = validate_user_input(padded)
        assert result == core


# ---------------------------------------------------------------------------
# Boundary: MAX_INPUT_CHARS constant is sane
# ---------------------------------------------------------------------------

class TestConstant:

    def test_max_input_chars_is_positive(self):
        assert MAX_INPUT_CHARS > 0

    def test_max_input_chars_is_reasonable(self):
        # Should be at least 500 to allow meaningful messages,
        # but not so large it defeats the guard purpose.
        assert 500 <= MAX_INPUT_CHARS <= 10_000
