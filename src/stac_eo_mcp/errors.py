"""Errors meant to be read by an AI agent.

Every error carries a short message (what went wrong) and an optional hint
(what to try next), so the agent can recover without guessing.
"""


class StacMcpError(Exception):
    """Base error. str() gives 'message Hint: hint'."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} Hint: {self.hint}"
        return self.message


class InvalidInputError(StacMcpError):
    """A tool argument is malformed or out of range."""


class AreaTooLargeError(InvalidInputError):
    """The bbox exceeds the configured maximum area."""


class NoScenesError(StacMcpError):
    """A search matched no scenes."""


class CatalogError(StacMcpError):
    """The STAC catalog failed or returned something unusable."""


class DataUnavailableError(StacMcpError):
    """A scene exists but lacks the assets or valid pixels needed."""
