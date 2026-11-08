"""Sentence-boundary aggregation of LLM token streams into TTS-ready text units.

Sending single tokens to TTS wrecks prosody and multiplies request overhead; waiting
for the full response wrecks latency. The chunker emits a unit as soon as a linguistic
boundary is *certain*:

* **Sentence end** -- ``.``, ``!``, ``?``, ``…`` (runs such as ``?!`` and ``...``),
  optionally followed by closing quotes or brackets, and then whitespace. The
  whitespace requirement rejects decimals (``3.14``) and dotted forms (``e.g.x``).
  A ``.`` after a known abbreviation (``Dr.``, ``e.g.``) or a capital initial
  (``J. Smith``) is not a boundary.
* **CJK sentence end** -- ideographic full stop and full-width ``!`` / ``?`` (U+3002,
  U+FF01, U+FF1F), plus closers, with no whitespace required.
* **Clause end** -- ``,`` ``;`` ``:`` and the em / en dashes (U+2014, U+2013) followed
  by whitespace, but only once the unit is
  long enough: ``first_clause_min_chars`` for the *first* unit of a response (fast
  first audio) and ``clause_min_chars`` afterwards (natural phrasing).
* **Length cap** -- a unit never exceeds ``max_chars``. Without a boundary it is cut at
  the last whitespace inside the cap, or hard-cut if there is none.

A boundary whose character class depends on text not yet received is deferred, never
guessed, so the output is identical however the stream is split into tokens. The
scanner never revisits a position: total cost is O(characters) per response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, final, override

__all__ = ["ABBREVIATIONS", "ChunkerConfig", "SentenceChunker"]

_TERMINATORS: Final = frozenset(".!?…")
_CJK_TERMINATORS: Final = frozenset("\u3002\uff01\uff1f")
_CLOSERS: Final = frozenset("\"')]}\u201d\u2019\u00bb\u300d\u300f")
_CLAUSE_MARKS: Final = frozenset(",;:\u2014\u2013")
ABBREVIATIONS: Final = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "approx", "inc", "ltd"}
)
_WAIT: Final = -1


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkerConfig:
    """Boundary policy.

    Attributes:
        max_chars: hard cap on a unit's length.
        first_clause_min_chars: minimum length for a clause cut in the first unit.
        clause_min_chars: minimum length for a clause cut in later units.
    """

    max_chars: int = 200
    first_clause_min_chars: int = 16
    clause_min_chars: int = 60

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ValueError: unless ``1 <= clause minimums <= max_chars``.
        """
        if self.max_chars < 1:
            msg = f"max_chars must be positive, got {self.max_chars}"
            raise ValueError(msg)
        for name in ("first_clause_min_chars", "clause_min_chars"):
            if not 1 <= getattr(self, name) <= self.max_chars:
                msg = f"{name} must be in [1, max_chars]"
                raise ValueError(msg)


def _has_speech(text: str) -> bool:
    return any(char.isalnum() for char in text)


@final
class SentenceChunker:
    """Incremental token-to-unit aggregator for one response at a time."""

    __slots__ = ("_buffer", "_config", "_emitted", "_scan")

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Create an empty chunker."""
        self._config = ChunkerConfig() if config is None else config
        self._buffer = ""
        self._scan = 0
        self._emitted = 0

    @property
    def config(self) -> ChunkerConfig:
        """Boundary policy."""
        return self._config

    @property
    def pending(self) -> str:
        """Text received but not yet emitted."""
        return self._buffer

    def feed(self, text: str) -> list[str]:
        """Append an LLM delta and return every unit it completed, in order."""
        self._buffer += text
        return self._drain()

    def flush(self) -> list[str]:
        """End of the response: emit everything left and reset for the next response.

        A remainder without any alphanumeric character (stray punctuation) carries no
        speech and is dropped. A deferred boundary can only sit at the very end of the
        buffer, so emitting the remainder as one unit is exactly what resolving it would
        produce.
        """
        units = self._drain()
        rest = self._buffer.strip()
        self._buffer, self._scan, self._emitted = "", 0, 0
        if _has_speech(rest):
            units.append(rest)
        return units

    @override
    def __repr__(self) -> str:
        return f"SentenceChunker(pending={len(self._buffer)}, emitted={self._emitted})"

    # ------------------------------------------------------------------ scanning

    def _drain(self) -> list[str]:
        buffer, max_chars = self._buffer, self._config.max_chars
        length = len(buffer)
        start = self._skip_space(buffer, 0)
        index = max(self._scan, start)
        units: list[str] = []
        while True:
            cut: int | None = None
            while index < length:
                end = self._boundary(buffer, index, start)
                if end == _WAIT:
                    break
                if end is not None:
                    cut = end
                    break
                index += 1
            if (cut is None and length - start > max_chars) or (
                cut is not None and cut - start > max_chars
            ):
                cut = self._overflow_cut(buffer, start)
            if cut is None:
                break
            units.append(buffer[start:cut].strip())
            self._emitted += 1
            start = self._skip_space(buffer, cut)
            index = max(index, start)
        self._buffer = buffer[start:]
        self._scan = index - start
        return units

    def _boundary(self, buffer: str, index: int, start: int) -> int | None:
        char = buffer[index]
        if char in _TERMINATORS or char in _CJK_TERMINATORS:
            end = self._terminal_end(buffer, index)
        elif char in _CLAUSE_MARKS:
            end = self._clause_end(buffer, index)
            minimum = (
                self._config.first_clause_min_chars
                if self._emitted == 0
                else self._config.clause_min_chars
            )
            if end is not None and end != _WAIT and end - start < minimum:
                end = None
        else:
            return None
        if end is not None and end != _WAIT and not _has_speech(buffer[start:end]):
            return None
        return end

    @staticmethod
    def _skip_space(buffer: str, position: int) -> int:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        return position

    @staticmethod
    def _skip(buffer: str, position: int, chars: frozenset[str]) -> int:
        while position < len(buffer) and buffer[position] in chars:
            position += 1
        return position

    def _terminal_end(self, buffer: str, index: int) -> int | None:
        end = self._skip(buffer, self._skip(buffer, index + 1, _TERMINATORS), _CLOSERS)
        if end == len(buffer):
            return _WAIT
        if buffer[index] in _CJK_TERMINATORS:
            return end
        if not buffer[end].isspace():
            return None
        if buffer[index] == "." and self._is_abbreviation(buffer, index):
            return None
        return end

    def _clause_end(self, buffer: str, index: int) -> int | None:
        end = self._skip(buffer, index + 1, _CLOSERS)
        if end == len(buffer):
            return _WAIT
        return end if buffer[end].isspace() else None

    @staticmethod
    def _is_abbreviation(buffer: str, dot: int) -> bool:
        begin = dot
        while begin > 0 and (buffer[begin - 1].isalpha() or buffer[begin - 1] == "."):
            begin -= 1
        word = buffer[begin:dot]
        if len(word) == 1 and word.isupper():
            return word != "I"  # a capital initial ("J. Smith"), but "I." ends sentences
        return word.lower() in ABBREVIATIONS

    def _overflow_cut(self, buffer: str, start: int) -> int:
        # ``start`` never points at whitespace, so the window holds speech-bearing text.
        limit = start + self._config.max_chars
        window = buffer[start:limit]
        head = window.rstrip()
        if len(head) < len(window):
            return start + len(head)  # the window already ends on a word boundary
        words = head.rsplit(None, 1)
        return start + len(words[0]) if len(words) == 2 else limit  # noqa: PLR2004
