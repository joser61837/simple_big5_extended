"""Decoder for Big5 byte sequences with ETEN and Unicode-at-on extensions.

The Big5 encoding as commonly deployed in Taiwan is not a single standard. Two
extensions in particular are widely used:

* ETEN extensions, which fill the user-defined areas of the original Big5
  range with additional CJK characters and symbols.
* The Unicode-at-on extension, which maps Big5 lead byte 0x81, trail byte 0x40
  to U+FFFD (the Unicode replacement character). This is a de facto convention
  used by several browsers and text converters.

This module implements a decoder that follows the Big5 structure and applies
those two extensions. It is deliberately table-free: the standard Big5 range is
handled by the built-in ``big5`` codec, and the extension ranges are handled by
small explicit tables. That keeps the module small and dependency-free while
still covering the bytes that real-world Big5 data actually contains.

Decoding is strict by default: malformed input raises :class:`DecodeError`. A
lenient mode is available for callers that prefer replacement characters.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


class DecodeError(UnicodeDecodeError):
    """Raised when a byte sequence is not valid Big5.

    Subclassing :class:`UnicodeDecodeError` means callers that already handle
    the standard codec error can catch this without importing a new name, while
    still being able to distinguish it if they want to.
    """

    def __init__(self, message: str, data: bytes, start: int, end: int) -> None:
        super().__init__("big5-extended", data, start, end, message)


#: Unicode-at-on maps the single sequence 0x81 0x40 to U+FFFD.
_UNICODE_AT_ON_LEAD = 0x81
_UNICODE_AT_ON_TRAIL = 0x40
_UNICODE_AT_ON_CHAR = "\ufffd"

#: ETEN extension ranges. Each entry maps a (lead, trail) pair to a character.
#: These are the low-frequency additions that the original Big5 codec does not
#: know about. The table is intentionally explicit rather than algorithmic: the
#: ETEN blocks are not contiguous in a way that a simple formula can capture.
_ETEN_TABLE: Dict[Tuple[int, int], str] = {
    (0xA1, 0xC3): "\u20ac",  # euro sign
    (0xA1, 0xC4): "\uFF0D",  # fullwidth hyphen-minus
    (0xA1, 0xC5): "\uFF5E",  # fullwidth tilde
    (0xA1, 0xC6): "\u2016",  # double vertical line
    (0xA1, 0xC7): "\u2018",  # left single quotation mark
    (0xA1, 0xC8): "\u2019",  # right single quotation mark
    (0xA1, 0xC9): "\u201C",  # left double quotation mark
    (0xA1, 0xCA): "\u201D",  # right double quotation mark
    (0xA1, 0xCB): "\u2032",  # prime
    (0xA1, 0xCC): "\u2035",  # reversed prime
    (0xA1, 0xCD): "\uFF3E",  # fullwidth circumflex accent
    (0xA1, 0xCE): "\uFF40",  # fullwidth grave accent
    (0xA1, 0xCF): "\uFFE3",  # fullwidth macron
    (0xA1, 0xD0): "\u02C7",  # caron
    (0xA1, 0xD1): "\u02CA",  # modifier letter acute accent
    (0xA1, 0xD2): "\u02CB",  # modifier letter grave accent
    (0xA1, 0xD3): "\u02D9",  # dot above
    (0xA1, 0xD4): "\u02DD",  # double acute accent
    (0xA1, 0xD5): "\u02DB",  # ogonek
    (0xA1, 0xD6): "\u02DA",  # ring above
    (0xA1, 0xD7): "\u02DC",  # small tilde
    (0xA1, 0xD8): "\u02D8",  # breve
    (0xA1, 0xD9): "\u02C6",  # modifier letter circumflex accent
    (0xA1, 0xDA): "\u02C9",  # modifier letter macron
    (0xA1, 0xDB): "\u02C8",  # modifier letter vertical line
    (0xA1, 0xDC): "\u02CC",  # modifier letter low vertical line
    (0xA1, 0xDD): "\u02CE",  # modifier letter low grave accent
    (0xA1, 0xDE): "\u02CF",  # modifier letter low acute accent
    (0xA1, 0xDF): "\u02D7",  # modifier letter minus sign
    (0xA1, 0xE0): "\u02D0",  # modifier letter triangular colon
    (0xA1, 0xE1): "\u02D1",  # modifier letter half triangular colon
    (0xA1, 0xE2): "\u02D4",  # modifier letter up tack
    (0xA1, 0xE3): "\u02D5",  # modifier letter down tack
    (0xA1, 0xE4): "\u02D6",  # modifier letter plus sign
    (0xA1, 0xE5): "\u02D3",  # modifier letter low left arrow
    (0xA1, 0xE6): "\u02D2",  # modifier letter low right arrow
    (0xA1, 0xE7): "\u02D9",  # dot above (duplicate; kept for completeness)
    (0xA1, 0xE8): "\u02DA",  # ring above (duplicate)
    (0xA1, 0xE9): "\u02DB",  # ogonek (duplicate)
    (0xA1, 0xEA): "\u02DC",  # small tilde (duplicate)
    (0xA1, 0xEB): "\u02DD",  # double acute accent (duplicate)
}


def _is_lead_byte(byte: int) -> bool:
    """Return True when *byte* can start a Big5 double-byte sequence.

    Big5 lead bytes are 0x81 through 0xFE. 0x80 and 0xFF are never valid.
    """

    return 0x81 <= byte <= 0xFE


def _is_trail_byte(byte: int) -> bool:
    """Return True when *byte* can follow a Big5 lead byte.

    Trail bytes occupy 0x40 through 0x7E and 0xA1 through 0xFE. The gap
    0x7F-0xA0 is reserved and never appears in valid Big5.
    """

    return 0x40 <= byte <= 0x7E or 0xA1 <= byte <= 0xFE


class Big5ExtendedDecoder:
    """Decode Big5 bytes with ETEN and Unicode-at-on extension support.

    The decoder accepts bytes and returns text. It can be used incrementally by
    calling :meth:`decode` repeatedly with ``final=False`` and then a final call
    with ``final=True``; bytes that form a partial sequence are buffered between
    calls.

    Args:
        errors: ``"strict"`` (default) raises :class:`DecodeError` on malformed
            input. ``"replace"`` substitutes U+FFFD instead. ``"ignore"`` drops
            the offending bytes.

    The choice to default to ``"strict"`` is deliberate: silent corruption of
    Chinese text is harder to notice than an exception, and callers who want
    lenient behaviour can ask for it explicitly.
    """

    def __init__(self, errors: str = "strict") -> None:
        if errors not in ("strict", "replace", "ignore"):
            raise ValueError("errors must be 'strict', 'replace', or 'ignore'")
        self.errors = errors
        self._pending = bytearray()

    def decode(self, data: bytes, final: bool = True) -> str:
        """Decode *data*, returning the decoded text.

        When *final* is False, an incomplete trailing sequence is buffered and
        the returned string contains only complete characters. When *final* is
        True, any buffered bytes are flushed and an incomplete sequence is
        treated as an error according to :attr:`errors`.
        """

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        buffer = bytes(self._pending) + bytes(data)
        self._pending.clear()
        out: List[str] = []
        i = 0
        n = len(buffer)
        while i < n:
            byte = buffer[i]
            if byte < 0x80:
                out.append(chr(byte))
                i += 1
                continue
            if not _is_lead_byte(byte):
                # 0x80 and 0xFF are invalid anywhere.
                if self.errors == "strict":
                    raise DecodeError("invalid lead byte", buffer, i, i + 1)
                if self.errors == "replace":
                    out.append("\ufffd")
                i += 1
                continue
            if i + 1 >= n:
                # Incomplete sequence at the end of the buffer.
                if final:
                    if self.errors == "strict":
                        raise DecodeError("truncated sequence", buffer, i, n)
                    if self.errors == "replace":
                        out.append("\ufffd")
                    i = n
                else:
                    self._pending.extend(buffer[i:])
                    i = n
                continue
            trail = buffer[i + 1]
            if not _is_trail_byte(trail):
                # The lead byte is valid but the trail byte is not. Treat the
                # lead byte as an error and re-examine the trail byte, since it
                # might itself be a valid ASCII or lead byte.
                if self.errors == "strict":
                    raise DecodeError("invalid trail byte", buffer, i, i + 2)
                if self.errors == "replace":
                    out.append("\ufffd")
                i += 1
                continue
            pair = (byte, trail)
            if byte == _UNICODE_AT_ON_LEAD and trail == _UNICODE_AT_ON_TRAIL:
                out.append(_UNICODE_AT_ON_CHAR)
                i += 2
                continue
            if pair in _ETEN_TABLE:
                out.append(_ETEN_TABLE[pair])
                i += 2
                continue
            # Fall back to the standard library codec for the main Big5 range.
            chunk = buffer[i : i + 2]
            try:
                out.append(chunk.decode("big5"))
            except UnicodeDecodeError:
                if self.errors == "strict":
                    raise DecodeError("unmapped Big5 sequence", buffer, i, i + 2)
                if self.errors == "replace":
                    out.append("\ufffd")
                # For "ignore", append nothing.
            i += 2
        return "".join(out)

    def reset(self) -> None:
        """Discard any buffered partial sequence."""

        self._pending.clear()


def decode(data: bytes, errors: str = "strict") -> str:
    """Decode a complete Big5 byte string in one call.

    This is a convenience wrapper around :class:`Big5ExtendedDecoder` for the
    common case where the whole input is available at once.
    """

    return Big5ExtendedDecoder(errors=errors).decode(data, final=True)


def decode_iter(chunks: Iterable[bytes], errors: str = "strict") -> str:
    """Decode an iterable of byte chunks as a single stream.

    Useful when reading from a file or network socket in fixed-size blocks. The
    decoder buffers partial sequences across chunk boundaries.
    """

    decoder = Big5ExtendedDecoder(errors=errors)
    parts: List[str] = []
    for chunk in chunks:
        parts.append(decoder.decode(chunk, final=False))
    parts.append(decoder.decode(b"", final=True))
    return "".join(parts)
