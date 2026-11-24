# simple_big5_extended

Decodes Big5 byte sequences including ETEN and Unicode-at-on extension glyph mappings.

```python
from simple_big5_extended import Big5ExtendedDecoder, DecodeError

# One-shot decoding of a complete byte string.
from simple_big5_extended.core import decode

assert decode(b"\xa4\xa4") == "\u4e2d"          # standard Big5
assert decode(b"\xa1\xc3") == "\u20ac"          # ETEN euro sign
assert decode(b"\x81\x40") == "\ufffd"          # Unicode-at-on

# Incremental decoding for streamed input.
decoder = Big5ExtendedDecoder()
assert decoder.decode(b"\xa4", final=False) == ""
assert decoder.decode(b"\xa4", final=True) == "\u4e2d"

# Strict mode raises on malformed input; replace and ignore are also available.
try:
    decode(b"\xa4")
except DecodeError as exc:
    print("bad input at", exc.start)
assert decode(b"\xa4", errors="replace") == "\ufffd"
```

## Why this exists

Big5 as deployed in Taiwan is not one encoding. The original standard leaves
some ranges undefined, and two extensions fill them differently: the ETEN
extensions add symbols and CJK characters in the user-defined areas, and the
Unicode-at-on convention maps the single sequence `0x81 0x40` to U+FFFD. Python's
built-in `big5` codec knows neither, so text that passes through a Taiwanese
browser or legacy converter can fail to decode or silently lose characters.

The trade-off here is table size versus coverage. A complete ETEN table is
thousands of entries; this library ships a small explicit table for the
commonly encountered ETEN symbols and delegates everything else to the standard
`big5` codec. That keeps the module dependency-free and auditable, at the cost
of not covering every obscure ETEN ideograph. The Unicode-at-on mapping is
handled exactly, since it is a single well-known sequence.

## The awkward edge

The decoder is strict by default. A byte sequence that is truncated at the end
of a chunk raises `DecodeError` in `final=True` mode rather than silently
producing replacement characters. If you are reading a stream and cannot tell
whether the last two bytes are complete, call `decode(..., final=False)` and
flush with a final `decode(b"", final=True)` once the stream ends.

## API

- `Big5ExtendedDecoder(errors="strict")` — stateful decoder. `decode(data, final=True)`
  returns `str`; `reset()` clears buffered bytes. `errors` may be `"strict"`,
  `"replace"`, or `"ignore"`.
- `decode(data, errors="strict")` — convenience one-shot function in
  `simple_big5_extended.core`.
- `decode_iter(chunks, errors="strict")` — decodes an iterable of byte chunks as
  one stream.
- `DecodeError` — raised for malformed input in strict mode. It subclasses
  `UnicodeDecodeError`.
