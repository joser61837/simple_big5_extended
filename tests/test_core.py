import unittest

from simple_big5_extended import Big5ExtendedDecoder, DecodeError
from simple_big5_extended.core import decode, decode_iter


class DecodeBasicTests(unittest.TestCase):
    def test_ascii_passthrough(self):
        self.assertEqual(decode(b"hello"), "hello")

    def test_standard_big5_via_codec(self):
        # 0xA4 0xA4 is the standard Big5 encoding of U+4E2D (CJK "middle").
        self.assertEqual(decode(b"\xa4\xa4"), "\u4e2d")

    def test_empty_input(self):
        self.assertEqual(decode(b""), "")


class ExtensionTests(unittest.TestCase):
    def test_unicode_at_on(self):
        self.assertEqual(decode(b"\x81\x40"), "\ufffd")

    def test_eten_table_entry(self):
        # 0xA1 0xC3 is the ETEN euro sign.
        self.assertEqual(decode(b"\xa1\xc3"), "\u20ac")

    def test_eten_table_entry_inside_text(self):
        self.assertEqual(decode(b"A\xa1\xc4B"), "A\uff0dB")


class ErrorHandlingTests(unittest.TestCase):
    def test_strict_rejects_lone_lead(self):
        with self.assertRaises(DecodeError):
            decode(b"\xa4")

    def test_strict_rejects_invalid_lead(self):
        with self.assertRaises(DecodeError):
            decode(b"\x80")

    def test_strict_rejects_invalid_trail(self):
        with self.assertRaises(DecodeError):
            decode(b"\xa4\x20")

    def test_replace_mode(self):
        self.assertEqual(decode(b"\xa4", errors="replace"), "\ufffd")

    def test_ignore_mode(self):
        self.assertEqual(decode(b"\xa4", errors="ignore"), "")

    def test_invalid_errors_value(self):
        with self.assertRaises(ValueError):
            Big5ExtendedDecoder(errors="bogus")


class IncrementalTests(unittest.TestCase):
    def test_split_sequence_across_calls(self):
        decoder = Big5ExtendedDecoder()
        first = decoder.decode(b"\xa4", final=False)
        second = decoder.decode(b"\xa4", final=True)
        self.assertEqual(first + second, "\u4e2d")

    def test_unicode_at_on_split(self):
        decoder = Big5ExtendedDecoder()
        first = decoder.decode(b"\x81", final=False)
        second = decoder.decode(b"\x40", final=True)
        self.assertEqual(first + second, "\ufffd")

    def test_decode_iter(self):
        chunks = [b"\xa4", b"\xa4\x81", b"\x40"]
        self.assertEqual(decode_iter(chunks), "\u4e2d\ufffd")

    def test_reset_discards_pending(self):
        decoder = Big5ExtendedDecoder()
        decoder.decode(b"\xa4", final=False)
        decoder.reset()
        self.assertEqual(decoder.decode(b"\xa4\xa4", final=True), "\u4e2d")

    def test_final_flush_of_partial_strict(self):
        decoder = Big5ExtendedDecoder()
        decoder.decode(b"\xa4", final=False)
        with self.assertRaises(DecodeError):
            decoder.decode(b"", final=True)


if __name__ == "__main__":
    unittest.main()
