from __future__ import annotations

import unittest

from service.auth import (
    API_KEY_PREFIX,
    api_key_prefix,
    bearer_api_key,
    generate_api_key,
    hash_api_key,
)
from service.github_oauth import _selected_email


class AuthPrimitiveTests(unittest.TestCase):
    def test_keys_are_unique_and_hash_stably(self) -> None:
        first = generate_api_key()
        second = generate_api_key()
        self.assertTrue(first.startswith(API_KEY_PREFIX))
        self.assertNotEqual(first, second)
        self.assertEqual(len(hash_api_key(first)), 64)
        self.assertEqual(hash_api_key(f"  {first}  "), hash_api_key(first))
        self.assertEqual(api_key_prefix(first), first[:12])

    def test_github_email_selection_prefers_verified_primary(self) -> None:
        self.assertEqual(
            _selected_email(
                {"login": "ada", "email": None},
                [
                    {"email": "other@example.com", "verified": True, "primary": False},
                    {"email": "ADA@EXAMPLE.COM", "verified": True, "primary": True},
                ],
            ),
            "ada@example.com",
        )
        self.assertEqual(
            _selected_email({"login": "ada", "email": None}, []),
            "ada@users.noreply.github.com",
        )

    def test_bearer_header_is_explicit(self) -> None:
        self.assertEqual(bearer_api_key("Bearer old_example"), "old_example")
        with self.assertRaisesRegex(ValueError, "missing API key"):
            bearer_api_key(None)
        with self.assertRaisesRegex(ValueError, "Authorization"):
            bearer_api_key("old_example")


if __name__ == "__main__":
    unittest.main()
