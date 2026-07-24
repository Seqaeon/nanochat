from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from service.config import Settings


class SettingsTests(unittest.TestCase):
    def test_github_oauth_settings_are_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PUBLIC_URL": "https://one-layer.example/",
                "GITHUB_CLIENT_ID": "client-id",
                "GITHUB_CLIENT_SECRET": "client-secret",
            },
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.public_url, "https://one-layer.example")
        self.assertEqual(settings.github_client_id, "client-id")
        self.assertEqual(settings.github_client_secret, "client-secret")


if __name__ == "__main__":
    unittest.main()
