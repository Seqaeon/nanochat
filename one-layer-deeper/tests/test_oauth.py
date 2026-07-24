from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException

from client.cli import build_parser, command_login
from service import app as service_app


class OAuthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            public_url="https://one-layer.example",
            github_client_id="client-id",
            github_client_secret="client-secret",
        )

    def test_github_login_requests_only_identity_email_scope(self) -> None:
        with (
            patch.object(service_app, "settings", self.settings),
            patch.object(
                service_app.database,
                "create_oauth_state",
                return_value="one-time-state",
            ),
        ):
            response = service_app.github_login(cli_port=43210)

        location = response.headers["location"]
        parsed = urlparse(location)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.netloc, "github.com")
        self.assertEqual(query["client_id"], ["client-id"])
        self.assertEqual(query["scope"], ["user:email"])
        self.assertEqual(query["state"], ["one-time-state"])
        self.assertEqual(
            query["redirect_uri"],
            ["https://one-layer.example/auth/github/callback"],
        )

    def test_missing_github_configuration_is_unavailable(self) -> None:
        missing = SimpleNamespace(
            public_url="https://one-layer.example",
            github_client_id="",
            github_client_secret="",
        )
        with patch.object(service_app, "settings", missing):
            with self.assertRaisesRegex(
                HTTPException, "GitHub login is not configured"
            ) as raised:
                service_app.github_login()
        self.assertEqual(raised.exception.status_code, 503)


class OAuthCliTests(unittest.TestCase):
    def test_login_command_uses_github_handler(self) -> None:
        args = build_parser().parse_args(["login", "--no-open"])
        self.assertIs(args.handler, command_login)
        self.assertTrue(args.no_open)


if __name__ == "__main__":
    unittest.main()
