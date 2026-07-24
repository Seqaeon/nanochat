"""Operator-only participant administration using the service database URL."""

from __future__ import annotations

import argparse
import json

from .config import Settings
from .db import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m service.admin",
        description="Manage One Layer Deeper participant access.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    users = commands.add_parser("users", help="list registered participants")
    users.set_defaults(action="users")

    ban = commands.add_parser("ban", help="ban a GitHub handle, email, or key prefix")
    ban.add_argument("participant")
    ban.add_argument("--reason", default="operator ban")
    ban.set_defaults(action="ban")

    unban = commands.add_parser(
        "unban", help="restore a GitHub handle, email, or key prefix"
    )
    unban.add_argument("participant")
    unban.set_defaults(action="unban")

    unlimit = commands.add_parser(
        "unlimit",
        help="exempt a trusted participant from attempt and active-run limits",
    )
    unlimit.add_argument("participant")
    unlimit.set_defaults(action="unlimit")

    limit = commands.add_parser(
        "limit", help="restore attempt and active-run limits for a participant"
    )
    limit.add_argument("participant")
    limit.set_defaults(action="limit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = Database(Settings.from_env().database_url)
    database.initialize()
    if args.action == "users":
        print(json.dumps(database.list_users(), indent=2, default=str))
        return
    if args.action in {"unlimit", "limit"}:
        exempt = args.action == "unlimit"
        row = database.set_user_rate_limit_exempt(
            args.participant,
            exempt=exempt,
        )
        if row is None:
            raise SystemExit(f"participant not found: {args.participant}")
        identity = (
            f"@{row['github_login']}" if row.get("github_login") else row["email"]
        )
        state = "unlimited" if exempt else "rate limited"
        print(
            f"{state}: {row['display_name']} "
            f"({identity}, {row['api_key_prefix']})"
        )
        return
    banned = args.action == "ban"
    row = database.set_user_banned(
        args.participant,
        banned=banned,
        reason=args.reason if banned else None,
    )
    if row is None:
        raise SystemExit(f"participant not found: {args.participant}")
    identity = f"@{row['github_login']}" if row.get("github_login") else row["email"]
    print(
        f"{'banned' if banned else 'unbanned'} "
        f"{row['display_name']} ({identity}, {row['api_key_prefix']})"
    )


if __name__ == "__main__":
    main()
