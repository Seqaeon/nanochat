"""GitHub OAuth exchange and identity lookup."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class GitHubIdentity:
    user_id: int
    login: str
    display_name: str
    email: str


def _selected_email(profile: dict, emails: list[dict]) -> str:
    profile_email = profile.get("email")
    if isinstance(profile_email, str) and profile_email:
        return profile_email.lower()
    for email in emails:
        if email.get("primary") and email.get("verified") and email.get("email"):
            return str(email["email"]).lower()
    for email in emails:
        if email.get("verified") and email.get("email"):
            return str(email["email"]).lower()
    return f"{profile['login']}@users.noreply.github.com".lower()


def exchange_github_identity(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> GitHubIdentity:
    with httpx.Client(timeout=30.0) as client:
        token_response = client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        token_response.raise_for_status()
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise ValueError(
                str(token_payload.get("error_description") or "GitHub did not issue a token")
            )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        profile_response = client.get("https://api.github.com/user", headers=headers)
        profile_response.raise_for_status()
        profile = profile_response.json()
        emails_response = client.get("https://api.github.com/user/emails", headers=headers)
        emails = emails_response.json() if emails_response.is_success else []

    login = str(profile["login"])
    display_name = str(profile.get("name") or login)[:80]
    return GitHubIdentity(
        user_id=int(profile["id"]),
        login=login,
        display_name=display_name,
        email=_selected_email(profile, emails),
    )
