"""Dependency-free GitHub REST client used only by trusted workflow code."""

import json
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str, errors: list | None = None):
        self.status = status
        self.errors = errors or []
        super().__init__(message)


class SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (urlsplit(req.full_url).scheme, urlsplit(req.full_url).netloc) != (
            urlsplit(newurl).scheme,
            urlsplit(newurl).netloc,
        ):
            raise GitHubError(code, "Refusing cross-origin GitHub API redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class GitHub:
    def __init__(self, token: str, api_url: str = "https://api.github.com"):
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.opener = build_opener(SameOriginRedirect())

    def request(self, method: str, path: str, data: dict | None = None):
        if path.startswith(("http:", "https:", "//")):
            raise ValueError("GitHub API paths must be relative")
        request = Request(
            f"{self.api_url}/{path.lstrip('/')}",
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as error:
            try:
                details = json.load(error)
            except (ValueError, TypeError):
                details = {}
            finally:
                error.close()
            errors = details.get("errors", []) if isinstance(details, dict) else []
            raise GitHubError(
                error.code, f"GitHub {method} {path} failed (HTTP {error.code})", errors
            ) from error

    def paginate(self, path: str) -> list:
        result = []
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            items = self.request(
                "GET", path + separator + urlencode({"per_page": 100, "page": page})
            )
            if not isinstance(items, list):
                raise ValueError("Expected a GitHub API list")
            result.extend(items)
            if len(items) < 100:
                return result
            page += 1


def warning(message: str) -> None:
    # Escape untrusted source text before emitting workflow commands.
    escaped = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::warning::{escaped}")


def anchor_validation_error(error: Exception) -> bool:
    """Retry only positively identified diff-anchor errors, never spam limits."""
    errors = getattr(error, "errors", [])
    if (
        getattr(error, "status", None) != 422
        or not isinstance(errors, list)
        or not errors
    ):
        return False
    for item in errors:
        if not isinstance(item, dict):
            return False
        field = item.get("field")
        if (
            item.get("resource") == "PullRequestReviewComment"
            and field
            in {"path", "line", "side", "position", "start_line", "start_side"}
            and item.get("code") in {"invalid", "missing_field", "unprocessable"}
        ):
            continue
        if (
            item.get("resource") == "PullRequestReview"
            and field == "comments"
            and any(
                text in str(item.get("message", "")).lower()
                for text in (
                    "line could not",
                    "path is invalid",
                    "line is invalid",
                    "diff hunk",
                    "position is invalid",
                )
            )
        ):
            continue
        return False
    return True
