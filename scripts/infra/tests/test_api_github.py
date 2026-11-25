"""REST transport behavior without live credentials or network calls."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.github_api import GitHub, GitHubError, SameOriginRedirect


class Response(io.BytesIO):
    pass


class GitHubTest(unittest.TestCase):
    def test_json_write_has_types_auth_and_timeout(self):
        api = GitHub("test-token")
        api.opener.open = Mock(return_value=Response(b'{"id":1}'))
        self.assertEqual(
            api.request(
                "POST", "repos/org/repo/statuses/sha", {"enabled": True, "count": 2}
            ),
            {"id": 1},
        )
        args, kwargs = api.opener.open.call_args
        self.assertEqual(kwargs["timeout"], 60)
        request = args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(json.loads(request.data), {"enabled": True, "count": 2})

    def test_paginate_past_one_hundred(self):
        api = GitHub("test-token")
        api.request = Mock(side_effect=[list(range(100)), [100]])
        self.assertEqual(
            api.paginate("repos/org/repo/pulls?state=open"), list(range(101))
        )
        paths = [args[1] for args, kwargs in api.request.call_args_list]
        self.assertIn("state=open&per_page=100&page=2", paths[1])

    def test_http_error_preserves_status_and_204_is_empty(self):
        api = GitHub("test-token")
        api.opener.open = Mock(
            side_effect=HTTPError(
                "https://api.github.com/missing", 404, "missing", {}, None
            )
        )
        with self.assertRaises(GitHubError) as error:
            api.request("GET", "missing")
        self.assertEqual(error.exception.status, 404)
        api.opener.open = Mock(return_value=Response(b""))
        self.assertIsNone(api.request("DELETE", "repos/org/repo/issues/comments/1"))

    def test_cross_origin_requests_and_redirects_cannot_leak_token(self):
        api = GitHub("test-token")
        with self.assertRaises(ValueError):
            api.request("GET", "https://other.example/path")
        request = Mock(full_url="https://api.github.com/repos/org/repo")
        with self.assertRaises(GitHubError):
            SameOriginRedirect().redirect_request(
                request, None, 302, "redirect", {}, "https://other.example/path"
            )

    def test_validation_details_survive_http_error_translation(self):
        errors = [
            {"resource": "PullRequestReviewComment", "field": "line", "code": "invalid"}
        ]
        api = GitHub("test-token")
        api.opener.open = Mock(
            side_effect=HTTPError(
                "https://api.github.com/reviews",
                422,
                "invalid",
                {},
                io.BytesIO(
                    json.dumps(
                        {"message": "Validation Failed", "errors": errors}
                    ).encode()
                ),
            )
        )
        with self.assertRaises(GitHubError) as caught:
            api.request("POST", "repos/org/repo/pulls/3/reviews", {})
        self.assertEqual(caught.exception.errors, errors)
