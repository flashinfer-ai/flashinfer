"""Small REST fake shared by the migrated API workflow tests."""

from copy import deepcopy


class FakeGitHub:
    def __init__(self):
        self.routes = {}
        self.calls = []

    def request(self, method, path, data=None):
        path = path.split("?", 1)[0]
        self.calls.append((method, path, deepcopy(data)))
        handler = self.routes.get((method, path))
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(data)
        if (method, path) not in self.routes:
            raise AssertionError(f"Unexpected API request: {method} {path}")
        return deepcopy(handler)

    def paginate(self, path):
        return self.request("GET", path)
