"""Resolve advisory owner hints from a trusted base-branch CODEOWNERS file."""

import re


def pattern_regex(pattern: str) -> re.Pattern:
    rooted = pattern.startswith("/")
    pattern = pattern.lstrip("/")
    directory = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    prefix = "^" if rooted or "/" in pattern else r"(?:^|.*/)"
    result = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern[index : index + 3] == "**/":
            result.append(r"(?:.*/)?")
            index += 3
        elif pattern[index : index + 2] == "**":
            result.append(".*")
            index += 2
        else:
            result.append(
                "[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char)
            )
            index += 1
    # Literal directory names also own their descendants; wildcard file rules
    # such as docs/* match one level, as documented by GitHub.
    suffix = (
        r"(?:/.*)?$"
        if directory or not any(c in pattern.rsplit("/", 1)[-1] for c in "*?")
        else "$"
    )
    return re.compile(prefix + "".join(result) + suffix)


def owners_for_paths(source: str, paths: list[str]) -> list[str]:
    rules = []
    for line in source.splitlines():
        fields = line.split("#", 1)[0].split()
        if (
            not fields
            or fields[0].startswith("!")
            or any(c in fields[0] for c in "[]\\")
        ):
            continue
        owners = [
            field[1:]
            for field in fields[1:]
            if re.fullmatch(r"@[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)?", field)
            and field.lower() != "@flashinfer-bot"
        ]
        rules.append((pattern_regex(fields[0]), owners))
    result = []
    for path in dict.fromkeys(paths):
        owners = []
        for pattern, candidates in rules:
            if pattern.search(path):
                owners = candidates
        for owner in owners:
            if owner.lower() not in {name.lower() for name in result}:
                result.append(owner)
    return result
