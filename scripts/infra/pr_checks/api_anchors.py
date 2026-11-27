"""Map source ranges to actual commentable LEFT/RIGHT diff lines."""

import re


def diff_lines(patch: str = "") -> list[dict]:
    old_line = new_line = 0
    active = False
    result = []
    for row in patch.split("\n"):
        hunk = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", row)
        if hunk:
            old_line, new_line = map(int, hunk.groups())
            active = True
            continue
        if not active:
            continue
        if row.startswith("-"):
            result.append({"side": "LEFT", "line": old_line, "changed": True})
            old_line += 1
        elif row.startswith("+"):
            result.append({"side": "RIGHT", "line": new_line, "changed": True})
            new_line += 1
        elif row.startswith(" "):
            result.extend(
                [
                    {"side": "LEFT", "line": old_line, "changed": False},
                    {"side": "RIGHT", "line": new_line, "changed": False},
                ]
            )
            old_line += 1
            new_line += 1
    return result


def select_anchor(finding: dict, files: list[dict]) -> dict | None:
    ranges = finding.get("anchors") or [
        [finding.get("path"), "RIGHT", finding.get("line"), finding.get("line")]
    ]
    matches = []
    for source, side, start, end in ranges:
        file = next(
            (
                f
                for f in files
                if (
                    f.get("previous_filename", f["filename"])
                    if side == "LEFT"
                    else f["filename"]
                )
                == source
            ),
            None,
        )
        if (
            file is None
            or side not in ("LEFT", "RIGHT")
            or type(start) is not int
            or type(end) is not int
        ):
            continue
        for line in diff_lines(file.get("patch", "")):
            if line["side"] == side and start <= line["line"] <= end:
                matches.append({"path": file["filename"], **line})
    if matches:
        best = min(
            matches, key=lambda x: (not x["changed"], x["side"] == "LEFT", x["line"])
        )
        return {k: best[k] for k in ("path", "side", "line")}
    # Preserve the explanation without inventing an unrelated line number.
    file = next(
        (
            f
            for f in files
            if any(
                (
                    f.get("previous_filename", f["filename"])
                    if side == "LEFT"
                    else f["filename"]
                )
                == source
                for source, side, _, _ in ranges
            )
        ),
        None,
    )
    return {"path": file["filename"], "subject_type": "file"} if file else None
