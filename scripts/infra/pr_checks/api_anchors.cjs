// Map source ranges to actual commentable LEFT/RIGHT diff lines.
function diffLines(patch = "") {
  let oldLine = 0,
    newLine = 0,
    active = false;
  const result = [];
  for (const row of patch.split("\n")) {
    const hunk = row.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) {
      oldLine = Number(hunk[1]);
      newLine = Number(hunk[2]);
      active = true;
      continue;
    }
    if (!active) continue;
    if (row.startsWith("-"))
      result.push({ side: "LEFT", line: oldLine++, changed: true });
    else if (row.startsWith("+"))
      result.push({ side: "RIGHT", line: newLine++, changed: true });
    else if (row.startsWith(" ")) {
      result.push(
        { side: "LEFT", line: oldLine++, changed: false },
        { side: "RIGHT", line: newLine++, changed: false },
      );
    }
  }
  return result;
}
function selectAnchor(finding, files) {
  const ranges = finding.anchors?.length
    ? finding.anchors
    : [[finding.path, "RIGHT", finding.line, finding.line]];
  const matches = [];
  for (const [source, side, start, end] of ranges) {
    const file = files.find(
      (f) =>
        f.filename === source ||
        (side === "LEFT" && f.previous_filename === source),
    );
    if (
      !file ||
      !["LEFT", "RIGHT"].includes(side) ||
      !Number.isInteger(start) ||
      !Number.isInteger(end)
    )
      continue;
    for (const line of diffLines(file.patch)) {
      if (line.side === side && line.line >= start && line.line <= end)
        matches.push({ path: file.filename, ...line });
    }
  }
  matches.sort(
    (a, b) =>
      Number(b.changed) - Number(a.changed) ||
      Number(a.side === "LEFT") - Number(b.side === "LEFT") ||
      a.line - b.line,
  );
  if (matches.length) {
    const { path, side, line } = matches[0];
    return { path, side, line };
  }
  // Explicit file comments preserve explanations without inventing an unrelated line.
  const sourcePaths = new Set(ranges.map((r) => r[0]));
  const file = files.find(
    (f) => sourcePaths.has(f.filename) || sourcePaths.has(f.previous_filename),
  );
  return file ? { path: file.filename, subject_type: "file" } : null;
}
module.exports = { selectAnchor, diffLines };
