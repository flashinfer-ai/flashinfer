const { test } = require("node:test");
const assert = require("node:assert/strict");
const { selectAnchor } = require("../pr_checks/api_anchors.cjs");

test("multiline signature attaches to changed argument within its range", () => {
  const f = {
    path: "api.py",
    line: 200,
    anchors: [
      ["api.py", "RIGHT", 200, 220],
      ["api.py", "LEFT", 200, 221],
    ],
  };
  const files = [
    {
      filename: "api.py",
      patch: "@@ -208,3 +208,3 @@\n x,\n-old=0,\n+new=0,\n ):",
    },
  ];
  assert.deepEqual(selectAnchor(f, files), {
    path: "api.py",
    line: 209,
    side: "RIGHT",
  });
});
test("removed parameter uses LEFT changed line before RIGHT context", () => {
  const f = {
    anchors: [
      ["api.py", "RIGHT", 1, 6],
      ["api.py", "LEFT", 1, 7],
    ],
  };
  const files = [
    { filename: "api.py", patch: "@@ -3,3 +3,2 @@\n x,\n-removed=0,\n ):" },
  ];
  assert.deepEqual(selectAnchor(f, files), {
    path: "api.py",
    line: 4,
    side: "LEFT",
  });
});
test("deleted API uses old signature on LEFT", () => {
  const f = { anchors: [["api.py", "LEFT", 10, 12]] };
  const files = [
    {
      filename: "api.py",
      status: "removed",
      patch: "@@ -10,3 +9,0 @@\n-def foo(\n- x,\n-): pass",
    },
  ];
  assert.deepEqual(selectAnchor(f, files), {
    path: "api.py",
    line: 10,
    side: "LEFT",
  });
});
test("renamed file maps old path to the current diff filename", () => {
  const f = { anchors: [["old.py", "LEFT", 10, 12]] };
  const files = [
    {
      filename: "new.py",
      previous_filename: "old.py",
      patch: "@@ -10,2 +10,2 @@\n-old\n+new\n context",
    },
  ];
  assert.deepEqual(selectAnchor(f, files), {
    path: "new.py",
    line: 10,
    side: "LEFT",
  });
});
test("never attaches a signature finding to an unrelated changed line", () => {
  const f = { path: "api.py", anchors: [["api.py", "RIGHT", 2, 8]] };
  const files = [{ filename: "api.py", patch: "@@ -100 +100 @@\n-old\n+new" }];
  assert.deepEqual(selectAnchor(f, files), {
    path: "api.py",
    subject_type: "file",
  });
});
test("missing patch uses an explicit file review comment", () => {
  assert.deepEqual(
    selectAnchor({ path: "api.py", line: 12 }, [{ filename: "api.py" }]),
    { path: "api.py", subject_type: "file" },
  );
});
test("no comment on an unchanged unrelated file", () => {
  assert.equal(
    selectAnchor({ path: "api.py", line: 12 }, [
      { filename: "other.py", patch: "@@ -1 +1 @@\n-a\n+b" },
    ]),
    null,
  );
});

test("real #8953 and #8910 missed-inline regressions get valid line anchors", () => {
  const fixtures = require("./api_inline_real_prs.json");
  assert.equal(fixtures.length, 3);
  for (const { finding, file, upstream_pr } of fixtures) {
    const anchor = selectAnchor(finding, [file]);
    assert.ok(
      anchor?.line,
      `${upstream_pr}: ${finding.api} needs a real diff line`,
    );
    assert.notEqual(anchor.subject_type, "file");
  }
});
