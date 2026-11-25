const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  renderSummary,
  renderInline,
  groupFindings,
} = require("../pr_checks/api_report.cjs");

const make = (api, classification) => ({
  api,
  classification,
  check: "public_api_signature_changed",
  path: "flashinfer/example.py",
  line: 12,
  before: `def ${api}(x)`,
  after: `def ${api}(x, y)`,
  message: "Old calls omit the required parameter.",
  level: classification === "BREAKING" ? "error" : "warning",
});
const report = {
  execution_status: "COMPLETE",
  findings: [
    make("foo", "BREAKING"),
    make("bar", "BREAKING"),
    make("baz", "LIKELY_BREAKING"),
    make("qux", "LIKELY_BREAKING"),
    make("sampling", "NON_BREAKING"),
  ],
};

test("mixed sticky report lists changes without impact prose", () => {
  const text = renderSummary(report, {
    repo: "org/repo",
    head: "a".repeat(40),
    maintainers: ["alice"],
  });
  assert.match(text, /BLOCKED.*2 breaking.*2 likely breaking/);
  assert.doesNotMatch(text, /non-breaking/);
  assert.match(text, /🚨 API BREAK 🚨 · 2/);
  assert.match(text, /⚠️ API LIKELY BREAK · 2/);
  assert.match(text, /Maintainers @alice can comment/);
  assert.match(text, /Reason: …/);
  assert.doesNotMatch(text, /Old calls omit/);
  assert.equal((text.match(/```diff/g) || []).length, 4);
  assert.match(text, /org\/repo\/blob\/a{40}\/flashinfer\/example.py#L12/);
});
test("inline has impact and correct heading, no global controls", () => {
  assert.match(renderInline(make("foo", "BREAKING")), /🚨 API BREAK 🚨/);
  const warning = renderInline(make("baz", "LIKELY_BREAKING"));
  assert.match(warning, /⚠️ API LIKELY BREAK/);
  assert.match(warning, /Old calls omit/);
  assert.doesNotMatch(warning, /🚨|expected|Maintainers/);
});
test("one API counted at highest severity and evidence retained", () => {
  const grouped = groupFindings([
    make("foo", "LIKELY_BREAKING"),
    make("foo", "BREAKING"),
  ]);
  assert.equal(grouped.length, 1);
  assert.equal(grouped[0].classification, "BREAKING");
});
test("PR source text cannot inject mentions or break markdown fences", () => {
  const finding = make("foo", "BREAKING");
  finding.before = "```\n@someone <script>";
  const text = renderInline(finding);
  assert.doesNotMatch(text, /@someone/);
  assert.equal((text.match(/```/g) || []).length, 2); // HTML stays literal inside an unbroken fence.
});

test("active inline thread is reused after an unrelated push", async () => {
  const { publishReport, inlineKey } = require("../pr_checks/api_report.cjs");
  const item = make("foo", "BREAKING");
  const calls = [];
  const pr = { number: 3, head: { sha: "b".repeat(40) } };
  const github = {
    paginate: async (fn) => fn(),
    rest: {
      issues: { listComments: async () => [], createComment: async () => {} },
      pulls: {
        get: async () => ({ data: pr }),
        listFiles: async () => [
          { filename: item.path, patch: "@@ -10,5 +10,5 @@\n a\n b\n+c\n d" },
        ],
        listReviewComments: async () => [
          {
            id: 7,
            user: { login: "github-actions[bot]" },
            body: `<!-- flashinfer-api-inline --><!-- key:${inlineKey(item)} -->`,
            path: item.path,
            line: 12,
            commit_id: "a".repeat(40),
          },
        ],
        updateReviewComment: async (x) => calls.push(["update", x.comment_id]),
        createReviewComment: async () => calls.push(["create"]),
      },
    },
  };
  await publishReport(
    github,
    { warning() {} },
    pr,
    { execution_status: "COMPLETE", findings: [item] },
    { owner: "org", repo: "repo" },
  );
  assert.deepEqual(calls, [["update", 7]]);
});

test("long signatures show the changed region rather than all unchanged arguments", () => {
  const item = make("foo", "BREAKING");
  const unchanged = Array.from({ length: 60 }, (_, i) => `arg_${i}`).join(", ");
  item.before = `def foo(${unchanged}, y)`;
  item.after = `def foo(${unchanged}, *, y)`;
  const result = renderInline(item);
  assert.match(result, /\*, y/);
  assert.doesNotMatch(result, /arg_0, arg_1/);
  assert.match(result, /…/);
});

test("fenced signature diffs keep native return arrows", () => {
  const item = make("foo", "BREAKING");
  item.before = "def foo(x) -> Tensor";
  item.after = "def foo(x) -> None";
  const body = renderInline(item);
  assert.match(body, /-> Tensor/);
  assert.doesNotMatch(body, /&gt;/);
});

test("multiple inline findings publish in one submitted review", async () => {
  const { publishReport } = require("../pr_checks/api_report.cjs");
  const calls = [];
  const pr = { number: 3, head: { sha: "b".repeat(40) } };
  const github = {
    paginate: async (fn) => fn(),
    rest: {
      issues: { listComments: async () => [], createComment: async () => {} },
      pulls: {
        get: async () => ({ data: pr }),
        listFiles: async () => [
          {
            filename: "flashinfer/example.py",
            patch: "@@ -12 +12 @@\n-old\n+new",
          },
        ],
        listReviewComments: async () => [],
        createReview: async (data) => calls.push(data),
        createReviewComment: async () => {
          throw Error("must batch line comments");
        },
      },
    },
  };
  const result = await publishReport(
    github,
    { warning() {} },
    pr,
    {
      execution_status: "COMPLETE",
      findings: [make("foo", "BREAKING"), make("bar", "LIKELY_BREAKING")],
    },
    { owner: "org", repo: "repo" },
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0].comments.length, 2);
  assert.equal(calls[0].event, "COMMENT");
  assert.equal(result.posted, 2);
  assert.equal(result.line, 2);
  assert.equal(result.failed, 0);
});

test("complete clean reports render no PR comment", () => {
  const options = { repo: "org/repo", head: "a".repeat(40) };
  assert.equal(
    renderSummary({ execution_status: "COMPLETE", findings: [] }, options),
    null,
  );
  assert.equal(
    renderSummary(
      {
        execution_status: "COMPLETE",
        findings: [make("safe", "NON_BREAKING")],
      },
      options,
    ),
    null,
  );
});
test("warnings, expected breaks and incomplete scans remain visible", () => {
  const options = { repo: "org/repo", head: "a".repeat(40) };
  assert.match(
    renderSummary(
      {
        execution_status: "COMPLETE",
        findings: [make("risk", "LIKELY_BREAKING")],
      },
      options,
    ),
    /API LIKELY BREAK/,
  );
  assert.match(
    renderSummary(
      { execution_status: "COMPLETE", findings: [make("risk", "BREAKING")] },
      { ...options, expected: true },
    ),
    /PASS \(expected\)/,
  );
  assert.match(
    renderSummary({ execution_status: "FAILED", findings: [] }, options),
    /INCOMPLETE/,
  );
});
test("clean publication is silent and removes only its own stale summary", async () => {
  const { publishReport } = require("../pr_checks/api_report.cjs");
  const pr = { number: 3, head: { sha: "a".repeat(40) } };
  for (const stale of [false, true]) {
    const calls = [];
    const github = {
      paginate: async (fn) => fn(),
      rest: {
        pulls: {
          get: async () => ({ data: pr }),
          listFiles: async () => {
            throw Error("must not fetch diff for a clean result");
          },
        },
        issues: {
          listComments: async () => [
            {
              id: 1,
              user: { login: "someone" },
              body: "<!-- flashinfer-api-compatibility -->",
            },
            ...(stale
              ? [
                  {
                    id: 2,
                    user: { login: "github-actions[bot]" },
                    body: "<!-- flashinfer-api-compatibility -->old",
                  },
                ]
              : []),
          ],
          deleteComment: async (x) => calls.push(["delete", x.comment_id]),
          createComment: async () => {
            throw Error("must not post PASS");
          },
          updateComment: async () => {
            throw Error("must not replace stale summary with PASS");
          },
        },
      },
    };
    const delivery = await publishReport(
      github,
      { warning() {} },
      pr,
      {
        execution_status: "COMPLETE",
        findings: [make("safe", "NON_BREAKING")],
      },
      { owner: "org", repo: "repo" },
    );
    assert.deepEqual(calls, stale ? [["delete", 2]] : []);
    assert.equal(delivery.total, 0);
    assert.equal(delivery.posted, 0);
  }
});
