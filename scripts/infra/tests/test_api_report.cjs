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
  assert.match(text, /BLOCKED.*2 breaking.*2 likely breaking.*1 non-breaking/);
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
  assert.doesNotMatch(text, /@someone|<script>/);
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
