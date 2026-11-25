// Compact API-only reports. PR source text is untrusted even in a trusted scan.
const crypto = require("node:crypto");
const MARKER = "<!-- flashinfer-api-compatibility -->";
const INLINE = "<!-- flashinfer-api-inline -->";
const rank = { NON_BREAKING: 0, LIKELY_BREAKING: 1, BREAKING: 2 };
const text = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/@/g, "@\u200b")
    .replace(/`/g, "ˋ")
    .replace(/[\r\n]/g, " ")
    .slice(0, 2500);
const heading = (finding) =>
  finding.classification === "BREAKING"
    ? "🚨 API BREAK 🚨"
    : "⚠️ API LIKELY BREAK";

function groupFindings(findings) {
  const groups = new Map();
  for (const finding of findings) {
    const key = finding.api || `${finding.path}:${finding.check}`;
    const previous = groups.get(key);
    if (!previous) groups.set(key, { ...finding });
    else {
      const higher =
        rank[finding.classification] > rank[previous.classification]
          ? finding
          : previous;
      groups.set(key, {
        ...higher,
        message: [...new Set([previous.message, finding.message])].join(" "),
      });
    }
  }
  return [...groups.values()];
}
function diff(finding) {
  const before = String(finding.before || "");
  const after = String(finding.after || "");
  let prefix = 0;
  let suffix = 0;
  if (Math.max(before.length, after.length) > 220 && before !== after) {
    while (
      prefix < Math.min(before.length, after.length) &&
      before[prefix] === after[prefix]
    )
      prefix++;
    while (
      suffix < Math.min(before.length, after.length) - prefix &&
      before[before.length - suffix - 1] === after[after.length - suffix - 1]
    )
      suffix++;
  }
  const start = prefix > 120 ? prefix - 70 : 0;
  const trimEnd = suffix > 120 ? suffix - 70 : 0;
  const fragment = (value) =>
    `${start ? "…" : ""}${text(value.slice(start, value.length - trimEnd))}${trimEnd ? "…" : ""}`;
  return [
    "```diff",
    `- ${fragment(before)}`,
    `+ ${fragment(after)}`,
    "```",
  ].join("\n");
}
function inlineKey(finding) {
  return crypto
    .createHash("sha256")
    .update(finding.api || `${finding.path}:${finding.check}`)
    .digest("hex");
}
function renderInline(finding) {
  return `${INLINE}<!-- key:${inlineKey(finding)} -->\n**${heading(finding)}**\n\n${diff(finding)}\n\n${text(finding.message)}`;
}
function renderSummary(
  report,
  { repo, head, expected = false, maintainers = [], runUrl = "" },
) {
  const findings = groupFindings(report.findings || []);
  const counts = Object.fromEntries(
    Object.keys(rank).map((kind) => [
      kind,
      findings.filter((f) => f.classification === kind).length,
    ]),
  );
  const state =
    report.execution_status !== "COMPLETE"
      ? "INCOMPLETE"
      : counts.BREAKING && !expected
        ? "BLOCKED"
        : expected && counts.BREAKING
          ? "PASS (expected)"
          : "PASS";
  const lines = [
    MARKER,
    `**API check: ${state} · ${counts.BREAKING} breaking · ${counts.LIKELY_BREAKING} likely breaking · ${counts.NON_BREAKING} non-breaking**`,
  ];
  if (report.execution_status !== "COMPLETE")
    lines.push(
      "",
      `Scan did not complete: ${text(report.error || "See the check run.")}`,
    );
  for (const kind of ["BREAKING", "LIKELY_BREAKING"]) {
    const items = findings.filter((f) => f.classification === kind);
    if (!items.length) continue;
    lines.push("", `**${heading(items[0])} · ${items.length}**`);
    for (const item of items) {
      const path = String(item.path)
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      const url = `https://github.com/${repo}/blob/${head}/${path}#L${Number(item.line) || 1}`;
      lines.push(
        "",
        `**\`${text(item.api)}\`** · [${text(item.path)}:${Number(item.line) || 1}](${url})`,
        "",
        diff(item),
      );
    }
  }
  if (state === "BLOCKED") {
    const names = maintainers.length
      ? maintainers.map((name) => `@${name}`).join(", ")
      : "with maintain/admin permission";
    lines.push(
      "",
      `Maintainers ${names} can comment \`@flashinfer-bot API BREAK expected\`, Reason: …, to unblock all API BREAK findings for the current head.`,
    );
  }
  if (runUrl) lines.push("", `[Full report and check run](${runUrl})`);
  return lines.join("\n");
}

// Only actual added/context RIGHT-side diff lines can receive inline comments.
function rightLines(patch = "") {
  let line = 0;
  const result = new Set();
  for (const row of patch.split("\n")) {
    const match = row.match(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (match) {
      line = Number(match[1]);
      continue;
    }
    if (row.startsWith("+") || row.startsWith(" ")) result.add(line++);
  }
  return result;
}
async function publishReport(github, core, pr, report, options) {
  const { owner, repo } = options;
  const current = (
    await github.rest.pulls.get({ owner, repo, pull_number: pr.number })
  ).data;
  if (current.head.sha !== pr.head.sha) return;
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const body = renderSummary(report, {
    ...options,
    repo: `${owner}/${repo}`,
    head: pr.head.sha,
  });
  if (body.length > 60000)
    throw new Error("API report exceeds comment size; inspect the artifact.");
  const previous = comments.find(
    (c) =>
      c.user?.login === "github-actions[bot]" && c.body?.startsWith(MARKER),
  );
  if (previous)
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: previous.id,
      body,
    });
  else
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: pr.number,
      body,
    });
  const [files, existing] = await Promise.all([
    github.paginate(github.rest.pulls.listFiles, {
      owner,
      repo,
      pull_number: pr.number,
      per_page: 100,
    }),
    github.paginate(github.rest.pulls.listReviewComments, {
      owner,
      repo,
      pull_number: pr.number,
      per_page: 100,
    }),
  ]);
  const eligible = new Map(
    files.map((file) => [file.filename, rightLines(file.patch)]),
  );
  for (const finding of groupFindings(report.findings || []).filter(
    (f) => f.classification !== "NON_BREAKING",
  )) {
    if (!eligible.get(finding.path)?.has(finding.line)) continue; // Details remain in the artifact.
    const key = `<!-- key:${inlineKey(finding)} -->`;
    const prior = existing.find(
      (c) =>
        c.user?.login === "github-actions[bot]" &&
        c.body?.startsWith(INLINE) &&
        c.body.includes(key) &&
        c.path === finding.path &&
        c.line === finding.line,
    );
    try {
      if (prior)
        await github.rest.pulls.updateReviewComment({
          owner,
          repo,
          comment_id: prior.id,
          body: renderInline(finding),
        });
      else
        await github.rest.pulls.createReviewComment({
          owner,
          repo,
          pull_number: pr.number,
          commit_id: pr.head.sha,
          path: finding.path,
          line: finding.line,
          side: "RIGHT",
          body: renderInline(finding),
        });
    } catch (error) {
      core.warning(`Inline API report failed: ${error.message}`);
    }
  }
}
module.exports = {
  groupFindings,
  renderSummary,
  renderInline,
  publishReport,
  rightLines,
  inlineKey,
};
