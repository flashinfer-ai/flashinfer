// Compact API-only reports. PR source text is untrusted even in a trusted scan.
const crypto = require("node:crypto");
const { selectAnchor, diffLines } = require("./api_anchors.cjs");
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
        anchors: [
          ...new Map(
            [...(previous.anchors || []), ...(finding.anchors || [])].map(
              (a) => [JSON.stringify(a), a],
            ),
          ).values(),
        ],
      });
    }
  }
  return [...groups.values()];
}
const codeText = (value) =>
  String(value ?? "")
    .replace(/@/g, "@\u200b")
    .replace(/`/g, "ˋ")
    .replace(/[\r\n]/g, " ")
    .slice(0, 2500);
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
    `${start ? "…" : ""}${codeText(value.slice(start, value.length - trimEnd))}${trimEnd ? "…" : ""}`;
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
  if (
    report.execution_status === "COMPLETE" &&
    !counts.BREAKING &&
    !counts.LIKELY_BREAKING
  )
    return null;
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
    `**API check: ${state} · ${counts.BREAKING} breaking · ${counts.LIKELY_BREAKING} likely breaking**`,
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
      const source =
        (item.anchors || []).find((a) => a[1] === "RIGHT") || item.anchors?.[0];
      const sourcePath = source?.[0] || item.path;
      const sourceLine = source?.[2] || item.line;
      const revision = source?.[1] === "LEFT" ? report.base || head : head;
      const path = String(sourcePath)
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      const url = `https://github.com/${repo}/blob/${revision}/${path}#L${Number(sourceLine) || 1}`;
      lines.push(
        "",
        `**\`${text(item.api)}\`** · [${text(sourcePath)}:${Number(sourceLine) || 1}](${url})`,
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

function rightLines(patch = "") {
  return new Set(
    diffLines(patch)
      .filter((x) => x.side === "RIGHT")
      .map((x) => x.line),
  );
}
async function publishReport(github, core, pr, report, options) {
  const { owner, repo } = options;
  const delivery = {
    total: 0,
    line: 0,
    file: 0,
    posted: 0,
    updated: 0,
    unplaced: 0,
    failed: 0,
  };
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
  const previous = comments.find(
    (c) =>
      c.user?.login === "github-actions[bot]" && c.body?.startsWith(MARKER),
  );
  if (body === null) {
    if (previous)
      await github.rest.issues.deleteComment({
        owner,
        repo,
        comment_id: previous.id,
      });
    return delivery;
  }
  if (body.length > 60000)
    throw new Error("API report exceeds comment size; inspect the artifact.");
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
  const pendingLines = [];
  for (const finding of groupFindings(report.findings || []).filter(
    (f) => f.classification !== "NON_BREAKING",
  )) {
    delivery.total++;
    const anchor = selectAnchor(finding, files);
    if (!anchor) {
      delivery.unplaced++;
      core.warning(`No changed file for API review comment: ${finding.api}`);
      continue;
    }
    const key = `<!-- key:${inlineKey(finding)} -->`;
    const fileLevel = anchor.subject_type === "file";
    const prior = existing.find(
      (c) =>
        c.user?.login === "github-actions[bot]" &&
        c.body?.startsWith(INLINE) &&
        c.body.includes(key) &&
        c.path === anchor.path &&
        (fileLevel
          ? c.subject_type === "file"
          : c.line === anchor.line && (c.side || "RIGHT") === anchor.side),
    );
    const body = renderInline(finding);
    try {
      if (prior) {
        await github.rest.pulls.updateReviewComment({
          owner,
          repo,
          comment_id: prior.id,
          body,
        });
        delivery.updated++;
      } else if (
        !fileLevel &&
        typeof github.rest.pulls.createReview === "function"
      ) {
        pendingLines.push({ ...anchor, body });
        continue;
      } else {
        await github.rest.pulls.createReviewComment({
          owner,
          repo,
          pull_number: pr.number,
          commit_id: pr.head.sha,
          ...anchor,
          body,
        });
        delivery.posted++;
      }
      delivery[fileLevel ? "file" : "line"]++;
    } catch (error) {
      delivery.failed++;
      core.warning(
        `API review comment failed for ${finding.api}: ${error.message}`,
      );
    }
  }
  if (pendingLines.length) {
    try {
      await github.rest.pulls.createReview({
        owner,
        repo,
        pull_number: pr.number,
        commit_id: pr.head.sha,
        event: "COMMENT",
        body: "<!-- flashinfer-api-inline-review -->",
        comments: pendingLines,
      });
      delivery.posted += pendingLines.length;
      delivery.line += pendingLines.length;
    } catch (error) {
      // A validation failure on one anchor must not hide unrelated findings.
      // Do not amplify permission/rate-limit failures with per-line retries.
      if (error.status === 422) {
        for (const comment of pendingLines) {
          try {
            await github.rest.pulls.createReviewComment({
              owner,
              repo,
              pull_number: pr.number,
              commit_id: pr.head.sha,
              ...comment,
            });
            delivery.posted++;
            delivery.line++;
          } catch (individual) {
            delivery.failed++;
            core.warning(`API inline comment failed: ${individual.message}`);
          }
        }
      } else {
        delivery.failed += pendingLines.length;
        core.warning(`API inline review failed: ${error.message}`);
      }
    }
  }
  return delivery;
}
module.exports = {
  groupFindings,
  renderSummary,
  renderInline,
  publishReport,
  rightLines,
  inlineKey,
};
