// Trusted API gate and comment command handler. Never load code from a PR head.
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { publishReport } = require("./api_report.cjs");
const GATE = "API compatibility";
const parseCommand = (body) =>
  String(body)
    .trim()
    .match(/^@flashinfer-bot[ \t]+API[ \t]+BREAK[ \t]+(expected|revoke)$/i)?.[1]
    .toLowerCase() || null;
const canApprove = (role, actor, author, type) =>
  ["admin", "maintain"].includes(role) &&
  type === "User" &&
  actor.toLowerCase() !== author.toLowerCase();
const runUrl = (context) =>
  `${context.serverUrl || "https://github.com"}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
const scanContext = (pr) => `api-scan/pr-${pr.number}`;
const exceptionContext = (pr) => `api-break-exception/pr-${pr.number}`;
function gateState(report, expected) {
  if (
    !report ||
    !Array.isArray(report.findings) ||
    !["COMPLETE", "FAILED"].includes(report.execution_status)
  )
    throw Error("Invalid API report");
  if (report.execution_status !== "COMPLETE") return "error";
  for (const f of report.findings) {
    if (
      !["BREAKING", "LIKELY_BREAKING", "NON_BREAKING"].includes(
        f.classification,
      ) ||
      f.level !==
        {
          BREAKING: "error",
          LIKELY_BREAKING: "warning",
          NON_BREAKING: "notice",
        }[f.classification]
    )
      throw Error("Invalid API classification");
  }
  return !expected &&
    report.findings.some((f) => f.classification === "BREAKING")
    ? "failure"
    : "success";
}
async function statuses(github, context, pr) {
  return github.paginate(github.rest.repos.listCommitStatusesForRef, {
    ...context.repo,
    ref: pr.head.sha,
    per_page: 100,
  });
}
async function status(
  github,
  context,
  pr,
  name,
  state,
  description,
  target = runUrl(context),
) {
  await github.rest.repos.createCommitStatus({
    ...context.repo,
    sha: pr.head.sha,
    context: name,
    state,
    description,
    target_url: target,
  });
}
async function currentHead(github, context, pr) {
  const latest = (
    await github.rest.pulls.get({ ...context.repo, pull_number: pr.number })
  ).data;
  return (
    latest.state === "open" &&
    latest.head.sha === pr.head.sha &&
    latest.base.sha === pr.base.sha
  );
}
async function provenance(github, context, item, event, workflow) {
  if (item.creator?.login !== "github-actions[bot]") return false;
  const prefix = `${context.serverUrl || "https://github.com"}/${context.repo.owner}/${context.repo.repo}/actions/runs/`;
  if (!item.target_url?.startsWith(prefix)) return false;
  const id = item.target_url.slice(prefix.length);
  if (!/^[1-9][0-9]*$/.test(id)) return false;
  const { data: run } = await github.rest.actions.getWorkflowRun({
    ...context.repo,
    run_id: Number(id),
  });
  return (
    run.event === event &&
    run.path?.split("@")[0] === `.github/workflows/${workflow}` &&
    run.repository?.full_name === `${context.repo.owner}/${context.repo.repo}`
  );
}
async function commandPermission(github, context, pr, comment) {
  const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...context.repo,
    username: comment.user.login,
  });
  return canApprove(
    data.role_name,
    comment.user.login,
    pr.user.login,
    comment.user.type,
  );
}
async function exceptionComment(github, context, pr, item) {
  if (
    item.context !== exceptionContext(pr) ||
    !/^expected:\d+$|^revoke:\d+$/.test(item.description || "")
  )
    return null;
  if (
    !(await provenance(
      github,
      context,
      item,
      "issue_comment",
      "ci-bot-commands.yml",
    ))
  )
    return null;
  const id = Number(item.description.split(":")[1]);
  const { data: comment } = await github.rest.issues.getComment({
    ...context.repo,
    comment_id: id,
  });
  const expectedUrl = `${context.serverUrl || "https://github.com"}/${context.repo.owner}/${context.repo.repo}/pull/${pr.number}#issuecomment-${id}`;
  if (
    comment.html_url !== expectedUrl ||
    comment.updated_at !== comment.created_at ||
    parseCommand(comment.body) !== item.description.split(":")[0] ||
    !(await commandPermission(github, context, pr, comment))
  )
    return null;
  return comment;
}
async function validException(github, context, pr, item) {
  return (
    item.state === "success" &&
    item.description?.startsWith("expected:") &&
    Boolean(await exceptionComment(github, context, pr, item))
  );
}
async function approved(github, context, pr, records) {
  // Command ID orders intent, not workflow completion time. A delayed older
  // expected command must never override a newer revoke.
  const items = records
    .filter((s) => s.context === exceptionContext(pr))
    .sort(
      (a, b) =>
        Number(b.description?.split(":")[1] || 0) -
        Number(a.description?.split(":")[1] || 0),
    );
  for (const item of items) {
    if (
      await provenance(
        github,
        context,
        item,
        "issue_comment",
        "ci-bot-commands.yml",
      )
    ) {
      // An edited/deleted latest approval invalidates it; never resurrect an
      // older approval after a revocation or invalid latest command.
      return validException(github, context, pr, item);
    }
  }
  return false;
}
function scan(pr) {
  if (
    !/^[a-f0-9]{40}$/.test(pr.head.sha) ||
    !/^[a-f0-9]{40}$/.test(pr.base.sha)
  )
    throw Error("Invalid PR revisions");
  const command = (file, args) =>
    execFileSync(file, args, {
      encoding: "utf8",
      timeout: 180000,
      maxBuffer: 20 * 1024 * 1024,
    });
  // Fetch objects only. No checkout/submodules/install/import from the PR.
  command("git", ["fetch", "--no-tags", "origin", pr.base.sha, pr.head.sha]);
  const directory = process.env.RUNNER_TEMP || require("node:os").tmpdir();
  const reportPath = path.join(
    fs.mkdtempSync(path.join(directory, "api-scan-")),
    "public-api.json",
  );
  const scanner = path.join(__dirname, "..", "check_pr_api_diff.py");
  try {
    command("python3", [
      scanner,
      "--base",
      pr.base.sha,
      "--head",
      pr.head.sha,
      "--fail-on",
      "error",
      "--report-json",
      reportPath,
    ]);
  } catch (error) {
    if (![1, 2].includes(error.status) || !fs.existsSync(reportPath))
      throw error;
  }
  const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
  if (report.head !== pr.head.sha) throw Error("API report head mismatch");
  fs.mkdirSync("reports", { recursive: true });
  fs.writeFileSync(
    "reports/public-api.json",
    JSON.stringify(report, null, 2) + "\n",
  );
  gateState(report, false); // Validate before publication/approval.
  return report;
}
async function evaluate({ github, core, context, pr, report, publish = true }) {
  const expected = await approved(
    github,
    context,
    pr,
    await statuses(github, context, pr),
  );
  const state = gateState(report, expected);
  if (!(await currentHead(github, context, pr))) return false;
  await status(
    github,
    context,
    pr,
    GATE,
    state,
    state === "error"
      ? "API scan incomplete"
      : expected
        ? "API breaks accepted for this head"
        : state === "failure"
          ? "Breaking API changes require maintainer confirmation"
          : "No blocking API changes",
  );
  if (publish) {
    let maintainers = [];
    try {
      const collaborators = await github.paginate(
        github.rest.repos.listCollaborators,
        { ...context.repo, per_page: 100 },
      );
      maintainers = collaborators
        .filter((c) => ["admin", "maintain"].includes(c.role_name))
        .map((c) => c.login);
    } catch (error) {
      core.warning(`Maintainer accounts could not be listed: ${error.message}`);
    }
    try {
      const delivery = await publishReport(github, core, pr, report, {
        ...context.repo,
        expected,
        maintainers,
        runUrl: runUrl(context),
      });
      report.comment_delivery = delivery;
      fs.mkdirSync("reports", { recursive: true });
      fs.writeFileSync(
        "reports/public-api.json",
        JSON.stringify(report, null, 2) + "\n",
      );
    } catch (error) {
      core.warning(`API comments could not be updated: ${error.message}`);
    }
  }
  return state === "success" && (await currentHead(github, context, pr));
}
async function runScan({ github, core, context }) {
  const pr = (
    await github.rest.pulls.get({
      ...context.repo,
      pull_number: context.payload.pull_request.number,
    })
  ).data;
  await status(
    github,
    context,
    pr,
    GATE,
    "pending",
    "Scanning public API compatibility",
  );
  try {
    const report = scan(pr);
    if (!(await currentHead(github, context, pr))) return;
    await status(
      github,
      context,
      pr,
      scanContext(pr),
      report.execution_status === "COMPLETE" ? "success" : "error",
      report.execution_status === "COMPLETE"
        ? "Trusted API scan complete"
        : "Trusted API scan incomplete",
    );
    await evaluate({ github, core, context, pr, report });
  } catch (error) {
    core.warning(error.message);
    await status(
      github,
      context,
      pr,
      scanContext(pr),
      "error",
      "Trusted API scan failed",
    );
    await status(
      github,
      context,
      pr,
      GATE,
      "error",
      "API scan failed; inspect workflow logs",
    );
    throw error;
  }
}
async function handleCommand({ github, core, context, scan: scanFn = scan }) {
  const comment = context.payload.comment;
  const command = parseCommand(comment.body);
  if (!command) return;
  const react = (content) =>
    github.rest.reactions.createForIssueComment({
      ...context.repo,
      comment_id: comment.id,
      content,
    });
  await react("eyes");
  const pr = (
    await github.rest.pulls.get({
      ...context.repo,
      pull_number: context.payload.issue.number,
    })
  ).data;
  if (
    !(await commandPermission(github, context, pr, comment)) ||
    pr.state !== "open" ||
    comment.updated_at !== comment.created_at
  )
    return;
  const records = await statuses(github, context, pr);
  const latestScan = records
    .filter((s) => s.context === scanContext(pr))
    .sort((a, b) => b.id - a.id)[0];
  if (
    command === "expected" &&
    (!latestScan ||
      latestScan.state !== "success" ||
      Date.parse(latestScan.created_at) > Date.parse(comment.created_at) ||
      !(await provenance(
        github,
        context,
        latestScan,
        "pull_request_target",
        "pr-api-compatibility.yml",
      )))
  )
    return;
  // Read live source comment, so edits/deletions during queueing cannot approve.
  const { data: live } = await github.rest.issues.getComment({
    ...context.repo,
    comment_id: comment.id,
  });
  if (
    live.body !== comment.body ||
    live.updated_at !== comment.updated_at ||
    !(await currentHead(github, context, pr))
  )
    return;
  const newer = records.filter(
    (s) =>
      s.context === exceptionContext(pr) &&
      Number(s.description?.split(":")[1]) > comment.id,
  );
  for (const record of newer)
    if (
      await provenance(
        github,
        context,
        record,
        "issue_comment",
        "ci-bot-commands.yml",
      )
    )
      return;
  if (command === "revoke") {
    await status(
      github,
      context,
      pr,
      exceptionContext(pr),
      "failure",
      `revoke:${comment.id}`,
    );
  }
  await status(
    github,
    context,
    pr,
    GATE,
    "pending",
    "Re-evaluating API compatibility",
  );
  let report;
  try {
    report = await scanFn(pr);
    if (gateState(report, true) === "error")
      throw Error("API rescan did not complete");
  } catch (error) {
    core.warning(error.message);
    await status(
      github,
      context,
      pr,
      scanContext(pr),
      "error",
      "Trusted API rescan failed",
    );
    await status(
      github,
      context,
      pr,
      GATE,
      "error",
      "API rescan failed; inspect workflow logs",
    );
    return;
  }
  if (!(await currentHead(github, context, pr))) return;
  if (command === "expected") {
    await status(
      github,
      context,
      pr,
      exceptionContext(pr),
      "success",
      `expected:${comment.id}`,
    );
  }
  // The approval status links this trusted workflow run; its description links
  // the immutable command ID. Both are checked when reading it back.
  const passed = await evaluate({ github, core, context, pr, report });
  if (command === "expected" && passed) await react("+1");
}
module.exports = {
  parseCommand,
  canApprove,
  gateState,
  validException,
  handleCommand,
  runScan,
  approved,
  scan,
};
