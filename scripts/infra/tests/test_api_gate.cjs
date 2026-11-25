const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  parseCommand,
  canApprove,
  gateState,
  handleCommand,
  validException,
} = require("../pr_checks/api_gate.cjs");

test("only exact standalone commands are recognized", () => {
  assert.equal(parseCommand("@flashinfer-bot API BREAK expected"), "expected");
  assert.equal(parseCommand(" @flashinfer-bot api break revoke\n"), "revoke");
  for (const body of [
    "text @flashinfer-bot API BREAK expected",
    "> @flashinfer-bot API BREAK expected",
    "```\n@flashinfer-bot API BREAK expected\n```",
    "@flashinfer-bot API BREAK expected all",
    "@flashinfer_bot API BREAK expected",
    "@flashinfer-bot API BREAK expected\nmore",
  ]) {
    assert.equal(parseCommand(body), null, body);
  }
});
test("CI run permission does not authorize exceptions or author self-approval", () => {
  assert.equal(canApprove("maintain", "alice", "bob", "User"), true);
  assert.equal(canApprove("admin", "alice", "alice", "User"), false);
  for (const role of ["write", "triage", "none"])
    assert.equal(canApprove(role, "alice", "bob", "User"), false);
  assert.equal(canApprove("admin", "bot", "bob", "Bot"), false);
});
test("exception only waives complete classified API errors", () => {
  const complete = {
    execution_status: "COMPLETE",
    findings: [{ classification: "BREAKING", level: "error" }],
  };
  assert.equal(gateState(complete, false), "failure");
  assert.equal(gateState(complete, true), "success");
  assert.equal(
    gateState({ ...complete, execution_status: "FAILED" }, true),
    "error",
  );
  assert.equal(
    gateState(
      {
        execution_status: "COMPLETE",
        findings: [{ classification: "LIKELY_BREAKING", level: "warning" }],
      },
      false,
    ),
    "success",
  );
  assert.throws(() => gateState({ findings: [] }, true));
});

function fixture({
  role = "maintain",
  author = "bob",
  stale = false,
  scanState = "success",
  scanAt = "2026-11-25T00:00:00Z",
} = {}) {
  const calls = [];
  const sha = "a".repeat(40);
  const pr = {
    number: 3,
    state: "open",
    user: { login: author },
    head: { sha },
    base: { sha: "b".repeat(40) },
  };
  const comment = {
    id: 10,
    body: "@flashinfer-bot API BREAK expected",
    user: { login: "alice", type: "User" },
    html_url: "https://github.com/org/repo/pull/3#issuecomment-10",
    created_at: "2026-11-25T01:00:00Z",
    updated_at: "2026-11-25T01:00:00Z",
  };
  const statuses = [
    {
      id: 1,
      context: "api-scan/pr-3",
      state: scanState,
      created_at: scanAt,
      target_url: "https://github.com/org/repo/actions/runs/1",
      creator: { login: "github-actions[bot]" },
    },
  ];
  const github = {
    paginate: async (fn) => fn(),
    rest: {
      pulls: {
        list: async () => [pr],
        get: async () => ({
          data: { ...pr, head: { sha: stale ? "c".repeat(40) : pr.head.sha } },
        }),
      },
      repos: {
        getCollaboratorPermissionLevel: async () => ({
          data: { role_name: role },
        }),
        listCommitStatusesForRef: async () => statuses,
        createCommitStatus: async (data) => {
          calls.push(["status", data]);
          statuses.unshift({
            id: 2,
            ...data,
            creator: { login: "github-actions[bot]" },
            created_at: "2026-11-25T01:00:01Z",
          });
        },
      },
      issues: { getComment: async () => ({ data: comment }) },
      reactions: {
        createForIssueComment: async (data) => {
          calls.push(["reaction", data.content]);
        },
      },
      actions: {
        getWorkflowRun: async ({ run_id }) => ({
          data: {
            event: run_id === 1 ? "pull_request_target" : "issue_comment",
            path:
              run_id === 1
                ? ".github/workflows/pr-api-compatibility.yml"
                : ".github/workflows/ci-bot-commands.yml",
            repository: { full_name: "org/repo" },
          },
        }),
      },
    },
  };
  const context = {
    repo: { owner: "org", repo: "repo" },
    runId: 2,
    serverUrl: "https://github.com",
    payload: { issue: { number: 3 }, comment },
  };
  return { github, context, calls, pr, statuses, comment };
}
test("unauthorized command gets eyes but cannot approve or thumbs-up", async () => {
  const f = fixture({ role: "write" });
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("must not scan");
    },
  });
  assert.deepEqual(f.calls, [["reaction", "eyes"]]);
});
test("queued command cannot approve a scan that completed after its creation", async () => {
  const f = fixture({ scanAt: "2026-11-25T02:00:00Z" });
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("must not scan");
    },
  });
  assert.deepEqual(f.calls, [["reaction", "eyes"]]);
});
test("failed scan cannot be approved", async () => {
  const f = fixture({ scanState: "error" });
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("must not scan");
    },
  });
  assert.deepEqual(f.calls, [["reaction", "eyes"]]);
});
test("authorized command records exact head and only then passes gate and thumbs-up", async () => {
  const f = fixture();
  const report = {
    execution_status: "COMPLETE",
    findings: [{ classification: "BREAKING", level: "error" }],
  };
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => report,
  });
  assert.equal(f.calls[0][1], "eyes");
  assert.equal(
    f.calls.find(
      (x) => x[0] === "status" && x[1].context === "api-break-exception/pr-3",
    )[1].sha,
    f.pr.head.sha,
  );
  assert.equal(f.calls.at(-1)[1], "+1");
});
test("forged exception provenance is rejected", async () => {
  const f = fixture();
  const forged = {
    ...f.statuses[0],
    context: "api-break-exception/pr-3",
    description: "expected:10",
    state: "success",
  };
  assert.equal(await validException(f.github, f.context, f.pr, forged), false); // scan workflow cannot approve
  forged.creator = { login: "alice" };
  assert.equal(await validException(f.github, f.context, f.pr, forged), false);
});

test("author cannot approve their own PR", async () => {
  const f = fixture({ author: "alice" });
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("must not scan");
    },
  });
  assert.deepEqual(f.calls, [["reaction", "eyes"]]);
});
test("head change during scan cannot persist approval or thumbs-up", async () => {
  const f = fixture();
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      f.pr.head.sha = "c".repeat(40);
      return { execution_status: "COMPLETE", findings: [] };
    },
  });
  assert.equal(
    f.calls.some(
      (c) => c[0] === "status" && c[1].context === "api-break-exception/pr-3",
    ),
    false,
  );
  assert.equal(
    f.calls.some((c) => c[0] === "reaction" && c[1] === "+1"),
    false,
  );
});
test("incomplete rescan cannot be waived", async () => {
  const f = fixture();
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => ({ execution_status: "FAILED", findings: [] }),
  });
  assert.equal(
    f.calls.some(
      (c) => c[0] === "status" && c[1].context === "api-break-exception/pr-3",
    ),
    false,
  );
  assert.equal(
    f.calls.some((c) => c[0] === "reaction" && c[1] === "+1"),
    false,
  );
});
test("newer revoke wins even if older expected workflow finishes later", async () => {
  const f = fixture();
  f.statuses.push({
    id: 7,
    context: "api-break-exception/pr-3",
    description: "revoke:11",
    state: "failure",
    target_url: "https://github.com/org/repo/actions/runs/2",
    creator: { login: "github-actions[bot]" },
  });
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("must not scan");
    },
  });
  assert.deepEqual(f.calls, [["reaction", "eyes"]]);
});
test("edited latest approval does not resurrect an older expected", async () => {
  const f = fixture();
  const older = { ...f.comment };
  const newer = {
    ...f.comment,
    id: 11,
    html_url: f.comment.html_url.replace("-10", "-11"),
    updated_at: "2026-11-25T02:00:00Z",
  };
  f.github.rest.issues.getComment = async ({ comment_id }) => ({
    data: comment_id === 10 ? older : newer,
  });
  const records = [10, 11].map((id) => ({
    context: "api-break-exception/pr-3",
    description: `expected:${id}`,
    state: "success",
    target_url: "https://github.com/org/repo/actions/runs/2",
    creator: { login: "github-actions[bot]" },
  }));
  const { approved } = require("../pr_checks/api_gate.cjs");
  assert.equal(await approved(f.github, f.context, f.pr, [records[0]]), true);
  assert.equal(await approved(f.github, f.context, f.pr, records), false);
});
test("revoke reblocks the gate without posting an unblock thumbs-up", async () => {
  const f = fixture();
  f.comment.body = "@flashinfer-bot API BREAK revoke";
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => ({
      execution_status: "COMPLETE",
      findings: [{ classification: "BREAKING", level: "error" }],
    }),
  });
  const gate = f.calls
    .filter((c) => c[0] === "status" && c[1].context === "API compatibility")
    .at(-1);
  assert.equal(gate[1].state, "failure");
  assert.equal(
    f.calls.some((c) => c[0] === "reaction" && c[1] === "+1"),
    false,
  );
});

test("trusted scanner ignores a PR replacement checker and never imports package code", () => {
  const fs = require("node:fs");
  const os = require("node:os");
  const path = require("node:path");
  const { execFileSync } = require("node:child_process");
  const { scan } = require("../pr_checks/api_gate.cjs");
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "api-trust-test-"));
  const cwd = process.cwd();
  const git = (...args) =>
    execFileSync("git", args, {
      cwd: directory,
      env: Object.fromEntries(
        Object.entries(process.env).filter(([key]) => !key.startsWith("GIT_")),
      ),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  try {
    git("init", "-q");
    git("config", "user.name", "API test");
    git("config", "user.email", "api@example.com");
    git("remote", "add", "origin", directory);
    fs.mkdirSync(path.join(directory, "flashinfer"));
    fs.writeFileSync(
      path.join(directory, "flashinfer/example.py"),
      "@flashinfer_api\ndef api(x): pass\n",
    );
    fs.writeFileSync(
      path.join(directory, "flashinfer/__init__.py"),
      'raise RuntimeError("PR package must never execute")\n',
    );
    git("add", ".");
    git("commit", "-qm", "base");
    const base = git("rev-parse", "HEAD");
    fs.writeFileSync(
      path.join(directory, "flashinfer/example.py"),
      "@flashinfer_api\ndef api(x, workspace): pass\n",
    );
    fs.mkdirSync(path.join(directory, "scripts/infra"), { recursive: true });
    fs.writeFileSync(
      path.join(directory, "scripts/infra/check_pr_api_diff.py"),
      'raise RuntimeError("untrusted checker executed")\n',
    );
    git("add", ".");
    git("commit", "-qm", "head");
    const head = git("rev-parse", "HEAD");
    process.chdir(directory);
    const report = scan({ base: { sha: base }, head: { sha: head } });
    assert.equal(report.execution_status, "COMPLETE");
    assert.equal(
      report.findings.find((f) => f.classification === "BREAKING").api,
      "flashinfer.example.api",
    );
  } finally {
    process.chdir(cwd);
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("revoke persists before a failed rescan and leaves no passing gate", async () => {
  for (const fail of ["throw", "incomplete"]) {
    const f = fixture({ scanState: "error" });
    f.comment.body = "@flashinfer-bot API BREAK revoke";
    await handleCommand({
      ...f,
      core: { warning() {} },
      scan: async () => {
        if (fail === "throw") throw Error("scan failed");
        return { execution_status: "FAILED", findings: [] };
      },
    });
    assert.ok(
      f.calls.some(
        (c) =>
          c[0] === "status" &&
          c[1].context === "api-break-exception/pr-3" &&
          c[1].description === "revoke:10",
      ),
    );
    const gate = f.calls
      .filter((c) => c[0] === "status" && c[1].context === "API compatibility")
      .at(-1);
    assert.equal(gate[1].state, "error");
    assert.equal(
      f.calls.some((c) => c[0] === "reaction" && c[1] === "+1"),
      false,
    );
  }
});

test("deleted latest approval invalidates it without hiding other API errors", async () => {
  const f = fixture();
  const item = {
    context: "api-break-exception/pr-3",
    description: "expected:11",
    state: "success",
    target_url: "https://github.com/org/repo/actions/runs/2",
    creator: { login: "github-actions[bot]" },
  };
  f.github.rest.issues.getComment = async () => {
    throw Object.assign(Error("deleted"), { status: 404 });
  };
  assert.equal(await validException(f.github, f.context, f.pr, item), false);
  f.github.rest.issues.getComment = async () => {
    throw Object.assign(Error("rate limited"), { status: 403 });
  };
  await assert.rejects(
    validException(f.github, f.context, f.pr, item),
    /rate limited/,
  );
});

test("successful command rescan restores evidence for a later expected command", async () => {
  const f = fixture({ scanState: "error" });
  f.comment.body = "@flashinfer-bot API BREAK revoke";
  const scan = async () => ({
    execution_status: "COMPLETE",
    findings: [{ classification: "BREAKING", level: "error" }],
  });
  await handleCommand({ ...f, core: { warning() {} }, scan });
  const restored = f.calls
    .filter((x) => x[0] === "status" && x[1].context === "api-scan/pr-3")
    .at(-1);
  assert.equal(restored?.[1].state, "success");
  f.comment.id = 11;
  f.comment.html_url = f.comment.html_url.replace("-10", "-11");
  f.comment.body = "@flashinfer-bot API BREAK expected";
  f.comment.created_at = f.comment.updated_at = "2026-11-25T03:00:00Z";
  await handleCommand({ ...f, core: { warning() {} }, scan });
  assert.equal(f.calls.at(-1)[1], "+1");
});

test("failed revoke refreshes an old expected summary as incomplete", async () => {
  const f = fixture();
  f.comment.body = "@flashinfer-bot API BREAK revoke";
  f.github.rest.issues.listComments = async () => [
    {
      id: 5,
      user: { login: "github-actions[bot]" },
      body: "<!-- flashinfer-api-compatibility -->PASS (expected)",
    },
  ];
  let body;
  f.github.rest.issues.updateComment = async (x) => {
    body = x.body;
  };
  f.github.rest.pulls.listFiles = async () => [];
  f.github.rest.pulls.listReviewComments = async () => [];
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => {
      throw Error("syntax failure");
    },
  });
  assert.match(body, /INCOMPLETE/);
  assert.doesNotMatch(body, /PASS/);
});

test("one PR exception cannot pass a shared head used by another open PR", async () => {
  const f = fixture();
  f.github.rest.pulls.list = async () => [f.pr, { ...f.pr, number: 4 }];
  await handleCommand({
    ...f,
    core: { warning() {} },
    scan: async () => ({
      execution_status: "COMPLETE",
      findings: [{ classification: "BREAKING", level: "error" }],
    }),
  });
  const gate = f.calls
    .filter((x) => x[0] === "status" && x[1].context === "API compatibility")
    .at(-1)[1];
  assert.equal(gate.state, "failure");
  assert.match(gate.description, /shared|share/i);
  assert.equal(
    f.calls.some((x) => x[0] === "reaction" && x[1] === "+1"),
    false,
  );
});
