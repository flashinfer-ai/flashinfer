# Installation

For development, the easiest way to install flashinfer is through editable installation:

```
git clone git@github.com:flashinfer-ai/flashinfer.git --recursive
pip install --no-build-isolation -e . -v
```

We recommend using the `--no-build-isolation` flag to ensure compatibility with your existing environment. Without it, `pip` may attempt to resolve dependencies (e.g., `torch`) from PyPI, which could pull in packages built with older CUDA versions and lead to incompatibility issues.

> **Note:** When using `--no-build-isolation`, pip does not automatically install build dependencies. FlashInfer requires `setuptools>=77`. If you encounter an error like `AttributeError: module 'setuptools.build_meta' has no attribute 'prepare_metadata_for_build_editable'`, upgrade pip and setuptools first:
> ```bash
> python -m pip install --upgrade pip setuptools
> ```

# Code Structure

```
flashinfer/
| --include/  # kernel definitions and common utilities functions
| --csrc/  # op registration to frameworks (pytorch), and binding codes
| --python/  # python interface exposed to users
| --docs/  # documentation (using sphinx)
| --tests/  # unittests in python (using pytest)
| --benchmarks/  # kernel benchmarks in python
| --3rdparty/  # 3rdparty dependencies such as cutlass
```

Kernel definitions (framework-agnostic cuda code, accepting raw pointer as input) should be placed under the `include` directory. Whenever possible, reuse existing FlashInfer infrastructure such as logging, exception handling, and utility functions.
The operator registration code (i.e., framework-specific components, accepting torch tensors as input) should reside in the `csrc` directory. This is where Torch headers may be included and operators can be bound to PyTorch. Note that Torch headers must not be included in any files under the `include` directory.

Code Contribution Procedure
* Write kernel definitions in `include/`
* Write kernel registration and pytorch interface under `csrc/`
* Write python interface under `python/`
* Write unit tests in `tests/`
* (Optional) Add benchmark suites under `benchmark/`
* Update (python) documentation index under `docs/`
* Update `pyproject.toml` if you created new module in flashinfer

# Experimental APIs and Backends

FlashInfer distinguishes stable functionality (normal review and support
expectations) from **experimental** functionality intended for fast-moving
work — client-GPU (e.g. SM12x) kernels, new operations from the latest
models, or highly specialized kernels for specific problem sizes. Two
separate concerns:

* an **experimental API** — a public interface that may change or disappear;
* an **experimental backend** — an implementation not yet ready for stable support.

A stable API may route to an experimental backend once experimental features
are enabled, and an experimental API must be clearly identified and gated.

**Placement:**

* Experimental **APIs** live in core, marked with `@flashinfer_experimental_api`
  (defined in `flashinfer/api_logging.py`). The core entry point stays thin:
  signature, shared validation, feature-gate check, backend selection,
  direct handoff.
* Experimental **backends** and all backend-specific logic (support checks,
  heuristics, routing, compilation, caching, kernels) live under
  `flashinfer/experimental/`.

**Opt-in:** using experimental functionality is always explicit. Calling an
`@flashinfer_experimental_api` function, or naming an experimental backend
(`backend="<name>"`) in a stable API, needs no environment variable; both warn
once. Only *automatic* selection is gated: `backend="auto"` (dispatch
heuristics and autotuning) may pick an experimental backend only with
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. Experimental backends are marked
with `@experimental_backend` on their `@backend_requirement` checker, which
enforces this; hand-rolled `"auto"` routing calls
`require_experimental_auto_backends(...)`.

**Requirements for every experimental feature:**

* a named owner and a tracking issue (use case, reason for the experimental
  path, graduation plan and target release — default intent is graduation
  within four weeks);
* declaration in the PR: tick the **Experimental Track** checkbox in the PR
  template and link the tracking issue there (maintainers add the
  `experimental` label);
* correctness tests against a reference in `tests/experimental/`, validated on
  the intended hardware, plus a runnable example;
* no registration in `flashinfer/aot.py` (experimental features are JIT-only
  and never ship in pre-built packages).

Experimental features provide no compatibility guarantees and may be removed
without deprecation. Changes under `flashinfer/experimental/` receive narrower
review (eligibility, correctness, containment, licensing, obvious risks);
changes to core — including thin entry points — follow the normal review
process.

For the full policy (admission criteria, lifecycle reviews, containment rules,
graduation checklist), see
[flashinfer/experimental/README.md](flashinfer/experimental/README.md).

# Pull Request Guidelines

* **Use the default PR template.** When opening a PR, fill in the repository's PR template
  (`.github/pull_request_template.md`) — do not overwrite or replace it with a custom or
  tool-generated description format. The PR title and description normally become the commit
  title and message on (squash) merge, and are relied on when bisecting changes to identify
  owners and possible bugs — keep both accurate.
* **Report performance results for optimizations.** If your PR is a performance optimization,
  report the observed performance improvement in the PR description: before/after numbers from
  a reproducible benchmark (e.g. `benchmarks/flashinfer_benchmark.py`), along with the GPU and
  problem sizes used.
* **Understand your changes.** We support AI-assisted contributions, but we expect authors to
  understand the idea and rationale of their changes. Reviewers may raise questions about the
  design — especially when the code touches a relatively durable area of the library — and if
  the author cannot walk through the rationale upon being asked, the PR submission may be
  rejected.

For how we review, see [docs/code_review_guidance_human.md](docs/code_review_guidance_human.md)
(agent reviewers follow [docs/code_review_guidance.md](docs/code_review_guidance.md)).

# Guardword Hooks (NVIDIA Contributors)

NVIDIA contributors also work with names and details that must not be published. For them,
two checks scan with NVIDIA's internal guardword scanner:

* **The push check**, a git pre-push hook, scans what a push to GitHub publishes: new commits
  and their messages, branch and tag names, and tag messages. Pushes to other remotes are not
  scanned. This check is the gate. `pip install -e .`, `pytest`, and the commit check install it
  when they detect an NVIDIA contributor. It runs on every branch, including branches that
  predate it, and then runs whatever pre-push hook it replaced, kept as
  `.git/hooks/pre-push.chained`.
* **The commit check**, the `guardwords` hook in `.pre-commit-config.yaml`, scans staged changes
  next to the formatters, while a problem is easiest to fix. It runs wherever the pre-commit
  framework's git hook is installed (`pre-commit install`). The push check installs that hook
  once per clone, along with itself; `pre-commit uninstall` turns off both the formatters and
  the commit check, and stays in effect.

"Pre-commit" names two things here: the pre-commit framework (the `pre-commit` tool, configured
by `.pre-commit-config.yaml`), and git's `pre-commit` hook, the script git runs before each
commit, which the framework installs itself as. The commit check is a framework hook. The push
check deliberately is not, although the framework supports pre-push hooks: those see only one
branch of a push, so other branches and tags in it would go unscanned, and they don't run on
branches whose `.pre-commit-config.yaml` predates the check.

What to scan, and how, is decided by a hook runner that ships with the scanner and updates together
with its rules; `scripts/guardword_hooks.py` recognizes NVIDIA contributors, installs the push
check, and applies the settings below.

Everyone else is unaffected. Detection checks the `flashinfer.internal` git setting, then for an
`@nvidia.com` commit email, a guardword scanner checkout on the machine, or
`FLASHINFER_SCAN_REPO_URL` in the environment, then, at most once a day, whether the account
logged in to the GitHub CLI (`gh`) belongs to the NVIDIA GitHub org. A positive answer from `gh` is
recorded as
`flashinfer.internal true` in your global git config. When `CI` is set, nothing is installed
automatically and commits are not scanned; an explicit `setup` still installs the push check,
and an installed one still scans.

The scanner lives in an internal repository and updates itself on every push; if it can't be
updated for a week, pushes and `status` say so. Its location is never stored in this
repository. NVIDIA contributors follow the internal setup instructions once, then run:

```bash
python3 scripts/guardword_hooks.py setup
```

It checks for pre-commit and `gh`, installs both checks, asks for the scanner's location, and
clones the scanner. Until the scanner is set up, pushes to GitHub are blocked; commits go
through unscanned. The location is kept only in the scanner checkout,
`~/.flashinfer-guardwords-scan`, which every clone on the machine shares. The devcontainers keep
that directory in a volume, so after a rebuild `setup` reinstalls the checks without asking for
it again.

Each push to GitHub updates the scanner first. If both use SSH with a key that has a passphrase
and no ssh-agent holds it, a push asks for the passphrase twice: once for GitHub, then once for
the update. To be asked only once, load the key into a running agent with `ssh-add`, or add
`AddKeysToAgent yes` to `~/.ssh/config` so the first use loads it; on macOS, `UseKeychain yes`
also keeps it across restarts.

Fresh environments, such as containers, cloud machines, and coding agents' sandboxes, often
can't recognize you automatically. Coding agents are instructed (in `AGENTS.md`) to check and,
if needed, set up the checks before their first commit, and to stop rather than push if setup
fails. For agents and other automation, give the environment the scanner's location in
`FLASHINFER_SCAN_REPO_URL`, for example as a secret: the checks then install themselves, and
`setup --non-interactive` runs without prompts.

| Task | Command |
|------|---------|
| Check whether a checkout is protected (exits non-zero if not) | `python3 scripts/guardword_hooks.py status` |
| Set up without prompts, for agents and automation | `python3 scripts/guardword_hooks.py setup --non-interactive` |
| Skip the checks once | `SKIP=guardwords git commit`, `git commit --no-verify`, or `git push --no-verify` |
| Stop scanning commits (pushes stay scanned) | `git config flashinfer.guardwords.scanCommits false` |
| Detected by mistake (installed hooks then do nothing) | `git config --global flashinfer.internal false` |
| Opt back in after opting out | `git config --global --unset flashinfer.internal` |

If `core.hooksPath` is set, the push check is not installed; the warning gives the line to add
to your own pre-push hook, and `status` checks for it.

# Continuous Integration (CI)

FlashInfer has two CI systems: a public CI running on GitHub Actions and an NVIDIA internal CI running on GitLab.

> **Targeted tests are the default when you trigger CI.** Public: `@flashinfer-bot run tests/<dir-or-file> [tests/...]`;
> internal: `/bot run tests/<dir-or-file> [tests/...]` — whitespace-separated paths under `tests/`,
> naming the test files or directories your change touches. Bare `run` asks AI for a recommendation;
> focused and inconclusive results only post advice for review. `run all` requests the
> **full** suite, which is expensive and appropriate for a cross-cutting change or final check — tick the full-scope box in the PR template and say why.

## Public CI (GitHub Actions)

Public CI runs AOT build tests (x64/arm64) and GPU unit tests across different hardware on AWS self-hosted runners.

Every public CI run that runs tests, targeted or full, also runs all of `tests/cpu` in a "CPU Unittest" lane on a
CPU runner, whatever paths you named; the GPU lanes never run it. Tests under `tests/cpu` must therefore run without a GPU.
Two runs leave it out:
a PR that changes only docs or other skipped paths runs no tests at all, and a manual `workflow_dispatch` with
`skip_gpu` drops every unit-test lane, this one included.

The commands below are the user-facing reference. The design behind them -- why there are two CI systems, how a run is narrowed to specific tests, and the constraints that follow -- is recorded in [`docs/design_docs/ci_bot_and_targeted_testing.md`](docs/design_docs/ci_bot_and_targeted_testing.md).

Public CI does not start on its own for any PR. An explicit `@flashinfer-bot run <paths>` or `@flashinfer-bot run all` starts it, and works for anyone who can label the PR as well as for members of the `ci-users` team. Adding the `run-ci` label by hand does the same, for anyone whose permissions let them label a PR. This applies to everyone, including maintainers.

Starting CI applies to the commit that is current at that moment. Pushing new commits, rebasing, or merging `main` into your branch does **not** start a new run, so ask for `@flashinfer-bot run tests/...` again (with the same targets) once your PR is ready for a final check. Note that GitHub requires the checks to pass on the last commit before a PR can merge.

It is what applies the `run-ci` label that starts CI, not the label sitting on the PR, so the label stays behind after a run and adding it a second time does nothing. `@flashinfer-bot run` handles this for you by removing the label before re-adding it. If you would rather use the label directly, remove `run-ci` and add it again. Any other label leaves a running CI alone.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `@flashinfer-bot run` | Can label the PR, or `ci-users` | AI recommends a scope; only a full-suite recommendation starts CI, while focused or uncertain decisions are posted for review |
| `@flashinfer-bot run all` | Can label the PR, or `ci-users` | Start the full suite on the current commit without AI; replaces `run full` |
| `@flashinfer-bot run tests/<dir-or-file> [tests/...]` | Can label the PR, or `ci-users` | Start CI on the explicitly named paths, without calling AI |
| `@flashinfer-bot rerun` | Can label the PR, or `ci-users` | Cancel and rerun all workflows |
| `@flashinfer-bot rerun failed` | Can label the PR, or `ci-users` | Rerun only failed/cancelled jobs |
| `@flashinfer-bot stop` | Can label the PR, or `ci-users` | Cancel all in-progress workflows |

Bare `run` asks NVIDIA Inference Hub to recommend a scope. It prefers changed
or existing tests that exercise the changed behavior, using an isolated merge of
the current target-branch tip and PR head, complete inspected test bodies, and
current CI routing. Source hooks and external Git configuration are disabled;
PR code is never imported, executed or collected during selection.

Focused suggestions (up to ten paths), no-pytest-change decisions, and uncertain
decisions produce a PR comment with the AI rationale and instructions to review
and run explicit targets. They do not change scope statuses or labels or start CI.
Only an explicit `full_suite` decision with a supporting explanation launches the
full suite, without a test-path scope. Missing evidence or an inference failure
never becomes an automatic full run. Missing configuration, conflicts, changed
head/base, or invalid AI responses start no CI. `run all` and explicit targets
bypass AI; the old `run full` spelling is rejected.

Automatic full-suite runs publish selected base/head/tree metadata and an empty-scope
fingerprint, using the existing `run-ci-focused` label for snapshot verification.
Administrators must create both `run-ci` and `run-ci-focused` labels. Explicit
paths and `run all` use `run-ci`. CI reconstructs the pinned base/head merge,
verifies the tree, and initializes submodules only afterward. GitHub's merge ref
need not be current. Missing or overwritten automatic metadata fails preparation.
Author-declared `experimental-tests` behavior remains unchanged.

Repository administrators enable automatic selection with an `INFERENCE_API_KEY`
Actions secret entitled to the selected NVIDIA Inference Hub model. Optional
Actions variables are `INFERENCE_BASE_URL` (default
`https://inference-api.nvidia.com/v1`), `CI_BOT_TARGET_MODEL` (default
`us/aws/anthropic/eccn-claude-opus-5`), and
`CI_BOT_TARGET_TIMEOUT_SECONDS` (default `240`, maximum `420`). The public
command job has a ten-minute limit; the internal daemon's timeout is independent.
Explicit paths and `run all` need no AI key.
The selector uses Git, Python's standard library and two bounded JSON model calls,
matching the Inference Hub integration used by the cuDNN frontend triage job.
Only the authorized command step receives the inference key. The endpoint must be
reachable from the GitHub-hosted command runner; it is never given to PR code.

> **Note:** Draft PRs work the same way. They never run CI on their own, but anyone who can use the commands above can start a run on one when you need it.

## NVIDIA Internal CI (GitLab)

Internal CI runs an extended test matrix across NVIDIA GPU architectures. Explicit `/bot run <paths>` and `/bot run all` requests mirror the PR to GitLab, run the pipeline, and post results back. Bare `/bot run` asks AI for a scope: focused or uncertain decisions only post advice, while a full-suite recommendation launches an unscoped run.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `/bot run` | Allowed users | AI recommends a scope; focused or uncertain results only post advice, while a full-suite recommendation launches CI on the pinned merged source |
| `/bot run all` | Allowed users | Mirror PR to GitLab and run the full unit-test pipeline without AI selection |
| `/bot run tests/<dir-or-file> [tests/...]` | Allowed users | Same pipeline, scoped to one or more paths under `tests/` (whitespace-separated). Invalid tokens are rejected and do not start a pipeline. Unit jobs collect those trees. Multi-GPU and multi-node jobs run only their hardcoded files that fall under the same paths, and skip (exit 0) before package setup when none overlap. |
| `/bot status` | Allowed users | Check current pipeline status |
| `/bot stop` | Allowed users | Cancel a running pipeline |

> **Note:** Access to the NVIDIA internal CI is limited to NVIDIA employees and approved collaborators. To request access, please reach out to @yongwww, @dierksen, @yzh119, or @sricketts.

**Internal CI test matrix:**

| Test | GPU | CUDA | Notes |
|------|-----|------|-------|
| `unit_test_h100` | H100 | cu129, cu130 | |
| `unit_test_b200` | B200 | cu129, cu130 | |
| `unit_test_b300` | B300 | cu129, cu130 | |
| `unit_test_gb200` | GB200 | cu129, cu130 | |
| `unit_test_gb300` | GB300 | cu129, cu130 | |
| `unit_test_5090` | RTX 5090 | cu129, cu130 | |
| `unit_test_rtx_pro_6000` | RTX PRO 6000 Blackwell | cu129, cu130 | |
| `unit_test_spark` | Spark | cu129, cu130 | manual-trigger only |
| `unit_test_thor` | Thor | cu130 | manual-trigger only |
| `multi_gpu_test_b300` | B300 (multi-GPU) | cu129, cu130 | |
| `multi_node_test_b300` | B300 (multi-node) | cu129, cu130 | |
| `multi_node_test_gb200` | GB200 (multi-node) | cu129, cu130 | |
| `multi_node_test_gb300` | GB300 (multi-node) | cu129, cu130 | |

# Claiming Issues

Want to work on an issue? Use these commands in the issue comments:

| Command | Who can use | Description |
|---------|-------------|-------------|
| `!claim` | Anyone | Self-assign an unassigned issue |
| `!assign @username` | Admins/Maintainers | Assign a specific user to an issue |

**`!claim`** — Comment `!claim` on any open, unassigned issue to assign yourself. If the issue is already assigned, you'll be asked to contact a maintainer.

**`!assign @username`** — Maintainers can comment `!assign @username` to assign someone. If the user is not yet a collaborator, a triage invitation is sent automatically and they will be assigned once they accept.

# Release Versioning

When incrementing a version and creating a release, follow a "right-shifted" versioning scheme similar to [vLLM Release Versioning](https://github.com/vllm-project/vllm/blob/main/RELEASE.md) (`major.minor.patch[.post1]`) [^1]. In particular:

* _major_ increment signals architectural milestone and/or when incompatible API changes are made, similar to PyTorch 2.0.
* _minor_ increment signals significant backwards-compatible new features
* _patch_ increment signals small backwards-compatible features (e.g. new kernels, new SM support, etc) and backwards-compatible bug fixes
* _post1_ is an optional suffix for a quick follow up release with just backwards-compatible bug fixes

Like the vLLM scheme, this versioning scheme is similar to [SemVer](https://semver.org/) for compatibility purposes, except that backwards compatibility is only guaranteed for a limited number of minor releases (see the [vLLM deprecation policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy) for details).

To reduce disruption during deprecation and removal, we prefer "keyword only" (after an `*`, see [PEP-3102](https://peps.python.org/pep-3102/)) for parameters that are likely to come and go (e.g. perf parameters).

[^1]: We have not followed this strictly through v0.4.0. But after v0.4.0, the versioning should follow this "right-shifted" versioning scheme.
