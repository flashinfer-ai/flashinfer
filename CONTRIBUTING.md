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

# Continuous Integration (CI)

FlashInfer has two CI systems: a public CI running on GitHub Actions and an NVIDIA internal CI running on GitLab.

## Public CI (GitHub Actions)

Public CI runs AOT build tests (x64/arm64) and GPU unit tests across different hardware on AWS self-hosted runners.

The commands below are the user-facing reference. The design behind them -- why there are two CI systems, how a run is narrowed to specific tests, and the constraints that follow -- is recorded in [`docs/design_docs/ci_bot_and_targeted_testing.md`](docs/design_docs/ci_bot_and_targeted_testing.md).

Public CI does not start on its own for any PR. Commenting `@flashinfer-bot run` starts it, and works for anyone who can label the PR as well as for members of the `ci-users` team. Adding the `run-ci` label by hand does the same, for anyone whose permissions let them label a PR. This applies to everyone, including maintainers.

Starting CI applies to the commit that is current at that moment. Pushing new commits, rebasing, or merging `main` into your branch does **not** start a new run, so ask for `@flashinfer-bot run` again once your PR is ready for a final check. Note that GitHub requires the checks to pass on the last commit before a PR can merge.

It is what applies the `run-ci` label that starts CI, not the label sitting on the PR, so the label stays behind after a run and adding it a second time does nothing. `@flashinfer-bot run` handles this for you by removing the label before re-adding it. If you would rather use the label directly, remove `run-ci` and add it again. Any other label leaves a running CI alone.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `@flashinfer-bot run` | Can label the PR, or `ci-users` | Start CI on the PR's current commit |
| `@flashinfer-bot rerun` | Can label the PR, or `ci-users` | Cancel and rerun all workflows |
| `@flashinfer-bot rerun failed` | Can label the PR, or `ci-users` | Rerun only failed/cancelled jobs |
| `@flashinfer-bot stop` | Can label the PR, or `ci-users` | Cancel all in-progress workflows |

> **Note:** Draft PRs work the same way. They never run CI on their own, but anyone who can use the commands above can start a run on one when you need it.

## NVIDIA Internal CI (GitLab)

Internal CI runs an extended test matrix across NVIDIA GPU architectures. It is triggered by commenting `/bot run` on a GitHub PR. The bot mirrors the PR to an internal GitLab instance, runs the pipeline, and posts results back to the PR.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `/bot run` | Allowed users | Mirror PR to GitLab and run the full unit-test pipeline |
| `/bot run tests/<dir-or-file> [tests/...]` | Allowed users | Same pipeline, scoped to one or more paths under `tests/` (whitespace-separated). Invalid tokens are rejected and do not start a pipeline. Multi-GPU and multi-node jobs still run their dedicated scripts. |
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
