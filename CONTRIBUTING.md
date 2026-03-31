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

# Continuous Integration (CI)

FlashInfer has two CI systems: a public CI running on GitHub Actions and an NVIDIA internal CI running on GitLab.

## Public CI (GitHub Actions)

Public CI runs AOT build tests (x64/arm64) and GPU unit tests across different hardware on AWS self-hosted runners.

**For org members (`ci-users` team):** CI triggers automatically when you open or update a PR.

**For other contributors:** If you are not in the `ci-users` team, CI will not run automatically. A `ci-users` team member can approve it by commenting `@flashinfer-bot run` or by adding the `run-ci` label to the PR.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `@flashinfer-bot run` | `ci-users` team | Approve and trigger CI for a PR |
| `@flashinfer-bot rerun` | `ci-users` team | Cancel and rerun all workflows |
| `@flashinfer-bot rerun failed` | `ci-users` team | Rerun only failed/cancelled jobs |
| `@flashinfer-bot stop` | `ci-users` team | Cancel all in-progress workflows |

> **Note:** Draft PRs skip CI automatically. Mark your PR as ready for review to enable CI.

## NVIDIA Internal CI (GitLab)

Internal CI runs an extended test matrix across NVIDIA GPU architectures. It is triggered by commenting `/bot run` on a GitHub PR. The bot mirrors the PR to an internal GitLab instance, runs the pipeline, and posts results back to the PR.

| Command | Who can use | Description |
|---------|-------------|-------------|
| `/bot run` | Allowed users | Mirror PR to GitLab and run CI pipeline |
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
