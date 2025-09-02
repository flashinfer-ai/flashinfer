# Installation

For development, the easiest way to install flashinfer is through editable installation:

```
git clone git@github.com:flashinfer-ai/flashinfer.git --recursive
pip install --no-build-isolation -e . -v
```

We recommend using the `--no-build-isolation` flag to ensure compatibility with your existing environment. Without it, `pip` may attempt to resolve dependencies (e.g., `torch`) from PyPI, which could pull in packages built with older CUDA versions and lead to incompatibility issues.

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

# Release Versioning

When incrementing a version and creating a release, follow [Semantic Versioning](https://packaging.python.org/en/latest/discussions/versioning/) (`major.minor.patch`) [^1]. In particular:

* major increment signals incompatible API changes
* minor increment signals added functionality that is backwards-compatible (e.g. new kernels, new SM support, etc)
* patch increment signals backwards-compatible bug fixes (both for functional and performance issues)

Optionally, use post-releases (e.g., `X.Y.Z.post1`) for minor changes, like a documentation change.

[^1]: We have not followed this strictly through v0.2.14.post1. But after v0.2.14.post1, the versioning should follow SemVer.
