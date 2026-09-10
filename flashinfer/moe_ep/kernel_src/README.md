# kernel_src — kernel implementations and vendored snapshots

Directories group related kernel implementations. Vendored packages record
their upstream snapshot in `VENDOR.md` and drop-update workflow in `SKILL.md`.
`cutedsl_megamoe/src/moe_nvfp4_w4a16/` is a scoped FlashInfer-owned precision
addition beside the existing NVFP4, MXFP8, and BF16 implementations; it is
maintained locally and remains subject to formatting, lint, and type checks.
Layout is by **provenance**, not taxonomy: the user-facing
`sm<arch>/<dtype-style>` organization lives in `backends/mega/kernel/`, which
wraps these drops.

## Vendored packages remain verbatim

The existing vendored packages within each `src/` tree are byte-for-byte copies
of their recorded upstream drop. The owned W4A16 package named above is the
explicit exception; it does not change ownership of its vendor siblings.
**Do not edit it — not for bugs, and not for style.** That explicitly includes
docstrings, comments, formatting, lint appeasement, type annotations, and
import sorting. `diff -r` against the upstream drop must come back clean;
every local byte of drift in a vendored package makes the next re-sync harder
and hides real divergence.

This rule outranks tooling. When a linter, docstring-coverage gate, or AI
review bot (CodeRabbit, etc.) flags files under a `src/` tree, the fix is to
exclude the path from the check — never to "fix" the vendored file. Reviewers:
style findings inside vendored packages are not actionable. Findings in the
owned W4A16 package are actionable.

## Import layering

Access flows one direction only (full rules in the ``flashinfer.moe_ep``
package docstring): a drop's `shim/` is the only code that imports its `src/`;
backends are the only consumers of a drop, and only via the drop's package
`__init__` (never shim submodules); the layer/modes/core tiers use backend
APIs only. Sole exception: kernel-oracle tests may import a drop's package
`__init__` to validate it below the backend.

## Where changes actually go

- **Adaptation** (APIs, torch glue, caching, autotune plumbing): the drop's
  `shim/` layer, re-exported through the drop's `__init__.py`. Backends import
  the package `__init__` only, never `src/` directly.
- **Bug fixes**: upstream first, then re-sync the drop. If an emergency local
  edit is unavoidable, record it in the drop's `VENDOR.md` under
  "Pending local diffs vs upstream" until the next drop absorbs it.
- **New kernels from a new upstream repo**: a new sibling directory here, with
  its own `VENDOR.md`. One directory = one upstream commit; do not merge two
  upstream repos (or two commits of one repo) into one tree.
