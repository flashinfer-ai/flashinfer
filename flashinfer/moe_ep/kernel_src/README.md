# kernel_src — vendored kernel snapshots

Each snapshot lives under `sm<arch>/<tree_name>/`, such as
`sm100/cutedsl_megamoe` or `sm107/next_cutedsl_megamoe`. Its `VENDOR.md`
records the upstream revision; `SKILL.md` describes how to update it.
Backends under `backends/mega/kernel/sm<arch>/<dtype-style>/` wrap these kernels.

## The one rule: `src/` is verbatim

The `src/` tree of every drop is a byte-for-byte copy of its upstream source
or pinned upstream exporter output, as specified in that drop's `VENDOR.md`.
For exported drops, record the exporter revision, selected entry points and
transformations, and compare against a regenerated export.
**Do not edit it — not for bugs, and not for style.** That explicitly includes
docstrings, comments, formatting, lint appeasement, type annotations, and
import sorting. `diff -r` against the declared upstream drop/export must be clean;
every local byte of drift makes the next re-sync harder and hides real
divergence.

This rule outranks tooling. When a linter, docstring-coverage gate, or AI
review bot (CodeRabbit, etc.) flags files under a `src/` tree, the fix is to
exclude the path from the check — never to "fix" the vendored file. Reviewers:
style findings inside `src/` are not actionable.

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
