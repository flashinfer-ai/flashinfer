#!/usr/bin/env bash
#
# ci_no_cooperative.sh
# --------------------
# Static regression guard for the moe-monokernel-software-grid-sync spec.
# Scans `vllm/csrc/moe/moe_monokernel/` for any re-introduction of the
# cooperative-launch / cooperative-groups discipline that Phase 1 of the
# spec migrated the kernel away from.
#
# Checks performed (each ties back to a Requirement-1 clause):
#
#   [R1.1] `cudaLaunchCooperativeKernel(` — cooperative launch must not
#          reappear; the launcher must use `cudaLaunchKernel`.
#   [R1.2] `cooperative_groups::this_grid(` — grid-scope sync must go
#          through `moe_monokernel::grid_barrier<>` / `partial_barrier()`,
#          not `cooperative_groups::this_grid()`.
#   [R1.3] `#include <cooperative_groups.h>` — the header must not be
#          re-included now that the cooperative-groups primitives are no
#          longer used by the kernel.
#   [R1.4] `-rdc=true` — separable compilation is not required; do not
#          re-enable it in any CMake / build fragment.
#
# Exit code: 0 if all four checks are clean, 1 on the first violation
# found (all four checks still run before the exit so the operator sees
# every regression at once).
#
# The script runs from any working directory; SRC_ROOT is resolved from
# the script's own location so it works equally well from
# `tests/`, from the repo root, or from any other cwd.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "CI: moe-monokernel-software-grid-sync spec Requirement-1 regression guard"
echo "  Scanning: ${SRC_ROOT}"
echo

failures=0

# Filter out obvious non-code contexts so documentation references to
# the forbidden symbols (e.g. "formerly cooperative_groups::this_grid()")
# do NOT trigger failure.  We deliberately DO NOT try to implement a
# full comment parser — we just strip the common cases:
#
#   * C++ line comments:            `  // cooperative_groups::this_grid().sync() ...`
#   * C-style block-comment lines:  `  /* ... */`, `  * ...`, `  */`
#   * Markdown-style doc references inside backticks (e.g.
#     `` `cooperative_groups::this_grid()` ``), which typically appear
#     inside a block comment whose continuation lines are just indented
#     prose (no ` * ` prefix).
#
# `match_literal` is the exact literal string the check is looking for
# (paren-stripped where relevant); we use `index()` rather than a regex
# so we don't have to juggle regex escaping of the match string inside
# awk's own regex engine.
#
# New actual code usage (a real `cooperative_groups::this_grid().sync();`
# statement, or a fresh `#include <cooperative_groups.h>` directive) is
# NOT in any of those contexts and WILL be caught.
filter_noncode() {
  local match_literal="$1"
  awk -v pat="${match_literal}" '
    {
      # Split off the "file:lineno:" prefix that grep -n prepends.
      i1 = index($0, ":")
      if (i1 == 0) { next }
      rest = substr($0, i1 + 1)
      i2 = index(rest, ":")
      if (i2 == 0) { next }
      code = substr(rest, i2 + 1)
      # Strip leading whitespace.
      sub(/^[ \t]*/, "", code)
      # Skip C++ line comments.
      if (code ~ /^\/\//) { next }
      # Skip block-comment openers / bare continuation / closers:
      #   /* ...        * ...        */
      if (code ~ /^\/\*/) { next }
      if (code ~ /^\*\//) { next }
      if (code ~ /^\*[ \t]/) { next }
      # Skip when the forbidden pattern is wrapped in Markdown-style
      # backticks (`...`) — typical of doc references inside a block
      # comment whose continuation line has no ` * ` prefix.  We scan
      # for the pattern via literal index(), then inspect the bytes
      # before the match for a `-in-the-same-line and the bytes after
      # for a matching closing backtick (with no intervening backtick).
      pos = index(code, pat)
      if (pos > 0) {
        prefix = substr(code, 1, pos - 1)
        suffix = substr(code, pos + length(pat))
        # Count backticks before (opening) and find a closing one after.
        has_open = (index(prefix, "`") > 0)
        has_close = (index(suffix, "`") > 0)
        if (has_open && has_close) { next }
      }
      print $0
    }
  '
}

check() {
  local pattern="$1"
  local req="$2"
  local explanation="$3"
  # Literal string used for noncode-filter matching (index() in awk).
  # Strip the regex escapes we inserted into $pattern so the awk filter
  # can do a plain substring search.  For the include-directive pattern
  # we fall back to a stable literal that always appears in a real
  # include line.
  local literal="${pattern//\\/}"
  case "${pattern}" in
    *'#include'*)
      literal='cooperative_groups.h'
      ;;
  esac
  echo -n "  [R${req}] Checking for '${pattern}' ... "
  local matches=""
  local raw
  # `grep -e` so patterns beginning with '-' (e.g. -rdc=true) aren't
  # parsed as grep options.  `|| true` so `set -e` doesn't bail on the
  # expected "no match, exit 1" from grep.
  raw=$(grep -rnE -e "${pattern}" "${SRC_ROOT}" \
            --include='*.h'            \
            --include='*.hpp'          \
            --include='*.cuh'          \
            --include='*.cu'           \
            --include='*.c'            \
            --include='*.cc'           \
            --include='*.cpp'          \
            --include='CMakeLists.txt' \
            --include='*.cmake'        \
            --include='*.bazel'        \
            --include='*.sh'           \
            --exclude='ci_no_cooperative.sh' \
        2>/dev/null || true)
  if [ -n "${raw}" ]; then
    matches=$(printf '%s\n' "${raw}" | filter_noncode "${literal}")
  fi
  if [ -n "${matches}" ]; then
    echo "FAIL"
    echo "      ${explanation}"
    echo "      Matches:"
    printf '%s\n' "${matches}" | sed 's/^/        /'
    failures=$((failures + 1))
  else
    echo "clean"
  fi
}

check 'cudaLaunchCooperativeKernel\(' "1.1" \
  "Cooperative launch must not reappear; launcher must use cudaLaunchKernel (spec R1.1, design Component C 'Launch form'). Cooperative launch blocks CUDA Graph capture (spec R5)."

check 'cooperative_groups::this_grid\(' "1.2" \
  "Grid-scope sync must go through moe_monokernel::grid_barrier<> / partial_barrier(), not cooperative_groups::this_grid() (spec R1.2, R3.1)."

check '^[[:space:]]*#include[[:space:]]*<cooperative_groups\.h>' "1.3" \
  "cooperative_groups header must not be included; cooperative_groups primitives are no longer used by the kernel (spec R1.3)."

check '-rdc=true' "1.4" \
  "-rdc=true must not be re-enabled in any CMake or build fragment; separable compilation is not required for the software-barrier kernel (spec R1.4)."

echo
if [ "${failures}" -eq 0 ]; then
  echo "All 4 Requirement-1 regression checks passed."
  exit 0
else
  echo "Requirement-1 regression guard: ${failures} violation(s). Spec R1 regressed."
  exit 1
fi
