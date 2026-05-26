#!/usr/bin/env bash
# List all @flashinfer_api-decorated APIs (module-level and class methods),
# grouped by class (or "[Global Functions]" per file), with full multi-line
# signatures preserved.
#
# Usage:
#   scripts/list_apis.sh [-n] [-p] [-M] [-d] [--ref REF] [path...]
#
# Options:
#   -n, --no-lines       Omit line numbers
#   -p, --no-paths       Omit file paths (implies -n; signatures-only output)
#   -M, --methods-only   Skip module-level functions; only show class methods
#   -d, --deterministic  Sort files for stable, diff-friendly output
#                        (disables rg's parallel walk; slightly slower)
#   -r, --ref REF        Run against a git revision (tag/branch/sha) via temp worktree
#   -h, --help           Show this help
#
# Default path is flashinfer/
#
# Examples:
#   scripts/list_apis.sh --ref v0.6.9 -p
#   diff -u <(scripts/list_apis.sh -d -p --ref v0.6.9) <(scripts/list_apis.sh -d -p)

set -euo pipefail

# Dependency check — rg (ripgrep) is non-standard; fail early with a clear message
# rather than letting `set -e` surface a cryptic "command not found" mid-pipeline.
if ! command -v rg >/dev/null 2>&1; then
  echo "Error: ripgrep (rg) is required but not found. Install via your package manager (e.g. 'pacman -S ripgrep' / 'apt install ripgrep')." >&2
  exit 1
fi

show_lines=1
show_paths=1
include_global=1
deterministic=0
ref=""
paths=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--no-lines) show_lines=0; shift ;;
    -p|--no-paths) show_paths=0; show_lines=0; shift ;;
    -M|--methods-only) include_global=0; shift ;;
    -d|--deterministic) deterministic=1; shift ;;
    # Guard against `--ref` as the last arg — bare `shift 2` would trip `set -e`
    # with an opaque "shift count out of range" instead of a useful message.
    -r|--ref)
      [[ $# -lt 2 ]] && { echo "Error: $1 requires an argument" >&2; exit 1; }
      ref="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; paths+=("$@"); break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done

if [[ -n "$ref" ]]; then
  repo_root=$(git rev-parse --show-toplevel)
  if ! git -C "$repo_root" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
    for remote in upstream origin; do
      git -C "$repo_root" remote get-url "$remote" >/dev/null 2>&1 || continue
      echo "fetching $ref from $remote..." >&2
      if git -C "$repo_root" fetch --quiet "$remote" "tag" "$ref" 2>/dev/null \
         || git -C "$repo_root" fetch --quiet "$remote" "$ref" 2>/dev/null; then
        break
      fi
    done
    git -C "$repo_root" rev-parse --verify --quiet "$ref^{commit}" >/dev/null \
      || { echo "ref '$ref' not found locally or on remotes" >&2; exit 1; }
  fi
  wt=$(mktemp -d -t fi-apis-XXXXXX)
  trap 'git -C "$repo_root" worktree remove --force "$wt" >/dev/null 2>&1; rm -rf "$wt"' EXIT
  git -C "$repo_root" worktree add --detach --quiet "$wt" "$ref"
  [[ ${#paths[@]} -eq 0 ]] && paths=("flashinfer/")
  paths=("${paths[@]/#/$wt/}")
fi

[[ ${#paths[@]} -eq 0 ]] && paths=("flashinfer/")

# rg is parallel by default — file order is nondeterministic across runs.
# --sort path forces single-threaded but stable ordering; only opt-in (via -d)
# since most interactive use doesn't care, and parallel is faster.
rg_sort=()
[[ "$deterministic" -eq 1 ]] && rg_sort=(--sort path)

rg -HUn -U "${rg_sort[@]}" \
   "^class \w+[^\n]*:|^\s*@flashinfer_api(?:\([^)]*\))?|^\s*def \w+\([\s\S]*?\) *(?:-> *[^:]+)?:" \
   "${paths[@]}" \
| awk -v show_lines="$show_lines" -v show_paths="$show_paths" \
      -v include_global="$include_global" -v strip="${wt:-}/" '
    function emit(line,    out) {
      out = line
      if (strip != "/" && index(out, strip) == 1) out = substr(out, length(strip) + 1)
      if (!show_lines && !show_paths) sub(/^[^:]+:[0-9]+:/, "", out)
      else if (!show_lines)           sub(/:[0-9]+:/, ":", out)
      else if (!show_paths)           sub(/^[^:]+:/, "", out)
      print out
    }
    function flush(    n, i, parts) {
      if (pending) {
        n = split(pending, parts, "\n")
        for (i = 1; i <= n; i++) emit(parts[i])
        pending = ""; in_def = 0
      }
    }
    {
      # Split rg output "path:lineno:content" robustly. Anchoring on the
      # numeric :N: separator (not just the first colon) avoids breaking on
      # paths that contain colons (e.g. Windows-style or unusual filenames).
      if (match($0, /:[0-9]+:/)) {
        path = substr($0, 1, RSTART - 1)
        content = substr($0, RSTART + RLENGTH)
      } else {
        # Fallback: no lineno (rg without -n, or weird input)
        path = $0; sub(/:.*/, "", path)
        content = $0; sub(/^[^:]+:/, "", content)
      }

      if (path != lastpath) { flush(); cls=""; deco=""; printed=0; global_printed=0; lastpath=path }
      if (content ~ /^class /)                { flush(); cls=$0; printed=0; deco=""; next }
      if (content ~ /^[ \t]+@flashinfer_api/) { flush(); deco=$0; next }
      # Top-level (column 0) decorator — only meaningful with --global
      if (content ~ /^@flashinfer_api/ && include_global) { flush(); deco=$0; cls=""; next }

      if (content ~ /^[ \t]+def /) {
        flush()
        if (deco != "" && cls != "") {
          if (!printed) { emit(cls); printed=1 }
          emit(deco); pending = $0; in_def = 1
        }
        deco = ""
        next
      }
      # Module-level def — emit only when decorated AND --global is on.
      if (content ~ /^def / && deco != "" && include_global) {
        flush()
        if (!global_printed) {
          # Synthetic header to keep output structure consistent with class groups.
          emit(path ":0:[Global Functions]")
          global_printed = 1
        }
        emit(deco); pending = $0; in_def = 1
        deco = ""
        next
      }
      if (content ~ /^def /) { deco = ""; next }

      if (in_def) pending = pending "\n" $0
    }
    END { flush() }
'
