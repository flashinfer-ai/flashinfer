# Re-syncing the GC MoE drop

`src/` vendors one upstream commit plus the local patches listed in
`VENDOR.md`. Upstream fixes are preferred once permitted; until then a fix may
live in `src/` as a documented local patch.

## Procedure

1. Pick the upstream commit and record it:

   ```bash
   git -C <gc-moe-checkout> log --oneline -1 <commit>
   ```

2. Re-export over the existing tree (delete first so removed files disappear):

   ```bash
   rm -rf flashinfer/experimental/gc_moe/kernel_src/gc_moe/src
   mkdir -p flashinfer/experimental/gc_moe/kernel_src/gc_moe/src
   git -C <gc-moe-checkout> archive <commit> -- gc_moe \
     | tar -x -C flashinfer/experimental/gc_moe/kernel_src/gc_moe/src \
         --strip-components=1
   ```

3. Diff against the upstream checkout and record the new pin:

   ```bash
   diff -r --exclude=__pycache__ \
     flashinfer/experimental/gc_moe/kernel_src/gc_moe/src \
     <gc-moe-checkout>/gc_moe
   ```

   Re-apply the local patches from `VENDOR.md` (each anchors on a unique
   hunk of the pristine tree). Update the commit/date/line counts in
   `VENDOR.md`, `UPSTREAM_COMMIT` and `UPSTREAM_VERSION` in
   `kernel_src/gc_moe/__init__.py`, the pin in `README.md` and the source
   URL in `NOTICE`.

4. Re-check the shim against the new upstream surface:

   - public entries re-exported by `src/__init__.py` (`fp8_fp4_gc_moe`,
     `sm90_fp8_fp4_gc_moe`, `sm120_fp8_fp4_gc_moe`, the three workspace
     classes and the three weight transforms);
   - keyword names used by `shim/launch.py` and `shim/workspace.py`;
   - new modules or `__init__.py` files (affects the package-data note in
     `VENDOR.md`);
   - new `GC_MOE_*` environment variables (extend `plan.py` and the
     fingerprint).

5. Run the hardware tests:

   ```bash
   python -m pytest tests/experimental/gc_moe -q
   ```

   SM90/SM100/SM120 cases skip on hosts without the matching GPU; run the
   matching subset on each architecture before merging.
