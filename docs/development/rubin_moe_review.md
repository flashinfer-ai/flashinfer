# Review standard for Rubin MoE changes

**SM100 and SM103 must behave exactly as they did on `origin/main` when run
with their original configurations.** This includes callers that rely on
defaults. Preserve tactic selection, kernel execution, and output behavior.

Name each option for the behavior it controls. Base each decision on the
tensor property, caller setting, or tactic capability that directly governs
it. Keep independent choices separate even when today's tactics always make
the same combination of choices. Reuse existing conventions and add a
Rubin-specific option only for a genuinely new choice.

- Infer prepared NVFP4 weight layout from tensor rank. The packing option
  chooses which layout to produce.
- Keep Rubin non-swap-AB 512/1024 token tiles available with either fused or
  separate finalization.
- Use `routed_native_sfa` to select FC1's scale-factor loader. Resolve PDL
  from the caller's `enable_pdl` request and the selected tactic's `use_pdl`
  setting. Changing the loader alone must never change the PDL setting.
