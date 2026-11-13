# Update the Kimi K3 export

1. Read `VENDOR.md` and the repository's kernel-source rules.
2. Check out the Kimi revision and obtain the exact base container named by its
   install manifest. Treat the overlay and its base as one pinned export input.
3. Update `scripts/export_kimi_k3_megamoe.py` with the new pins and any required
   compatibility transforms. Keep device arithmetic changes separate from glue.
4. Run the exporter, then rerun it with `--check`. Review every manifest hash change.
5. Run CPU config/import/packaging checks and Blackwell numerical, graph and
   multi-rank tests. Check coexistence with current MegaMoE selectors.
6. Update this tree's source record, local-diff list and measured support matrix.

Do not edit generated `src/` files, import a current MegaMoE provider, modify
`sys.path`, or register TensorRT-LLM/global `megamoe` operators.
