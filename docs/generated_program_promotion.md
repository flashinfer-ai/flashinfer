# Deterministic generated-program promotion

`tools/import-generated-program-promotion` installs generated CUDA or cubin
artifacts without modifying their bytes. The producer supplies a JSON manifest;
the importer verifies every source before writing, copies through same-filesystem
temporary files, atomically replaces each destination, then rehashes both sides.

The schema is deliberately workload-neutral:

```json
{
  "artifacts": [
    {
      "destination": "csrc/example/kernel.cu",
      "executable": false,
      "sha256": "<64 lowercase hexadecimal characters>",
      "size_bytes": 1234,
      "source": "cuda/kernel.cu"
    }
  ],
  "kind": "flashinfer.generated_program_promotion",
  "mode": "cuda",
  "name": "example-program",
  "schema_version": 1
}
```

`mode` is either `cuda` or `cubin`. Paths are normalized POSIX-relative paths,
artifacts are sorted by destination, and no source or destination may traverse
a symlink. Duplicate JSON keys, unknown fields, repeated paths, parent/child
destination collisions, undeclared payload files, byte-count mismatches, and
digest mismatches are fatal. The payload directory is a dedicated exact
inventory; unrelated files already present in the destination checkout are not
part of that producer inventory.

Import into a checkout:

```bash
tools/import-generated-program-promotion \
  --manifest /path/to/manifest.json \
  --payload-root /path/to/payload \
  --output-root "$PWD" \
  --mode cuda
```

Existing identical files are accepted. Updating an existing artifact requires
the explicit `--replace` flag. CI and reviewers can perform a read-only
source-to-destination identity check with `--check`:

```bash
tools/import-generated-program-promotion \
  --manifest /path/to/manifest.json \
  --payload-root /path/to/payload \
  --output-root "$PWD" \
  --mode cuda \
  --check
```

Performance and correctness receipts remain separate inputs to promotion.
They decide whether an artifact is acceptable; this importer only proves that
the accepted artifact is the one copied into the public tree.

## Multi-target promotion packs

`tools/pack-generated-program-promotion` converts sanitized per-architecture
receipts into one importer payload. It accepts any generated-program family;
the workload name and installed runtime-manifest destination are explicit:

```bash
tools/pack-generated-program-promotion \
  --input sm100a=/path/to/sm100a-public \
  --input sm103a=/path/to/sm103a-public \
  --mode cubin \
  --name example-program \
  --runtime-manifest-destination csrc/example/runtime.json \
  --output /path/to/packed-promotion
```

Each input must contain only `promotion-receipt.json` and its declared files.
The packer verifies their bytes and executable modes, canonical
`runtime_inventory_identity`, architecture, selected mode, and shared
correctness/performance denominators. Target-prefixed payload paths prevent
same-named artifacts from colliding. The result is consumed by the existing
importer; it does not create a second installation mechanism.

Runtime adapters are workload-specific and are themselves eligible manifest
artifacts. For FP32 indexed recurrent-KDA prefill, the stable
`flashinfer.jit.kda_fp32_indexed_promotion` adapter validates
`csrc/kda/kda_fp32_indexed_promotion_manifest.json`. The checked-in manifest
contains the complete SM100a and SM103a route denominator and selects one exact
representation for both targets; existing BF16 dispatch does not probe this
adapter.

A complete runtime manifest fixes the public operation contract, exact SM100a
and SM103a file closures, an ordered `modules[]` inventory, ordered
dispatch-policy seed identities, each route's exact ordered module sequence,
immutable routes, and a hash-bound producer dispatcher. A seed may be shared by
routes with different module sequences; the combined dispatcher, seed, and
route identity binds both. The manifest also selects exactly one representation:

- `cuda` declares hashed generated sources, host bindings, an exact published
  build recipe, and every expected cubin digest. The recipe's ordered output
  report and rebuilt bytes must exactly match the manifest.
- `cubin` declares hashed target cubins and exact prebuilt TVM-FFI host shared
  libraries. The runtime loads the verified `.so`; it does not compile or
  rewrite host or device code.

The two representations carry the same representation-independent dispatcher,
seed, route, and contract identity. Users select one explicitly when packing
and when loading. The runtime never silently falls back from one representation
to the other.

`load` requires the caller to name `mode="cuda"` or `mode="cubin"`, and rejects
a mode different from the installed manifest. It binds only the fixed public
dispatcher ABI and its exact ordered module inventory. `prepare` selects and
caches one immutable route together with its launch arguments;
`Prepared.launch` is the repeated hot path, `Prepared.close` releases the bound
closure, and `run` is the prepare/launch/close convenience entry point.
`is_available` becomes true only after the exact modules and dispatcher bind
successfully. A missing artifact, single-byte mutation, reordered module,
unknown route or host-activity role, rebuilt-cubin mismatch, unsupported
architecture, or contract change fails closed.
