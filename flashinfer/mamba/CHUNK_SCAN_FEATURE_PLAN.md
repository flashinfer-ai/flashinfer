# Chunk Scan Combined: Feature Parity Plan

WIP implementation: `flashinfer/mamba/ssd_combined.py` and `flashinfer/mamba/ssd_kernel.py`

Missing features compared to the reference implementation
(`tests/mamba/triton_reference/ssd_combined.py`).

## Priority Table

| Priority | Feature | Impact on Kernel |
|----------|---------|-----------------|
| 1 | **varlen** (`seq_idx` + `chunk_indices`/`chunk_offsets` + `cu_seqlens`) | Fundamental restructure |
| 2 | **`z` (gate)** | New pipeline + epilog changes |
| 3 | **`out` (pre-allocated output)** | Python wrapper only |
| 4 | **`return_final_states` / `return_varlen_states`** | Python wrapper only |
| 5 | **`state_dtype`** | Python wrapper only |

---

**IMPORTANT** We don't care about the order of API input arguments for now, but remember to
update it to match the reference implementation once we're done with all features.

**TODO** Check that if chunk_indices is passed, then chunk_offsets is also passed (and vice versa). They should either both be None (no varlen) or both be non-None (varlen).

## 1. Variable-Length Sequence Support

`seq_idx`, `chunk_indices`/`chunk_offsets`, `cu_seqlens`

**Status**: Not started

### Description

These are effectively a single feature: continuous batching / variable-length
sequence support. They're deeply intertwined:

- **`seq_idx`** `(batch, seqlen)` — maps each position to its sequence ID.
  Used everywhere: in `_bmm_chunk_fwd` to zero out cross-sequence CB entries,
  in `_chunk_state_fwd` to mask contributions, in `_state_passing_fwd` to
  detect sequence boundaries and reset states, and in `_chunk_scan_fwd` to
  adjust scaling.
- **`chunk_indices`/`chunk_offsets`** — handle the case where a physical chunk
  contains the boundary between two sequences. `chunk_indices` maps logical
  (per-sequence) chunks to physical chunk IDs, and `chunk_offsets` gives the
  start position within that physical chunk. This creates "pseudo-chunks" split
  at sequence boundaries.
- **`cu_seqlens`** `(num_seq+1,)` — cumulative sequence lengths for packed
  variable-length inputs. Used in `chunk_state_varlen` to compute per-sequence
  final states.

### Typical inputs (from vLLM-style serving)

```
x.shape             torch.Size([1, 865, 128, 64])
seq_idx.shape       torch.Size([1, 865])
chunk_indices.shape torch.Size([7])     # tiny
chunk_offsets.shape torch.Size([7])     # tiny
cu_seqlens.shape    torch.Size([2])
initial_states.shape torch.Size([1, 128, 64, 128])
```

### Key architectural observation

The CuTe DSL kernel does NOT need a fundamental restructure. The reason is
that `chunk_indices`/`chunk_offsets` only affect:

1. **Which physical chunk** to TMA-load for each loop iteration
2. **The offset within that chunk** for masking / dA_cumsum boundary
3. **Which init_state to load** when a sequence boundary is crossed

The pipeline structure (4 MMAs, warp specialization, TMEM/SMEM flow) stays
identical. The MMAs themselves don't need masking — cross-sequence masking in
the CB matrix is handled by `_bmm_chunk_fwd` (a separate Triton kernel).

### What stays the same

- All 4 MMA operations (INTRA1, INTRA2, INTER1, INTER2)
- Pipeline stage counts and barrier management
- SharedStorage layout, TMEM allocation
- The tile scheduler grid decomposition over (B, EH) pairs
- The epilog warp (combine intra+inter, D scaling, TMA store Y)

### What changes

The changes are organized by warp specialization:

#### Step 1: Python wrapper (`ssd_combined.py`) — DONE
- [x] `ssd_combined_fwd()`: added optional params `seq_idx`, `chunk_indices`,
  `chunk_offsets`, `cu_seqlens` with docstrings
- [x] `ssd_combined_fwd()`: passes `seq_idx`, `chunk_indices`, `chunk_offsets`
  through to `kernel.run()`
- [x] `_SSDKernel.run()`: accepts `seq_idx`, `chunk_indices`, `chunk_offsets`
  (currently unused — ready for wiring into the CuTe kernel)

Note: The only Triton kernel in `ssd_combined.py` is `chunk_cumsum_fwd`
(step 1). Steps 2-5 (chunk_state, state_passing, bmm_chunk, chunk_scan)
are all fused into the single CuTe DSL kernel. There are no separate
sub-kernel calls to wire `seq_idx`/`chunk_offsets` into.

#### Step 2: Kernel arguments — pass chunk_indices/chunk_offsets through

The goal is to thread `chunk_indices` and `chunk_offsets` from
`_SSDKernel.run()` all the way down to the GPU `kernel()` function.
No behavioral changes yet — just plumbing.

These are tiny int32 tensors (~7 elements). No TMA needed — they'll be
scalar-loaded from global memory (L1-cached, uniform across all threads).

**Files and locations to change:**

All three varlen tensors (`seq_idx`, `chunk_indices`, `chunk_offsets`)
are plumbed through. `cu_seqlens` stays at Python level only (used for
`chunk_state_varlen` after the kernel, not inside it).

`ssd_combined.py` — `_SSDKernel.run()`:
- [x] (a) Convert PyTorch tensors to CuTe tensors via `from_dlpack`
- [x] (b) Add to `cute.compile(self.kernel, ...)` args
- [x] (c) Add to `self._compiled_kernel(...)` args

`ssd_kernel.py` — `SSDKernel.__call__()`:
- [x] (d) Add `seq_idx`, `chunk_indices`, `chunk_offsets` params
- [x] (e) Pass them through in `self.kernel(...)` call

`ssd_kernel.py` — `SSDKernel.kernel()`:
- [x] (f) Add `seq_idx`, `chunk_indices`, `chunk_offsets` params to
  the `@cute.kernel` signature

#### Step 3: Kernel inner loop — replace C dimension indexing

Currently every warp's chunk loop does `for chunk_idx in cutlass.range(C)`.
The TMA producer warps use `producer_state.count` (which equals the loop
iteration) to index the C dimension of global tensors. Consumer warps and
the epilog use `chunk_idx` or pipeline stage indices.

The test now uses real variable lengths (chunk-aligned: `[1,1]`,
`[2,2]`, `[1,2,1]`). With chunk-aligned sequences, `chunk_indices`
is still identity and `chunk_offsets` is all zeros, but the test
validates the full Triton reference varlen path. Non-chunk-aligned
lengths (step 4.2) are not yet tested.

**Key insight**: only the TMA producer warps and the epilog TMA store
index the C dimension of *global* tensors. The MMA warps, pre-intra, and
pre-inter warps only consume from smem/tmem pipelines — they never index
C directly. So the changes are confined to 3 warps.

**Warps that need changes:**

- [x] (a) **TMA load X/Delta/CumsumDelta warp** (L1047-1093):
  Currently:
  ```python
  for chunk_idx in cutlass.range(C, unroll=1):
      cute.copy(tma_atom_x, tXgX[None, x_producer_state.count], ...)
      cute.copy(tma_atom_delta, tDeltagDelta[None, deltas_producer_state.count], ...)
      cute.copy(tma_atom_cumsum_delta, tDeltagCumsumDelta[None, deltas_producer_state.count], ...)
  ```
  Change: replace `producer_state.count` with a lookup into
  `chunk_indices` (for now, identity — same value).

- [x] (b) **TMA load B/C warp** (L1167-1201):
  Currently:
  ```python
  for chunk_idx in cutlass.range(C, unroll=1):
      cute.copy(tma_atom_b, tBgB[None, b_producer_state.count], ...)
      cute.copy(tma_atom_c, tCgC[None, c_producer_state.count], ...)
  ```
  Same change as (a).

- [x] (c) **Epilog warp — TMA store Y** (L2516):
  Currently:
  ```python
  bSG_gY[None, epi_m, epi_n, chunk_idx]
  ```
  Change: replace `chunk_idx` with `chunk_indices[chunk_idx]`.

**Warps that DON'T need changes for step 3:**

- **MMA Intra warp**: consumes from pipelines, no C indexing.
- **MMA Inter warp**: consumes from pipelines, no C indexing.
- **Pre-Inter warp** (state passing): consumes from pipelines. The
  `last_column` read and `tState` update logic will need `chunk_offsets`
  in step 4, but not step 3.
- **Pre-Intra warp** (segsum): consumes from pipelines, no C indexing.
  Will need `chunk_offsets` in step 4 for dA_cumsum boundary.

#### Step 4.1: dA_cumsum boundary adjustment (`c_off > 0`)

When `c_off > 0` (logical chunk starts mid-physical-chunk), the
dA_cumsum values need a boundary adjustment: all exponentials should
be relative to `dA_cumsum[c_off - 1]` instead of 0. With uniform
chunk lengths (HACK test), `c_off == 0` always, so these changes are
no-ops until we restore real varlen.

The boundary value is:
```python
dA_cs_boundary = smem_cumsum_delta[c_off - 1, stage] if c_off > 0 else 0.0
```

**Locations that use dA_cumsum and need boundary adjustment:**

- [x] (a) **Pre-inter warp — `last_column` and B scaling** (L1849-1861):
  Currently:
  ```python
  last_column = smem_cumsum_delta[smem_cumsum_delta.shape[0] - 1, deltas_consumer_state.index]
  tScaledB = self.pre_inter_scale_bt_with_delta(tBrB_s2r, tBrDelta_r2s, tBrDeltaA_r2s, last_column)
  ```
  `pre_inter_scale_bt_with_delta` (L3438-3456) computes
  `exp(last_column - dA_cs[i]) * dt[i] * B[i]`. With `c_off > 0`,
  `last_column` should become `last_column - dA_cs_boundary` so the
  decay is relative to the sequence start within the chunk.

- [x] (b) **Pre-inter warp — state recurrence** (L1901-1912):
  Currently:
  ```python
  exp_last_column = cute.math.exp(last_column, fastmath=True)
  tTR_rP[i] = exp_last_column * tState[i] + tTR_rP[i]  # (via fma)
  ```
  The `exp(last_column)` scales the previous state by the full-chunk
  decay. With `c_off > 0`, this should be
  `exp(last_column - dA_cs_boundary)` so only the decay from the
  sequence's start within the chunk is applied.
  **Note**: since step 4.1(a) already adjusts `last_column`, and this
  code uses the same `last_column` variable, this is already handled.
  Just needs verification with real varlen tests.

- [x] (c) **Epilog warp — inter-chunk contribution scaling** (L2430-2440):
  Currently:
  ```python
  tRS_rCompute[i] = exp(tTR_rDeltaA[i]) * tTR_rInter[i] + tTR_rIntra[i]
  ```
  `tTR_rDeltaA` is loaded from smem_cumsum_delta. With `c_off > 0`,
  each element should be shifted: `exp(tTR_rDeltaA[i] - dA_cs_boundary)`.

- [x] (d) **Pre-intra warp — segsum** (L3229-3273):
  Currently computes `exp(dA_col[m] - dA_row[n]) * dt[n] * CB[m,n]`
  with a causal mask `m < n → -inf`. **No change needed**: the segsum
  uses dA_cumsum as a *difference* (`dA_col - dA_row`), so subtracting
  `dA_cs_boundary` from both cancels out. Cross-sequence masking of
  positions `[0, c_off)` is handled by the CB matrix (zeroed by
  `_bmm_chunk_fwd` via `seq_idx`).

Note: cross-sequence masking of positions `[0, c_off)` in the physical
chunk is handled by `_bmm_chunk_fwd` (zeroes CB entries where
`seq_idx[i] != seq_idx[j]`), not by the CuTe kernel itself.

#### Step 4.2: `chunk_size_limit` — truncating shared physical chunks

This is a separate concern from step 4.1. When two consecutive logical
chunks map to the **same physical chunk** (`c_idx == c_idx_next`), the
current logical chunk must NOT process positions that belong to the
next logical chunk (i.e., the next sequence).

**Example**: chunk_size=128, physical chunk covers positions 0..127.
Sequence A occupies positions 0..79, sequence B occupies 80..127.
`_compute_varlen_metadata` produces two logical chunks:
```
logical chunk 0: c_idx=0, c_off=0   → sequence A, positions 0..79
logical chunk 1: c_idx=0, c_off=80  → sequence B, positions 80..127
```
When processing logical chunk 0, we must limit computation to
positions 0..79 (i.e., `chunk_size_limit = c_off_next = 80`) instead
of the full 0..127. Without this, logical chunk 0 would include
data from sequence B in its state computation.

**How the Triton reference handles it** (`ssd_chunk_scan.py` L282-292):
```python
c_idx_n = chunk_indices[pid_c + 1]
if c_idx == c_idx_n:
    c_off_n = chunk_offsets[pid_c + 1]
    chunk_size_limit = min(c_off_n, chunk_size_limit)
```
This limits all load masks and output store masks to `chunk_size_limit`,
effectively ignoring positions beyond the sequence boundary.

**Challenge in the CuTe kernel**: TMA always loads full L-sized tiles
into smem. We can't partially load — the data for both sequences will
be in smem. The separation must happen at the computation level:

- **CB matrix** (`_bmm_chunk_fwd`): Already zeroed for cross-sequence
  pairs via `seq_idx`. This handles INTRA1 (CB computation) and
  INTRA2 (CB @ x) — positions from the wrong sequence contribute 0.
- **B scaling in pre-inter** (`pre_inter_scale_bt_with_delta`): The
  scaled B is used for INTER1 (chunk state = B_scaled^T @ X). Positions
  beyond `chunk_size_limit` belong to the next sequence and must be
  zeroed in the scaled B, otherwise they corrupt the chunk state.
- **C in INTER2** (C @ state): C values beyond `chunk_size_limit`
  are for the wrong sequence but multiply the *correct* state, so they
  produce wrong output for those positions. However, those positions
  will be *overwritten* when the next logical chunk processes them.
- **Epilog Y store**: Positions beyond `chunk_size_limit` get written
  by the current logical chunk but will be overwritten by the next
  logical chunk (which loads the same physical chunk with `c_off > 0`).

**Status**: Not yet implemented. Only matters with non-chunk-aligned
sequence lengths.

**Analysis of which computation paths are affected:**

Walking through the algorithm for logical chunk 0 (seq A, positions
0..79) when the full physical chunk (0..127) is loaded into smem:

1. **INTRA1 (CB = C @ B^T)**: CB is precomputed by `_bmm_chunk_fwd`
   with `seq_idx` masking → cross-sequence entries already zeroed. **OK.**

2. **INTRA2 (Q @ X)**: Q comes from segsum(CB) which is already
   masked. X has seq B data in positions 80..127, but they get
   multiplied by zero Q entries. **OK.**

3. **INTER1 (B_scaled^T @ X → chunk state)**: `pre_inter_scale_bt_with_delta`
   produces scaled B for ALL 128 positions. INTER1_MMA sums over all L
   positions → **WRONG: seq B data in positions 80..127 contaminates
   seq A's chunk state.** This is the critical failure path.

4. **State passing** (`tState` recurrence in pre-inter): Uses the
   corrupted chunk state from step 3 → **WRONG by propagation.**

5. **INTER2 (C @ state)**: C for positions 80..127 belongs to seq B
   but multiplies the (corrupted) state. However, these positions will
   be overwritten by logical chunk 1. **Output OK after overwrite, but
   state is already corrupted.**

6. **Epilog Y store**: Positions 80..127 get wrong values but are
   overwritten by logical chunk 1. **Output OK after overwrite.**

**Conclusion**: The only critical fix is in the **pre-inter warp's B
scaling** (step 3 above). We need to zero out the scaled B entries
for positions beyond `chunk_size_limit` before they enter INTER1_MMA.

**Implementation approach**: Use a coordinate tensor (same pattern as
the pre-intra warp's `tCoord` for causal masking in segsum). Steps:

1. Create `cute.make_identity_tensor()` matching the B tile shape
2. Partition it with the same `tiled_s2r_b` to get per-register L coords
3. After `pre_inter_scale_bt_with_delta` produces `tScaledB`, zero
   entries where L coord >= `chunk_size_limit`
4. This stays entirely in registers — no smem writes, no barriers,
   no TMA involvement. The pre-inter warp has 168 registers allocated,
   plenty of room for a small coordinate fragment.

**Action items**:
- [ ] (a) Write a test with non-chunk-aligned sequence lengths (e.g.,
  seq lengths 80 and 176 packed into 256 = 2 physical chunks of 128)
  to verify the reference works and expose the failure in our kernel
- [ ] (b) Compute `chunk_size_limit` in the pre-inter warp chunk loop:
  ```python
  # At L1831, inside the chunk loop:
  c_idx_next = chunk_indices[chunk_idx + 1] if chunk_idx + 1 < C else -1
  chunk_size_limit = L
  if c_idx_next == c_idx:
      chunk_size_limit = chunk_offsets[chunk_idx + 1]
  ```
- [ ] (c) Implement masking in pre-inter warp. Between the scaling
  (L1872-1873) and the store to `bt_smem_internal` (L1882), zero out
  `tScaledB` entries where L coord >= `chunk_size_limit`:

  **Setup** (once, before the chunk loop, ~L1640-1650):
  ```python
  # Create identity tensor matching smem_bt shape (N, L) without stages
  bt_coord_tensor = cute.make_identity_tensor(
      cute.dice(smem_bt.layout.shape, (1, 1, None))  # drop INPUT_STAGE dim
  )
  thr_s2r_b = tiled_s2r_b.get_slice(local_tidx)
  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
  tBCoord = thr_s2r_b.partition_D(bt_coord_tensor)
  ```

  **Masking** (inside chunk loop, after `pre_inter_scale_bt_with_delta`,
  before the `tBrB_r2s[i] = tScaledB[i]` store loop at L1878):
  ```python
  if chunk_size_limit < L:
      for reg_idx in range(cute.size(tScaledB)):
          n_coord, l_coord = tBCoord[reg_idx]
          if l_coord >= chunk_size_limit:
              tScaledB[reg_idx] = 0.0
  ```
  Note: the coordinate order (n, l) vs (l, n) depends on smem_bt's
  layout — verify with a print at dev time. The L dim is the K dim
  of `tile_shape_mnk_inter1 = (N, D, L)`.

#### Step 5.1: init_states reload for chunk-aligned sequence transitions

**Scope**: chunk-aligned sequences only (all `chunk_offsets` are 0).
Each logical chunk belongs to exactly one sequence, so at most one
init_state load per sequence transition.

Currently init_states are loaded **once** before the chunk loop (TMA
warp at L1014–1030, consumed in pre-inter at L1777–1789). With varlen,
the chunk loop walks through multiple sequences, so we need to reload
init_states when the sequence changes.

**SMEM constraint**: The kernel already uses 100% of shared memory
(232448 / 232448 bytes — see `ssd_kernel.py` L635–636). No room for
extra buffers. We reuse the existing `init_states_pipeline` stages
and accept an occasional stall on sequence transitions.

**Approach — "look-ahead" at end of each chunk iteration**:
Rather than tracking `prev_seq_id`, we use a look-ahead at the *end*
of each chunk iteration to decide whether to flush/reload:
```
is_last_chunk = (chunk_idx == C - 1)
next_is_new_seq = not is_last_chunk and (chunk_indices[chunk_idx + 1] != c_idx)
store_final_state = is_last_chunk or next_is_new_seq
```
When `store_final_state` is true:
1. Store the current state to gmem (same TMA store as post-loop L2066–2075)
2. If `next_is_new_seq`: release the init_states pipeline, wait for
   TMA warp to load the new init_state, reload into `tState`, and
   re-emit into `inter2_p`.

This keeps the pipeline structure intact — the TMA warp still produces
init_states into the same pipeline, the pre-inter warp consumes them.
The stall only happens at sequence boundaries (once per transition).

**Sequence ID derivation**: Both the TMA warp and the consumer
(pre-inter) already have `chunk_indices` and `chunk_offsets` per
logical chunk, and `seq_idx` is a kernel parameter. The sequence ID
for logical chunk `i` is
`seq_idx[chunk_indices[i] * L + chunk_offsets[i], 0]` (where
`L = chunk_size`). No new arrays need to be passed.

##### Substeps

- [ ] **a.** TMA warp: move init_states load into the chunk loop.

`flashinfer/mamba/ssd_kernel.py` L1014–1030 — currently loads once
before the chunk loop. Move into the chunk loop (L1048–1098). On
each iteration, derive the sequence ID and check for transition:
```python
    prev_seq_id = -1  # before loop
    for chunk_idx in cutlass.range(C, unroll=1):
        if cutlass.const_expr(self.has_init_states and self.has_varlen):
            c_idx = chunk_indices[x_producer_state.count] if chunk_indices is not None else x_producer_state.count
            c_off = chunk_offsets[x_producer_state.count] if chunk_offsets is not None else 0
            seq_id = seq_idx[c_idx * L + c_off, 0]
            if seq_id != prev_seq_id:
                tIstategIstate = tIstategIstate_pre_slice[None, 0, 0, eh_idx, seq_id]
                init_states_pipeline.producer_acquire(istate_producer_state)
                cute.copy(tma_atom_initial_states, tIstategIstate, ...)
                istate_producer_state.advance()
                prev_seq_id = seq_id
        # ... existing X/Delta/CumsumDelta loads ...
```
The non-varlen path keeps the existing pre-loop load unchanged.

- [ ] **b.** Pre-inter consumer: flush state + reload on sequence
transition (look-ahead approach).

**Pipeline flow recap** (crucial for understanding the placement):

The pre-inter warp produces C+1 inter2_p slots total. mma_inter
consumes C+1 slots (1 prefill + C from its loop). We cannot add
extra slots — mma_inter's loop count is fixed at C. So we must
**overwrite** the current slot before committing it.

Per-iteration inter2_p timeline (pre-inter loop body), showing the
full flow including the varlen look-ahead insertion:
```
L1981  inter2_p_pipeline.producer_acquire(...)       # acquire slot
L1985  cute.copy(tiled_t2r_inter1, ..., tTR_rP)     # load INTER1_ACC from tmem
       ...
L2000  tTR_rP = exp(last_col) * tState + tTR_rP     # FMA: new_state
L2012  tRS_rP[i] = tTR_rP[i].to(io_dtype)           # regs f32 → io_dtype
L2015  tState.store(tTR_rP.load())                   # update tState (f32)
L2018  inter2_p_coord = (..., inter2_p_producer_state.index)
L2019  cute.copy(tiled_r2s_p, tRS_rP, tRS_sP[...])  # write state → smem slot
L2023  fence_proxy(async_shared)
L2026  inter1_acc_pipeline.consumer_release(...)

       ─── VARLEN LOOK-AHEAD (inserted, only when has_init_states && has_varlen) ───
       if store_final_state (is_last_chunk or next seq differs):
         barrier(pre_inter_sync)                     # sync all pre_inter warps
         TMA store smem slot → fstate[seq_id] in gmem  # flush final state
         tma_p_pipeline.producer_commit/acquire      # wait for TMA store done
         barrier(pre_inter_sync)                     # sync after TMA store
       if next_is_new_seq (next seq differs, not last chunk):
         init_states_pipeline.consumer_release(...)  # release old init_state smem
         init_states_pipeline.consumer_wait(...)     # wait for TMA warp to load new init_state
         smem_p → tRS_rP → tState                   # load new init_state into regs
         tState → tRS_rP → smem inter2_p slot        # overwrite same slot with new init_state
         fence_proxy(async_shared)
       ─── END VARLEN LOOK-AHEAD ───

L2031  if count < C: inter2_p_pipeline.producer_commit(...)  # ← commit to mma_inter
       ... advance other pipelines ...
L2054  if count < C: inter2_p_producer_state.advance()
```

**Concrete example**: C=4, `chunk_indices` maps to `c_idx=[0,0,1,1]`
(2 sequences, 2 chunks each). Only state-related pipelines shown:
- `IS` = init_states_pipeline (TMA warp → pre_inter)
- `P`  = inter2_p_pipeline (pre_inter → mma_inter), 2 stages: slot0, slot1
- `tma_p` = tma_p_pipeline (pre_inter → gmem fstate store)

```
PRE-LOOP:
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
  IS.acquire                     ...                            ...
  TMA load init_state[seq0]      IS.consumer_wait               ...
  IS.advance                     smem → tRS_rP → tState       ...
                                 P.acquire(slot0)               ...
                                 tState → tRS_rP → smem[P0]   ...
                                 P.commit(slot0), P.advance     ...

chunk_idx=0, c_idx=0 (seq0, chunk 0)
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
  ...                            (process B/delta/scaledB)      P.wait(slot0) → MMA2
                                                                P has init_state[seq0]
                                                                P.release(slot0)
                                 inter1_acc.wait                ...
                                 P.acquire(slot1)               MMA1
                                 new_P=exp(lc)*tState+acc       ...
                                 tState=new_P                   ...
                                 new_P → smem[P1]              ...
                                 P.commit(slot1), P.advance     ...

chunk_idx=1, c_idx=0 (seq0, chunk 1 — last chunk of seq0)
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
  ...                            (process B/delta/scaledB)      P.wait(slot1) → MMA2
                                                                P has state_after_chunk0
                                                                P.release(slot1)
                                 inter1_acc.wait                ...
                                 P.acquire(slot0)               MMA1
                                 new_P=exp(lc)*tState+acc       ...
                                 tState=new_P (=fstate[seq0])   ...
                                 new_P → smem[P0]               ...
                                 *** INSERT LOOK-AHEAD HERE *** ...
                                 P.commit(slot0), P.advance     ...

*** At this point smem[P0] has fstate[seq0]. mma_inter will consume
    slot0 at chunk_idx=2. We must overwrite it with init_state[seq1]
    BEFORE the commit. ***

chunk_idx=2, c_idx=1 (seq1, chunk 0)
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
  ...                            (process B/delta/scaledB)      P.wait(slot0) → MMA2
                                                                 ❌ slot0=fstate[seq0]
                                                                 should be init[seq1]
                                                                P.release(slot0)
                                 inter1_acc.wait                ...
                                 P.acquire(slot1)               MMA1
                                 new_P=exp(lc)*tState+acc       ...
                                 tState=new_P                   ...
                                 new_P → smem[P1]               ...
                                 P.commit(slot1), P.advance     ...

chunk_idx=3, c_idx=1 (seq1, chunk 1 — last)
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
  ...                            (process B/delta/scaledB)      P.wait(slot1) → MMA2
                                                                P.release(slot1)
                                 inter1_acc.wait                ...
                                 P.acquire(slot0)               MMA1
                                 new_P=exp(lc)*tState+acc       ...
                                 tState=new_P (=fstate[seq1])   ...
                                 new_P → smem[P0]               ...
                                 (last iter: NO commit)         ...

POST-LOOP:
  TMA warp                       pre_inter                      mma_inter
  ─────────────────────────────  ─────────────────────────────  ─────────────────────────
                                 barrier(pre_inter_sync)
                                 TMA store smem[P0] → fstate
                                 tma_p.commit, tma_p.acquire
                                 barrier(pre_inter_sync)
                                 tma_p.producer_tail()
                                 IS.consumer_release
```

At the point between L2026 (inter1_acc release) and L2031 (commit):
- The accumulated state for this chunk is in smem (not yet committed)
- `tState` holds the same value in f32
- mma_inter has NOT yet seen this slot

This is the correct insertion point. We can:
1. TMA-store the accumulated state to gmem (final state for the ending sequence)
2. Overwrite the smem slot with the new init_state
3. Reset tState to the new init_state
4. Then let the existing commit fire — mma_inter sees the new init_state

The pre-loop init_state load (L1817–1825) stays for chunk_idx == 0.

**Checklist of changes** (all in `ssd_kernel.py` unless noted):

- [x] **a.** Uncomment TMA warp init_states reload (L1064–1082) — DONE.
  Uncommented the block that detects sequence transitions via
  `seq_id != prev_seq_id` and loads new init_states via the IS pipeline.
  Also fixed `seq_idx` indexing: kernel receives `(batch, seqlen)` layout,
  so index as `seq_idx[0, c_idx * L + c_off]` (not `seq_idx[pos, 0]`).
  Fixed in both the pre-loop first_seq_id lookup and the in-loop reload.
  Removed `.contiguous()` copy from `ssd_combined.py` — zero-copy via
  `from_dlpack`.

- [ ] **b.** Insert look-ahead block between L1980 and L1982.
  Between `inter1_acc_pipeline.consumer_release` (L1980) and
  `inter2_p_pipeline.producer_commit` (L1983). At this point the
  accumulated state is in smem but NOT yet committed to mma_inter.
  ```python
  # L1980: inter1_acc_pipeline.consumer_release(inter1_acc_consumer_state)
  # ─── INSERT HERE ───
  if cutlass.const_expr(self.has_init_states and self.has_varlen):
      is_last_chunk = chunk_idx == C - 1
      next_is_new_seq = False
      if not is_last_chunk:
          next_c_idx = chunk_indices[chunk_idx + 1]
          next_is_new_seq = next_c_idx != c_idx
      store_final_state = is_last_chunk or next_is_new_seq

      if store_final_state:
          # 1. TMA store accumulated state to fstate[seq_id] in gmem
          if local_warp_idx == 0:
              seq_id = seq_idx[c_idx * L + c_off, 0]
              bSG_gP_seq = bSG_gP_pre_slice[(None, 0, 0, eh_idx, seq_id)]
              cute.copy(tma_atom_p,
                        bSG_sP[(None, inter2_p_producer_state.index)],
                        bSG_gP_seq)
              tma_p_pipeline.producer_commit()
              tma_p_pipeline.producer_acquire()

      if next_is_new_seq:
          # 2. Release old IS buffer, wait for new one from TMA warp
          init_states_pipeline.consumer_release(istate_consumer_state)
          istate_consumer_state.advance()
          init_states_pipeline.consumer_wait(istate_consumer_state)

          # 3. Load new init_state: smem → tRS_rP → tState
          istate_coord = (None, None, None, istate_consumer_state.index)
          cute.copy(tiled_s2r_p, tS2R_sP[istate_coord], tRS_rP)
          for reg_idx in range(cute.size(tRS_rP)):
              tState[reg_idx] = tRS_rP[reg_idx].to(self.acc_dtype)

          # 4. Overwrite same inter2_p smem slot with new init_state
          for reg_idx in range(cute.size(tState)):
              tRS_rP[reg_idx] = tState[reg_idx].to(self.io_dtype)
          cute.copy(tiled_r2s_p, tRS_rP, tRS_sP[inter2_p_coord])
          cute.arch.fence_proxy(
              cute.arch.ProxyKind.async_shared,
              space=cute.arch.SharedSpace.shared_cta,
          )
  # L1983: if count < C: inter2_p_pipeline.producer_commit(...)
  ```
  The existing commit fires after — mma_inter sees the overwritten
  init_state. Pipeline slot counts stay the same (no extra slots).

  **CuTe DSL note**: `c_idx` and `c_off` are defined inside the loop
  body but also used after the loop (post-loop store). CuTe DSL
  requires variables used after dynamic control flow to be initialized
  before. Add `c_idx = 0` and `c_off = 0` before the `for` loop.

- [ ] **c.** Post-loop store (L2009–2036): use `seq_id` for varlen.
  Currently `bSG_gP = bSG_gP_pre_slice[(None, 0, 0, eh_idx, b_idx)]`
  is set at L1802. For varlen, the post-loop store must index by
  `seq_id` instead. The varlen in-loop path already stores mid-loop
  fstates, so the post-loop store is a harmless duplicate for the
  last sequence. Change the TMA store destination:
  ```python
  # L2020-2025: inside `if local_warp_idx == 0:`
  if cutlass.const_expr(self.has_init_states and self.has_varlen):
      seq_id = seq_idx[c_idx * L + c_off, 0]
      bSG_gP_final = bSG_gP_pre_slice[(None, 0, 0, eh_idx, seq_id)]
  else:
      bSG_gP_final = bSG_gP
  cute.copy(tma_atom_p,
            bSG_sP[(None, inter2_p_producer_state.index)],
            bSG_gP_final)
  ```

- [ ] **d.** `ssd_combined.py` fstate allocation: use `num_seqs`.
  The kernel indexes fstate by `seq_id`, which can be 0..num_seqs-1.
  Currently fstate is allocated with `batch` as dim 0 (L317).
  For varlen, `batch=1` but `num_seqs` may be larger → OOB TMA store.
  ```python
  # L316-322: change batch → fstate_batch
  fstate_batch = init_states.shape[0] if init_states is not None else batch
  fstate_tensor, fstate_cutlass = _create_cutlass_tensor(
      [fstate_batch, nheads, headdim, dstate],
      ...
  )
  ```

- [ ] **e.** Remove `@pytest.mark.xfail` from `test_variable_seqlen`
  (`tests/mamba/test_chunk_scan_combined.py` L825).

#### Step 5.2: init_states reload for non-chunk-aligned sequence transitions

**Scope**: non-chunk-aligned sequences (step 4.2 prerequisite). A single
physical chunk may contain data from two sequences, split into two
logical chunks. Each logical chunk has its own `c_idx`/`c_off`, so
the `seq_idx[c_idx * L + c_off, 0]` derivation from step 5.1 applies
without change — the TMA warp and consumer already detect transitions.

The only additional concern is interaction with step 4.2's
`chunk_size_limit` masking: when the consumer reloads init_states
mid-physical-chunk, the INTER2_P re-emit and `tState` reset must
happen **before** the truncated chunk's inter-MMA, not after.

This step is blocked on step 4.2 and tested by
`TestChunkScanCombinedVarlenNonAligned`.

#### Summary: file changes

| File | Change |
|------|--------|
| `ssd_kernel.py` `__call__` | Add chunk_indices, chunk_offsets, has_varlen args |
| `ssd_kernel.py` `kernel` | Add chunk_indices, chunk_offsets params; modify chunk loops in TMA warps (init_states load moved into loop, seq_id derived from seq_idx), pre-inter (reload on seq transition), pre-intra, epilog |
| `ssd_combined.py` `_SSDKernel.__init__` | Add has_varlen flag |
| `ssd_combined.py` `_SSDKernel.run` | Accept and pass varlen tensors |
| `ssd_combined.py` `ssd_combined_fwd` | Accept seq_idx, chunk_indices/offsets; remove NotImplementedError guard; pass to sub-kernels |
| `tests/mamba/test_chunk_scan_combined.py` | Remove xfail from chunk-aligned varlen test (step 5.1) |

### Test plan

`TestChunkScanCombinedVarlen` (already in test file) verifies correctness
by comparing:
1. Packed varlen path (single batch=1 with seq_idx/chunk_indices/offsets)
   through the full combined reference
2. Independent per-sequence computation (each seq run separately, results
   concatenated)

Current test cases (chunk-aligned, all passing):
- `[1, 1]` — two sequences, one chunk each
- `[2, 2]` — two sequences, two chunks each
- `[1, 2, 1]` — three sequences, mixed lengths

These compare the Triton reference varlen path against independent
per-sequence computation. The CuTe kernel is not yet tested with
varlen (needs step 5 for init_states reload).

Future test cases (non-chunk-aligned, for step 4.2):
- e.g., seq lengths 80 + 176 packed into 2 physical chunks of 128

---

## 2. Gate Tensor (`z`)

**Status**: Not started

### Description

`z` has shape `(batch, seqlen, nheads, headdim)` — same as `x`. Applies a
gated activation at the end of the chunk scan:

```
output = output * z * sigmoid(z)    # SiLU-style gating
```

### Impact

Applied as post-processing after all MMA and state-passing computations.
Requires:
- Loading an additional tensor the same size as `x` (new TMA pipeline)
- Element-wise multiply in the epilog, before writing `y`
- Same layout/tiling concerns as `x`

Does not change the core algorithm but adds memory bandwidth pressure and a
new pipeline stage.

---

## 3. Pre-allocated Output (`out`)

**Status**: Not started

### Description

Accept a user-provided output tensor instead of allocating internally.

### Impact

Python-level change in `ssd_combined.py`. The kernel writes into the provided
tensor instead of a freshly allocated one. Only subtlety is ensuring the
tensor has the right layout/strides, or copying into it afterward.

---

## 4. Return Flags (`return_final_states`, `return_varlen_states`)

**Status**: Not started

### Description

Python-only control flow flags:
- `return_final_states`: whether to include `final_states` in the return
  tuple (the kernel already computes them).
- `return_varlen_states`: whether to compute and return per-sequence final
  states via `chunk_state_varlen` (requires `cu_seqlens`).

### Impact

Trivial Python conditionals on the return value. `return_varlen_states`
becomes non-trivial only after `cu_seqlens` support (feature #1) is
implemented.

---

## 5. State Data Type (`state_dtype`)

**Status**: Not started

### Description

Controls the dtype used for intermediate SSM states (defaults to `C.dtype`).
In the reference, only used in `_state_passing_fwd`:
```python
out_dtype = state_dtype if state_dtype is not None else C.dtype
```

### Impact

Dtype cast on intermediate buffers. The kernel's accumulator dtype is already
configurable via `acc_dtype`. Likely just a Python-level dtype argument
mapping.
