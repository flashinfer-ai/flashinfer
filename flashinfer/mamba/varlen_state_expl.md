# Pre-inter warp: state-related logic

Only the state (`P`) and init_states (`IS`) logic is shown.
All B/delta/scaledB processing is omitted.

## Registers and buffers

- `tState` (f32): running accumulated state in registers
- `tRS_rP` (io_dtype): register tile used for smem↔reg transfers
- `smem_p`: **single physical shared memory buffer** with multiple layout views:
  - `smem_pt` / `tRS_sP`: view for reg↔smem transfers (inter2_p pipeline writes)
  - `smem_p_load` / `tS2R_sP`: view for TMA load of init_states (IS pipeline writes)
  - `smem_p_store` / `bSG_sP`: view for TMA store to gmem fstate
  - `smem_p`: view for mma_inter MMA2 B-operand reads

**All views alias the same physical smem.** This is why IS is not released
in the pre-loop — releasing it would let the TMA warp overwrite smem_p
while the P pipeline is still using it for mma_inter.

## Pipelines

- `IS` = init_states_pipeline: TMA warp produces into `tS2R_sP`, pre_inter consumes
- `P` = inter2_p_pipeline: pre_inter produces into `smem_p`, mma_inter consumes
- `tma_p` = tma_p_pipeline: pre_inter stores from `smem_p` to gmem fstate

## Current code (no varlen reload)

```
PRE-LOOP:                                               (ssd_kernel.py L1814-1860)
    # 1. Load init_state from IS pipeline into registers
    IS.consumer_wait()                                   L1816
    tS2R_sP[IS slot] → tRS_rP                           L1819  (smem IS buf → regs)
    tRS_rP → tState  (cast to f32)                       L1821  (regs → f32 state)

    # 2. Prefill inter2_p slot 0 with init_state
    P.producer_acquire(slot0)                             L1839
    tState → tRS_rP  (cast to io_dtype)                  L1848  (f32 → regs)
    tRS_rP → smem_p[slot0]                               L1850  (regs → P smem)
    fence                                                 L1853
    P.producer_commit(slot0)                              L1858  → mma_inter can now read slot0
    P.advance                                             L1860

LOOP (for chunk_idx in 0..C-1):                          L1862
    # ... B/delta/scaledB processing omitted ...

    # 3. Wait for mma_inter's result + acquire next P slot
    inter1_acc.consumer_wait()                            L1935
    P.producer_acquire(next_slot)                         L1936

    # 4. Load mma_inter's INTER1_ACC result from tmem
    tmem → tTR_rP                                        L1946

    # 5. Recurrence: new_state = exp(last_col) * old_state + inter1_acc
    tTR_rP = exp(last_col) * tState + tTR_rP             L1953  (FMA)

    # 6. Update registers
    tTR_rP → tRS_rP  (cast to io_dtype)                  L1963
    tState = tTR_rP  (keep f32 copy)                     L1967

    # 7. Write new state into P smem slot
    tRS_rP → smem_p[cur_slot]                            L1970
    fence                                                 L1975

    # 8. Release inter1_acc, commit P slot
    inter1_acc.consumer_release()                         L1980
    if not last_iter:
        P.producer_commit(cur_slot)                       L1983  → mma_inter can read
    # advance, peek ...
    if not last_iter:
        P.advance                                         L2006

POST-LOOP:                                               L2009
    # 9. TMA store last state to gmem
    barrier(pre_inter_sync)                               L2015
    if warp0:
        TMA store smem_p[last_slot] → fstate[b_idx]      L2022
        tma_p.commit, tma_p.acquire                       L2028-2029
    barrier(pre_inter_sync)                               L2032
    tma_p.producer_tail()                                 L2034

    # 10. Release IS buffer (held since pre-loop)
    IS.consumer_release()                                 L2039
    IS.advance                                            L2040
```

## New logic with varlen

```
PRE-LOOP:
    # 1. Load init_state from IS pipeline into registers
    IS.consumer_wait()                                   L1816
    tS2R_sP[IS slot] → tRS_rP                           L1819
    tRS_rP → tState  (cast to f32)                       L1821

    # 2. Prefill inter2_p slot 0 with init_state
    P.producer_acquire(slot0)                             L1839
    tState → tRS_rP  (cast to io_dtype)                  L1848
    tRS_rP → smem_p[slot0]                               L1850
    fence                                                 L1853
    P.producer_commit(slot0)                              L1858  → mma_inter can read slot0
    P.advance                                             L1860

LOOP (for chunk_idx in 0..C-1):                          L1862
    c_idx = chunk_indices[chunk_idx]                      L1866
    c_off = chunk_offsets[chunk_idx]                      L1867

    # ... B/delta/scaledB processing omitted ...

    # 3. Wait for mma_inter's result + acquire next P slot
    inter1_acc.consumer_wait()                            L1935
    P.producer_acquire(next_slot)                         L1936

    # 4. Load mma_inter's INTER1_ACC result from tmem
    tmem → tTR_rP                                        L1946

    # 5. Recurrence: new_state = exp(last_col) * old_state + inter1_acc
    tTR_rP = exp(last_col) * tState + tTR_rP             L1953

    # 6. Update registers
    tTR_rP → tRS_rP  (cast to io_dtype)                  L1963
    tState = tTR_rP  (keep f32 copy)                     L1967

    # 7. Write new state into P smem slot
    tRS_rP → smem_p[cur_slot]                            L1970
    fence                                                 L1975

    # 8. Release inter1_acc
    inter1_acc.consumer_release()                         L1980

    # ── NEW: varlen look-ahead (const_expr guarded) ──
    is_last_chunk = (chunk_idx == C - 1)
    next_is_new_seq = False
    if not is_last_chunk:
        next_is_new_seq = (chunk_indices[chunk_idx + 1] != c_idx)
    store_final_state = is_last_chunk or next_is_new_seq

    # A. Store final state of ending sequence to gmem
    if store_final_state:
        if warp0:
            seq_id = seq_idx[c_idx * L + c_off, 0]
            TMA store smem_p[cur_slot] → fstate[seq_id]
            tma_p.commit()
            tma_p.acquire()

    # B. Reload init_state for new sequence
    if next_is_new_seq:
        IS.consumer_release()           ← free old IS buffer
        IS.advance()
        IS.consumer_wait()              ← block until TMA warp loads new init_state

        tS2R_sP[IS slot] → tRS_rP      ← load new init_state from IS smem
        tRS_rP → tState                 ← update f32 state registers

        tState → tRS_rP                 ← cast back to io_dtype
        tRS_rP → smem_p[cur_slot]       ← overwrite same P slot with new init_state
        fence
    # ── END varlen look-ahead ──

    # 9. Commit P slot (unchanged)
    if not last_iter:
        P.producer_commit(cur_slot)                       L1983  → mma_inter can read
    # advance, peek ...
    if not last_iter:
        P.advance                                         L2006

POST-LOOP:                                               L2009
    # 10. TMA store last state to gmem
    #     For varlen: use seq_id instead of b_idx
    #     (harmless duplicate — in-loop already stored on is_last_chunk)
    barrier(pre_inter_sync)                               L2015
    if warp0:
        if varlen:
            seq_id = seq_idx[c_idx * L + c_off, 0]
            TMA store smem_p[last_slot] → fstate[seq_id]
        else:
            TMA store smem_p[last_slot] → fstate[b_idx]
        tma_p.commit, tma_p.acquire                       L2028-2029
    barrier(pre_inter_sync)                               L2032
    tma_p.producer_tail()                                 L2034

    # 11. Release IS buffer (held since pre-loop or last reload)
    IS.consumer_release()                                 L2039
    IS.advance                                            L2040
```

Key safety observations:
- `smem_p` is shared between IS, P, and TMA store — **all alias the same buffer**
- At the look-ahead point, the P slot is not yet committed → mma_inter cannot
  read smem_p, so we can safely overwrite it
- Step A (gmem store) reads smem_p via TMA before step B overwrites it
- Step B releases IS, letting TMA warp write new init_state into smem_p.
  This is safe because:
  (1) we already TMA-stored the old data to gmem in step A
  (2) mma_inter can't touch smem_p (P not committed yet)
  (3) we immediately wait on IS.consumer_wait() so the TMA write completes
      before we read smem_p again
- After step B overwrites smem_p with new init_state, the P.commit at step 9
  hands it to mma_inter — which sees init_state[new_seq]
- Pipeline slot counts are unchanged (we overwrite, not add)
- The post-loop store (step 10) is a harmless duplicate for varlen: the in-loop
  `store_final_state` on `is_last_chunk` already wrote the last sequence's fstate
