# SSM Cache Checkpointing: Fast-Forward for FlashInfer STP Kernel

## Summary

**Goal**: Modify the FlashInfer STP (single-token) `selective_state_update` kernel
so it can accept a buffer of K past tokens and fast-forward the state before
processing the current token — without writing state back to cache.

**Why**: Reduces SSM state cache IO by ~50% and makes FP8 quantized cache lossless
(quantization noise only accumulates at checkpoint boundaries, not every step).

**What we did so far**:
- Deep-dived the FlashInfer kernel code (STP, MTP, dispatch)
- Traced the full call chain from vLLM → FlashInfer
- Identified that vLLM [PR #36162](https://github.com/vllm-project/vllm/pull/36162)
  adds `--mamba-backend flashinfer` to vLLM (not yet merged)
- Found that the MTP kernel already has the exact token-loop we need (can use as
  reference), but we want to add the fast-forward loop to the STP kernel directly
  so it benefits from the SM90+ TMA optimizations
- Created correctness test (`tests/mamba/test_ssm_cache_checkpointing.py`)

---

## Background

### Vanilla decode (today)

Every decode step, the kernel:
1. **Reads** SSM state from HBM (96 MiB for Ultra)
2. **Computes** `h_new = h * exp(A*dt) + B*dt*x`, `y = dot(h_new, C) + D*x`
3. **Writes** updated state back to HBM (96 MiB)

Total IO per step: **192 MiB** (read + write).

### Checkpointing

Freeze state for L steps. Buffer the per-token inputs. Each step, fast-forward from
frozen state through the buffer to get the current output. Write state only every L steps.

```
counter = 0
for each decode step:
    if counter < L:
        y = fast_forward(frozen_state, buffer[0..counter], current_token)
        buffer[counter] = current_token_inputs
        counter += 1
        # NO state write (save 96 MiB)
    else:
        y = fast_forward(frozen_state, buffer[0..L-1], current_token)
        write state back  # checkpoint
        clear buffer, counter = 0
```

IO per step drops from 192 MiB to ~96 MiB + buffer_size (~365 KiB per sequence).
For Nemotron Ultra with L=10, that's nearly 50% IO reduction.

### Numbers for Nemotron Ultra

```
layers=48, heads=256, head_dim=64, n_groups=8, dstate=128

SSM state:    48 × 256 × 64 × 128 = 96 MiB (BF16) per sequence
xAB per token: 48 × (256×64 + 256 + 8×128) = 828 KiB (BF16) per sequence
Ratio: ~118×

Sweet spot: L ≈ 4–10 (IO savings dominate recompute overhead)
```

---

## Kernel Changes Required

### Overview

The STP kernel today processes 1 token. We need to add a fast-forward loop that
replays K buffered tokens (updating state in registers) before processing the
current token. The buffered tokens only need the state update — no output
computation (no C dot product), since we only need y for the current token.

The MTP kernel (`kernel_selective_state_update_mtp.cuh` lines 297–326) already
has the exact token loop we need as a reference:

```cuda
// MTP inner loop (reference) — processes multiple tokens sequentially
for (int step = 0; step < TOKENS_MTP; step++) {
    float x_value = toFloat(sram.x[step][d]);
    auto dt_value = rdt[step];
    auto const dA = __expf(A_value * dt_value);

    for (int ii = 0; ii < stateValuesPerThread; ii += packed_input_t::count) {
        rB = load(&sram.B[step][base_i]);
        for (int k = 0; k < packed_input_t::count; k++) {
            auto const dB = B_value * dt_value;
            state_value = state_value * dA + dB * x_value;   // state recurrence
            out_value += new_state * C_value;                 // ← we SKIP this for buffered tokens
        }
    }
}
```

For checkpointing, we use the same recurrence but **skip the output projection**
(`out_value += new_state * C_value`) for the buffered tokens. We only compute
output for the current (last) token.

### File-by-file changes

#### 1. `include/flashinfer/mamba/selective_state_update.cuh` — Params struct

Add xAB buffer pointers and length to `SelectiveStateUpdateParams`:

```cuda
struct SelectiveStateUpdateParams {
    // ... existing fields ...

    // Checkpointing: xAB buffer of past tokens to fast-forward through
    void* __restrict__ xab_x{nullptr};        // (batch, K, nheads, dim)
    void* __restrict__ xab_dt{nullptr};       // (batch, K, nheads)
    void* __restrict__ xab_B{nullptr};        // (batch, K, ngroups, dstate)
    uint32_t xab_length{0};                   // K = number of buffered tokens (0 = vanilla)

    int64_t xab_x_stride_batch{};             // stride to next batch in xab_x
    int64_t xab_x_stride_token{};             // stride to next token in xab_x
    int64_t xab_dt_stride_batch{};
    int64_t xab_dt_stride_token{};
    int64_t xab_B_stride_batch{};
    int64_t xab_B_stride_token{};
};
```

Note: we do NOT buffer C. C is only used for the output dot product, and we
only compute output for the current token (which gets C through the existing path).

#### 2. `include/flashinfer/mamba/kernel_selective_state_update_stp.cuh` — Simple kernel

In `selective_state_update_kernel_simple`, add a fast-forward loop between
state load and current-token processing.

Today:
```cuda
// Load state → compute new_state for current token → compute output → write state
for (int iter = 0, i = lane * load_state_t::count; i < DSTATE; ...) {
    auto rState = *load_state*;
    for (int ii = 0; ii < load_state_t::count; ii++) {
        auto state_value = toFloat(rState.val[ii]) * state_decode_scale;
        auto const new_state = state_value * dA + dB * x_value;      // current token
        out_value += new_state * C_value;                              // output
    }
    *write_state*;
}
```

After modification:
```cuda
for (int iter = 0, i = lane * load_state_t::count; i < DSTATE; ...) {
    auto rState = *load_state*;
    for (int ii = 0; ii < load_state_t::count; ii++) {
        auto state_value = toFloat(rState.val[ii]) * state_decode_scale;

        // === NEW: fast-forward through xAB buffer ===
        for (int t = 0; t < params.xab_length; t++) {
            float xab_x_val = load xab_x[batch, t, head, d];
            float xab_dt_val = load xab_dt[batch, t, head];
            float xab_B_val = load xab_B[batch, t, group, i + ii];
            float xab_dA = __expf(A_value * xab_dt_val);
            if (dt_bias) xab_dt_val += dt_bias_value;
            if (dt_softplus) xab_dt_val = thresholded_softplus(xab_dt_val);
            float xab_dB = xab_B_val * xab_dt_val;
            state_value = state_value * xab_dA + xab_dB * xab_x_val;
            // NO output computation — only updating state
        }
        // === END NEW ===

        // Existing: process current token (with output)
        auto const new_state = state_value * dA + dB * x_value;
        out_value += new_state * C_value;
    }
    *write_state*;   // controlled by params.update_state (skip if checkpointing)
}
```

The same pattern applies to the vertical and horizontal TMA kernels
(`consumer_func_vertical`, `consumer_func_horizontal`).

#### 3. `csrc/selective_state_update.cu` — C++ dispatch

In `run_selective_state_update_stp`, accept the new buffer tensors and populate
the params struct:

```cpp
void run_selective_state_update_stp(
    // ... existing args ...
    Optional<TensorView> xab_x,       // NEW
    Optional<TensorView> xab_dt,      // NEW
    Optional<TensorView> xab_B,       // NEW
) {
    // ... existing validation ...

    if (xab_x.has_value()) {
        p.xab_x = const_cast<void*>(xab_x.value().data_ptr());
        p.xab_length = xab_x.value().size(1);  // T dimension
        p.xab_x_stride_batch = xab_x.value().stride(0);
        p.xab_x_stride_token = xab_x.value().stride(1);
        // ... same for xab_dt, xab_B ...
    }

    invokeSelectiveStateUpdate<...>(p, algo, stream);
}
```

#### 4. `flashinfer/mamba/selective_state_update.py` — Python API

Add new optional parameters:

```python
def selective_state_update(
    # ... existing params ...
    xab_x: Optional[torch.Tensor] = None,      # NEW: (batch, K, nheads, dim)
    xab_dt: Optional[torch.Tensor] = None,     # NEW: (batch, K, nheads) or broadcasted
    xab_B: Optional[torch.Tensor] = None,      # NEW: (batch, K, ngroups, dstate)
) -> torch.Tensor:
```

When `xab_x` is provided, the kernel fast-forwards through the K buffered tokens
before processing the current token (passed as `x`, `dt`, `B`, `C` as today).

---

## vLLM Integration (runtime side)

Requires [PR #36162](https://github.com/vllm-project/vllm/pull/36162) merged
(adds `--mamba-backend flashinfer` to vLLM).

### Call site change (`mamba_mixer2.py` line 874)

Today:
```python
selective_state_update(ssm_state, hidden_states_d, dt_d, A_d, B_d, C_d, D_d, ...)
```

With checkpointing:
```python
# 1. Save current token inputs to buffer
xab_x[slot_indices, counter] = hidden_states_d
xab_dt[slot_indices, counter] = dt_d
xab_B[slot_indices, counter] = B_d

# 2. Call kernel with buffer
K = counter + 1
selective_state_update(
    ssm_state, hidden_states_d, dt_d, A_d, B_d, C_d, D_d,
    # ... existing args ...
    disable_state_update=(counter < L - 1),   # skip writeback except at checkpoint
    xab_x=xab_x[slot_indices, :K],            # buffered x for past tokens
    xab_dt=xab_dt[slot_indices, :K],           # buffered dt
    xab_B=xab_B[slot_indices, :K],             # buffered B
)

# 3. Bookkeeping
if counter >= L - 1:
    counter = 0          # checkpoint done, reset
else:
    counter += 1
```

### New allocations

Alongside `ssm_state` in vLLM's cache:

```python
xab_x:  (num_layers, pool_size, L_max, nheads, head_dim)        # ~same dtype as model
xab_dt: (num_layers, pool_size, L_max, nheads)                   # fp32
xab_B:  (num_layers, pool_size, L_max, ngroups, dstate)          # ~same dtype as model
```

For Ultra with L_max=10, pool_size=256: ~94 MiB total (vs 24 GiB for SSM state cache).

### Config

```
--mamba-backend flashinfer
--mamba-checkpoint-interval 10    # L=10, default 1 = vanilla (no checkpointing)
```

---

## Definition of Done

- [x] `SelectiveStateUpdateParams` has xAB buffer fields
- [x] `selective_state_update_kernel_simple` has fast-forward loop
- [x] Same for vertical and horizontal TMA variants
- [x] C++ dispatch accepts and validates buffer tensors
- [x] Python API exposes `xab_x`, `xab_dt`, `xab_B` parameters
- [x] Correctness test: STP with buffer matches Triton multi-token reference (27/27 pass)
- [x] Correctness test: state unchanged when `disable_state_update=True`
- [ ] Correctness test: checkpoint writeback produces correct final state
- [x] `xab_length=0` (no buffer) is identical to today's behavior

---

## Key Files

| File | What changes |
|------|-------------|
| `include/flashinfer/mamba/selective_state_update.cuh` | Add buffer fields to params struct |
| `include/flashinfer/mamba/kernel_selective_state_update_stp.cuh` | Add fast-forward loop to simple, vertical, horizontal kernels |
| `csrc/selective_state_update.cu` | Accept buffer tensors in C++ dispatch |
| `flashinfer/mamba/selective_state_update.py` | Add `xab_x`, `xab_dt`, `xab_B` to Python API |
| `tests/mamba/test_ssm_cache_checkpointing.py` | Correctness tests |

---

## Environment Setup

```bash
# Activate the venv (required before running tests or pip install)
source /my_home/venvs/fi/bin/activate

# Install in editable mode (only needed once, or after git clean)
pip install --no-build-isolation -e . -v

# Clear JIT cache (when kernel code changes aren't picked up)
rm -rf ~/.cache/flashinfer/
```

---

## Session Log

### Session 1 (2026-04-05)

- [x] Created `CHECKPOINTING.md` task tracking file
- [x] Ran baseline STP tests — **30/30 passed** (simple, vertical, horizontal × 10 param combos, 59s)
  ```
  cd /my_home/flashinfer && /my_home/venvs/fi/bin/python -m pytest --tb=short tests/mamba/test_selective_state_update_stp.py::TestSelectiveStateUpdate::test_output_correctness -x -v 2>&1 | tail -80
  ```
- [x] Implemented all kernel changes (params struct, simple/vertical/horizontal fast-forward)
- [x] Updated C++ dispatch, FFI bindings, and Python API
- [x] Created checkpointing correctness tests

### Session 2 (2026-04-05) — Resumed, debugged, all tests green

- [x] Reinstalled package in editable mode (site-packages had stale copy, not symlinked)
- [x] Debugged test failure (max_diff=56.8): `_get_single_token_dt` hardcoded
  `stride(0)=nheads` but sliced dt tensors have larger batch stride. Fixed by
  preserving the actual stride: `dt_base_slice.stride(0)`.
- [x] **All 27 checkpointing tests pass** (7 param combos × 3 algorithms + xab_length=0 + disable_state_update)
- [x] **All 30 baseline STP tests pass** (no regression)
- Run commands:
  ```
  cd /my_home/flashinfer && /my_home/venvs/fi/bin/python -m pytest --tb=short tests/mamba/test_ssm_cache_checkpointing.py -v
  cd /my_home/flashinfer && /my_home/venvs/fi/bin/python -m pytest --tb=short tests/mamba/test_selective_state_update_stp.py::TestSelectiveStateUpdate::test_output_correctness -v
  ```

### Session 3 (2026-04-06) — Replaced hand-rolled reference with Triton

- [x] Replaced `_reference_fast_forward` (hand-written fp32 Python loop) with
  `selective_state_update_triton` called with all K+1 tokens in a single
  multi-token pass. The Triton kernel processes all T tokens in fp32 registers
  without intermediate bf16 round-trips — same semantics as our checkpointing
  kernel. This is a stronger test: it uses the established ground-truth reference
  instead of a hand-rolled simulation.
- [x] Removed unused `torch.nn.functional` import and `A_base`/`D_base`/`dt_bias_base`
  dict entries that were only needed by the old reference.
- [x] **All 27 checkpointing tests pass** (Triton reference)
- [x] **All 59 baseline STP tests pass** (no regression)
- Run commands:
  ```
  source /my_home/venvs/fi/bin/activate
  cd /my_home/flashinfer && rm -rf ~/.cache/flashinfer/
  python -m pytest tests/mamba/test_ssm_cache_checkpointing.py -v
  python -m pytest tests/mamba/test_selective_state_update_stp.py -v
  ```

### Session 4 (2026-04-17) — Gemini review: hoist base pointers out of token loop

- [x] Applied Gemini code review suggestion on `apply_xab_fast_forward`:
  hoisted the three base pointers (`xab_x_ptr`, `xab_dt_ptr`, `xab_B_ptr`) out
  of the `t` loop. The batch/head/dim/group/dstate offsets are loop-invariant —
  previously recomputed on every iteration. Inner loop now only multiplies by
  `t * stride_token` per tensor.
- Commit: `3d347a81` on `danielafrimi/flashinfer:checkpointing`
- No test run yet on cluster (kernel-only change, logic unchanged)

---

## References

- MTP kernel inner loop (`kernel_selective_state_update_mtp.cuh:297–326`):
  exact same state recurrence, use as implementation reference
- [PR #36162](https://github.com/vllm-project/vllm/pull/36162): FlashInfer SSU backend for vLLM
- [Snakes and Ladders](https://arxiv.org/abs/2502.15297): speculative decoding with Mamba
- [Mamba in Llama §4](https://arxiv.org/abs/2408.15237): SSM cache analysis
