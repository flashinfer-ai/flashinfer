# Handoff: FMHA v2 bidirectional sliding window (H100)

Branch: `feat/fmha-v2-bidirectional-sliding-window`

This ports TensorRT-LLM FMHA v2 **bidirectional sliding window attention**
([NVIDIA/TensorRT-LLM#11212](https://github.com/NVIDIA/TensorRT-LLM/pull/11212))
into FlashInfer. Kernel + JIT + Python API + tests are in the branch. **They
have not been run on GPU.** This note is for an agent on a real **H100
(SM90 / Hopper)** to validate.

H100 exercises the **warp-specialized** kernels
(`csrc/fmha_v2/templates/kernel_hopper_ws.jinja` +
`csrc/fmha_v2/fmha/warpspec/{compute,dma,epilogue}.h`). It does **not**
exercise the SM120 tiled noloop path; do not treat a green H100 run as
Blackwell coverage.

---

## What changed

New `trtllm_fmha_v2_prefill` mask mode:

```python
trtllm_fmha_v2_prefill(
    ...,
    mask_mode="bidirectional_sliding_window",
    window_left=64,  # required, >= 0
)
```

### Semantics

| `mask_mode` | Query `i` attends |
|---|---|
| `causal` | `[0, i]` |
| `sliding_window` | `[i - window_left, i]` |
| `padding` | full sequence (encoder dense) |
| **`bidirectional_sliding_window`** | **`[i - window_left, i + window_left]`** (clamped to sequence) |

`window_left` is tokens **on each side of the query, excluding the query
itself**. `window_left=0` attends only the query token. `window_left=64`
attends 64 tokens left + self + 64 tokens right.

Host mapping in `csrc/fmha_v2_run.cu`:

- Causal SWA: `sliding_window_size = window_left + 1` (unchanged).
- Bidirectional: `sliding_window_size = 2 * window_left` because the kernel
  uses `± sliding_window_size / 2`.

The launcher must **not** rewrite bidirectional to
`SLIDING_OR_CHUNKED_CAUSAL` when `window_left > 0` (that was the old
behavior for all windowed requests).

Rejected combinations:

- `bidirectional_sliding_window` + `window_left < 0`
- `bidirectional_sliding_window` + `chunked_attention_size > 0`
- `padding` + `window_left >= 0` (still requires causal)

### Enum numbering (must stay consistent)

| Layer | padding | causal | sliding causal | **bidirectional** | custom |
|---|---|---|---|---|---|
| Host `Attention_mask_type` | 0 | 1 | 2 | **3** | **4** (was 3) |
| Ampere/Hopper `MASK_VERSION` | 2 | 3 | 4 | **5** | **6** (was 5) |
| Warpspec `ATTENTION_MASK_TYPE_` | 0 | 1 | 2 | **3** | **4** (was 3) |

Warpspec: `CAUSAL_MASK` is types 1 or 2 only (not 3).
`SLIDING_OR_CHUNKED_ATTENTION` is types 2 **or** 3. Bidirectional is **not**
causal; softmax still applies a right bound `row + W/2`.

Custom mask was shifted so `Mask<..., 5>` is bidirectional, not custom. If
tests that use custom masks exist, they must still work.

### Files (high level)

- Kernel: `csrc/fmha_v2/fmha/{mask,kernel_traits,hopper/kernel_traits}.h`,
  `csrc/fmha_v2/fmha/warpspec/{kernel_traits,compute,dma,epilogue}.h`,
  noloop kernels.
- JIT templates: `fa_kernel.jinja`, `kernel_hopper.jinja`,
  `kernel_hopper_ws.jinja`.
- Codegen: `flashinfer/jit/attention/fmha_v2/{utils,fmha_library,generator_utils}.py`.
- Host/API: `csrc/fmha_v2_run.cu`, `flashinfer/prefill.py`.
- Tests: `tests/attention/test_fmha_v2_prefill.py`.

Torch reference (`attention_ref_torch`): when `causal=False` and
`window_left >= 0`, apply both
`kv >= q - window_left` **and** `kv <= q + window_left`. Existing causal
SWA tests still pass `causal=True`, so they only get the left bound.

---

## Setup on H100

The branch lives on **Rohan's fork** (no write access to `flashinfer-ai/flashinfer`):

- Fork: https://github.com/nvrohanv/flashinfer
- Branch: `feat/fmha-v2-bidirectional-sliding-window`
- Commit: `5abb561`

```bash
# If this is already a flashinfer-ai checkout:
git remote add nvrohanv https://github.com/nvrohanv/flashinfer.git  # skip if present
git fetch nvrohanv feat/fmha-v2-bidirectional-sliding-window
git checkout -B feat/fmha-v2-bidirectional-sliding-window nvrohanv/feat/fmha-v2-bidirectional-sliding-window

# Or clone the fork directly:
# git clone -b feat/fmha-v2-bidirectional-sliding-window --recursive https://github.com/nvrohanv/flashinfer.git
# cd flashinfer

# Editable install if this machine is not already set up
pip install --no-build-isolation -e . -v

# Pick up kernel/template changes (mandatory if this box had an older fmha_v2 JIT cache)
rm -rf ~/.cache/flashinfer/

# Optional while debugging compile/dispatch
export FLASHINFER_JIT_VERBOSE=1
export FLASHINFER_FMHA_V2_VERBOSE=1
```

Confirm GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expect something like `NVIDIA H100 ... (9, 0)`.

---

## What to run

First call JIT-compiles; budget several minutes and a lot of RAM. `MAX_JOBS=4`
and `FLASHINFER_NVCC_THREADS=4` if the box is memory-tight.

### 1. New bidirectional tests (must pass)

```bash
pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_bidirectional_sliding_window -q --tb=short
pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_bidirectional_sliding_window_rejected -q --tb=short
```

The first is a cartesian product: layouts
`PACKED_QKV` / `CONTIGUOUS_Q_KV` / `Q_PAGED_KV_NHD`, dtypes fp16/bf16,
`head_dim` 64/128, MHA+GQA, `window_left` in `{0, 64, 127}`, seq up to 512.
It compares against the torch reference at rtol/atol `1e-2`.

The second is host-side rejection only (no kernel numeric check):

- `window_left=-1` → `ValueError` matching `window_left`
- `chunked_attention_size=16` → `ValueError` matching `chunked`
- `mask_mode="padding"` + `window_left=8` → `ValueError` matching `causal masking`

Smoke one case if the full product is too slow:

```bash
pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_bidirectional_sliding_window \
  -q --tb=short \
  -k "PACKED_QKV and float16 and head_dim64 and window_left0"
```

### 2. Regression: existing FMHAv2 prefill (must still pass)

These used to work on H100 and must not break:

```bash
pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill -q --tb=short
```

That covers `CAUSAL` and `SLIDING_WINDOW` (causal left window) across layouts,
including FP8. If wall-clock is an issue, at least:

```bash
pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill \
  -q --tb=short \
  -k "PACKED_QKV and causal"
```

Do **not** run `test_trtllm_fmha_v2_prefill_sm120_large_head_dim` on H100; it
skips unless SM12x.

### 3. Optional numeric spot-check

If pytest is green but you want to eyeball that future tokens are **not**
masked (the bug if bidirectional were still treated as causal):

```python
import torch, math
import flashinfer
from flashinfer.prefill import trtllm_fmha_v2_prefill

torch.manual_seed(0)
device = "cuda"
s, h, d = 32, 4, 64
window_left = 2
qkv = torch.randn(s, 3, h, d, device=device, dtype=torch.float16)
seq = torch.tensor([s], device=device, dtype=torch.int32)
cu = torch.tensor([0, s], device=device, dtype=torch.int32)
ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
scale = 1.0 / math.sqrt(d)

out = trtllm_fmha_v2_prefill(
    qkv, "PACKED_QKV", ws, seq, s, s, scale, 1.0, 1, cu, cu,
    mask_mode="bidirectional_sliding_window",
    window_left=window_left,
)

# Reference: query 0 attends keys 0..2; query 10 attends keys 8..12
q, k, v = qkv[:, 0].float(), qkv[:, 1].float(), qkv[:, 2].float()
logits = torch.einsum("qhd,khd->hqk", q, k) * scale
idx_q = torch.arange(s, device=device)[:, None]
idx_k = torch.arange(s, device=device)[None, :]
keep = (idx_k >= idx_q - window_left) & (idx_k <= idx_q + window_left)
logits = logits.masked_fill(~keep, float("-inf"))
ref = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), v)
torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
print("spot-check ok", out.shape)
```

If this fails with query-0 matching a **causal** window (`keys 0..0` only)
instead of `0..window_left`, the mask type was overridden to causal SWA.

---

## Pass / fail

**Pass**

- Bidirectional tests match torch within `1e-2`.
- Rejection test raises the three `ValueError`s above.
- Existing causal / sliding-window prefill tests still pass.
- JIT compiles without error (first run only).

**Fail — look here first**

| Symptom | Likely cause |
|---|---|
| Query attends only left of diagonal | Host overwrote mask to `SLIDING_OR_CHUNKED_CAUSAL`, or warpspec `CAUSAL_MASK` includes type 3 |
| Custom-mask / unrelated FMHAv2 tests break | Custom enum not shifted to 4 / `MASK_VERSION` 6 |
| `Unsupported FMHAv2 attention mask type` | Jinja/warpspec kernel not generated or launch dispatch missing `BIDIRECTIONAL_SLIDING_WINDOW` |
| Compile error on `Mask<...,5>` / traits | Numbering mismatch between host enum and `MASK_VERSION` |
| `window_left` mapping off by ~2x | Forgot `sliding_window_size = 2 * window_left` |
| JIT uses stale `.so` | Did not wipe `~/.cache/flashinfer/` |

Verbose JIT / dispatcher:

```bash
FLASHINFER_JIT_VERBOSE=1 FLASHINFER_FMHA_V2_VERBOSE=1 \
  pytest tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_bidirectional_sliding_window \
  -q --tb=short -k "PACKED_QKV and float16 and head_dim64 and window_left0"
```

---

## Out of scope on H100

- SM120 / Blackwell tiled kernels (`fa_kernel.jinja` noloop tiled,
  `test_trtllm_fmha_v2_prefill_sm120_*`). Needs an SM12x box.
- DeepSeek MLA SM120 specialized module (`gen_trtllm_fmha_v2_sm120_module` /
  `generator_utils.py`). Codegen was updated so custom is `MASK_VERSION` 6;
  that path was not GPU-tested.
- Wiring bidirectional into `BatchPrefillWithPagedKVCacheWrapper` / high-level
  `causal=` APIs. Only `trtllm_fmha_v2_prefill(..., mask_mode=...)` is exposed.

---

## If tests pass

Reply with:

1. GPU name + compute capability.
2. Pytest commands run and results (pass/fail counts).
3. Any skips, JIT compile errors, or numerical mismatches (shape + first
   failing assert).
4. Whether `~/.cache/flashinfer/` was cleared.

Do not commit or push from the H100 box unless asked.
