# Cake four-stream mHC training backend, deterministic (SM100 / SM103 / SM107)

This package provides the forward and backward kernels of a four-stream mHC (manifold-constrained hyper-connection) for training. Every output is deterministic. The connection collapses four BF16 residual streams into one branch input `U`. It then mixes the streams with a doubly stochastic 4x4 matrix and adds the branch output `B` back with per-stream weights. The branch is an attention or MLP block, and it is not part of this package. Per token, the arithmetic is FP32 and the activations are BF16 at the boundaries:

```text
u    = float32(X).reshape(T, 4H)
r    = rsqrt(mean(u * u) + 1e-5)                          # mean over the 4H values of the token
z    = (u * r) @ fn.T                                     # [T, 24]
pre  = sigmoid(z[:, :4] * scale[0] + base[:4]) + 1e-6
post = 2 * sigmoid(z[:, 4:8] * scale[1] + base[4:8])
C    = row_softmax((z[:, 8:] * scale[2] + base[8:]).reshape(T, 4, 4)) + 1e-6
C    = C / (column_sum(C) + 1e-6)
repeat 19 times: row-normalize C; column-normalize C      # divide by (sum + 1e-6); 20 Sinkhorn iterations in total
U[t, d]    = BF16(sum_i pre[t, i] * X[t, i, d])
B          = external_branch(U)                           # attention / MLP, BF16 [T, H]
Y[t, j, d] = BF16(post[t, j] * B[t, d] + sum_i C[t, i, j] * X[t, i, d])
```

The defaults are `rms_eps = 1e-5`, `hc_eps = 1e-6` and 20 Sinkhorn iterations. You can change them with keyword arguments on the entry points. The epsilons are launch arguments and never change the program. A Sinkhorn count other than 20 (1 to 64) runs the run-time-count programs, which are slower.

## Names and mode

- Every exported function and class name contains the mode word `det`; the constants `ARCHES` and `SUPPORTED_COMPUTE_CAPABILITIES` are the exceptions. The entry points take no `deterministic` keyword.
- Under `torch.use_deterministic_algorithms(True)` the entry points give the same results and raise no alert.
- The former names from FlashInfer 0.7.3 (`flashinfer.cake_mhc_train`) have no alias.
- The non-deterministic counterpart is `flashinfer.cake_mhc_nondet_train`. The head, which takes the stream mean of the last layer, is `flashinfer.cake_mhc_head_det_train`.
- Importing `flashinfer.cake_mhc_det_train` does not import a backend. The backend package loads on the first call or on the first access of a lazy name such as `ARCHES`.

## Public entry points (`flashinfer/cake_mhc_det_train.py`)

```python
from flashinfer.cake_mhc_det_train import (
    cake_mhc_det_train_prepare,
    cake_mhc_det_train_finish,
    cake_mhc_det_train_forward_prepare,
    cake_mhc_det_train_forward_finish,
    cake_mhc_det_train_backward_prepare,
    cake_mhc_det_train_backward_finish,
    is_cake_mhc_det_train_supported,
)

# X: BF16 [T, 4, H]; fn: FP32 [24, 4H]; scale: FP32 [3]; base: FP32 [24]
assert is_cake_mhc_det_train_supported(X.device, hidden=X.shape[2])

# Autograd around the external branch.
U, handle = cake_mhc_det_train_prepare(X, fn, scale, base)  # pass save_stats=False to replay in backward
Y = cake_mhc_det_train_finish(X, branch(U), handle)
Y.backward(dY)

# Low level: the caller runs the branch forward and backward.
U, stats = cake_mhc_det_train_forward_prepare(X, fn, scale, base)
B = branch(U)
Y = cake_mhc_det_train_forward_finish(X, B, stats)
dB, grad_stats = cake_mhc_det_train_backward_prepare(dY, B, X, stats)
dU = branch_backward(dB)
dX, dfn, dscale, dbase = cake_mhc_det_train_backward_finish(
    X, dY, dU, B, fn, scale, base, stats, grad_stats)
```

| Name | Purpose |
|---|---|
| `cake_mhc_det_train_prepare` / `_finish` | Autograd entry points around the branch. With `save_stats=False`, the backward recomputes the statistics (checkpoint replay). |
| `CakeMHCDetPrepareFunction` / `CakeMHCDetFinishFunction` | The autograd functions behind those two entry points. |
| `cake_mhc_det_train_forward_prepare` / `_forward_finish` | Low-level forward. |
| `cake_mhc_det_train_backward_prepare` / `_backward_finish` | Low-level backward. It can accumulate `dfn`, `dscale` and `dbase` into FP32 buffers that the caller provides. |
| `cake_mhc_det_train_forward_stats(...)` | Recomputes `stats` from `X` for checkpoint replay. |
| `cake_mhc_det_train_{forward,backward}_workspace_bytes`, `_{forward,backward}_workspace`, `_reset_workspace` | Workspace helpers. See [Workspaces](#workspaces). |
| `is_cake_mhc_det_train_supported(device, hidden=None)` | Reports whether the device and hidden size are supported. |
| `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES` | Supported architectures. |

### Contract

All operands must be contiguous.

| Operand | Dtype | Shape | Role |
|---|---|---|---|
| `X` | BF16 | `[T, 4, H]` | Input streams |
| `fn` | FP32 | `[24, 4H]` | Mapping weights |
| `scale` | FP32 | `[3]` | Scales for `pre`, `post` and `C` logits |
| `base` | FP32 | `[24]` | Biases for `pre`, `post` and `C` logits |
| `U` | BF16 | `[T, H]` | Branch input |
| `B` | BF16 | `[T, H]` | Branch output, produced outside the package |
| `Y` | BF16 | `[T, 4, H]` | Output streams |
| `stats` | FP32 | 256 bytes per token | `r`, `z`, `pre`, `post`, `C` |
| `dY`, `dX` | BF16 | `[T, 4, H]` | Stream gradients |
| `dU`, `dB` | BF16 | `[T, H]` | Branch gradients |
| `grad_stats` | | | From `backward_prepare`, consumed by `backward_finish` |
| `dfn`, `dscale`, `dbase` | FP32 | as `fn`, `scale`, `base` | Parameter gradients |

- `T` is a run-time scalar and may change between calls. `T == 1` is supported. `T == 0` returns empty outputs without a launch.
- The hidden size `H` comes from `X.shape[2]` and must be 4096, 5120 or 7168. One process can mix calls with different hidden sizes.

| Architecture | Devices | Target |
|---|---|---|
| SM100 | B200 / GB200 | `sm_100a` |
| SM103 | B300 / GB300 | `sm_103a` |
| SM107 | Rubin R200 | `sm_107a` |

## Determinism

Each output below repeats bit for bit when two calls meet all three conditions:

- identical inputs
- the same device, so the same SM count
- the same configuration: hidden size, token count and keyword arguments

| Call | Output | Bitwise repeatable | Notes |
|---|---|---|---|
| `forward_prepare` | `U` | yes | |
| `forward_prepare` | `stats` | yes | |
| `forward_finish` | `Y` | yes | |
| `backward_prepare` | `dB` | yes | |
| `backward_prepare` | `grad_stats` | yes | |
| `backward_finish` | `dX` | yes | |
| `backward_finish` | `dfn`, `dscale`, `dbase` | yes | Also when accumulating into caller buffers |

- **Concurrent work.** Work on other streams does not change the bits. Runs with a side load gave the same bits as runs without one.
- **Checkpoint replay.** `save_stats=False`, or `torch.utils.checkpoint` around the connection, gives gradients that match the backward from saved statistics bit for bit.
- **Mechanism.** `dfn`, `dscale` and `dbase` are reduced from per-block partials in a fixed order. Atomic operations appear only on integer work tickets; no value is accumulated atomically. Some kernels claim work units dynamically, but each unit's result and its place in every reduction do not depend on which CTA ran it.
- **Not guaranteed.** Bits may differ between devices with different SM counts and between package versions, because the work split and reduction order may change. Accuracy is held to the test error budgets (see [Numerics](#numerics)), not to equality with earlier versions.
- **Evidence.** The measurement used SM100, SM103 and SM107 boards. Each configuration ran 5 times, and every output was compared by SHA-256. It covered every supported hidden size and several token counts, each under three conditions: an idle device, a 16-CTA side load on another stream, and a side load that held every SM.

## Checkpointing and memory

- Between forward and backward, the package keeps only the inputs and `stats` (256 bytes per token), held as autograd saved tensors. `torch.utils.checkpoint` therefore frees them after the forward and recomputes them in the backward.
- Nothing stays alive after a non-reentrant checkpointed forward or after the backward of a connection. The finish node passes `dY` and the gradient statistics to the prepare node on autograd edges.
- At `H = 4096`, up to 4096 tokens on SM100 and SM107 and up to 2048 tokens on SM103, the replay also writes the transposed split of `fn` (3 MiB) on one more edge. This saves the backward from splitting `fn` again.

## Workspaces

- You can hold workspaces yourself or use the default ones. If you pass workspaces and preallocated outputs, the submit path allocates nothing.
- A workspace serves calls with the hidden size it was sized for, up to `T` tokens. It also serves every smaller `T`, including CUDA graph replays at a smaller `T`.
- Ticket words must start at zero. Allocate with `torch.zeros` or the helpers. The kernels leave the ticket words at zero.
- The default workspaces are reused across token counts: one buffer per device, kind, hidden size and power-of-two token bucket. The caller does not need to reset them.
- CUDA graph capture of the calls is supported. Use it when small `T` makes the calls host-bound.

| Helper | Purpose |
|---|---|
| `cake_mhc_det_train_forward_workspace_bytes(T, device, hidden=4096)` | Forward workspace size |
| `cake_mhc_det_train_backward_workspace_bytes(...)` | Backward workspace size |
| `cake_mhc_det_train_forward_workspace(T, device, hidden=...)` | Allocate a zeroed forward workspace |
| `cake_mhc_det_train_backward_workspace(...)` | Allocate a zeroed backward workspace |
| `cake_mhc_det_train_reset_workspace(ws)` | Zero a workspace after an interrupted launch (an asynchronous error or a cancelled stream capture) |

## Numerics

- The mapping GEMM has FP32 quality. `fn` is split exactly into three BF16 pieces, and the products accumulate in FP32. TF32 is not used.
- `dX` sums the mapping, collapse and residual-mix contributions in FP32 and rounds to BF16 once.
- `dB` is the BF16 rounding of the complete FP32 sum over the streams, with one rounding.
- The coefficients, their gradients and the parameter reductions are all FP32.
- The sigmoid tails keep their gradients where the FP32 forward sigmoid rounds to 1.
- The tests in `tests/mhc/test_cake_mhc_det_train*.py` validate `Y`, `U`, `dX`, `dB`, `dfn`, `dscale` and `dbase` against an FP64 reference. They check activation errors and parameter-gradient errors separately. Coverage includes cancellation, saturated sigmoid tails, empty, small, large and irregular `T`, checkpoint replay, and repeated gradient accumulation.

## Layout of this package

| Path | Contents |
|---|---|
| `__init__.py` | Entry points, autograd functions, host launch plans |
| `cake_jit.py` | Host rules: module registry, routes per architecture and hidden size, token-count bands. The host plan is hand-maintained; the exporter rewrites the derived record fields. |
| `cake_build.py` | Library loader. Serves `aot/<name>.so` when a builder compiled it from the manifest; this prebuilt path never imports `flashinfer`. Otherwise JIT-builds through FlashInfer from the same sources and flags. Written by the exporter; do not edit. |
| `build.json` | Manifest, schema `cake-build/1`. One entry per library and architecture (`<library>_<arch>`) with package-relative sources, nvcc flags (`-gencode` for `sm_100a` / `sm_103a` / `sm_107a`), toolkit-conditional flags and SHA-256 digests. Requires CUDA toolkit >= 12.8 for SM100, >= 12.9 for SM103 and >= 13.4 for SM107, and `apache-tvm-ffi >= 0.1.11, < 0.2`. |
| `csrc/` | Generated CUDA: `<library>_kernel.cu` and `<library>_binding.cu` per library, plus the package-local `cake_ffi_checks.h`. Not hand-edited. |
| `aot/` | Optional prebuilt libraries and the `build_info.json` sidecar. Not tracked. |

FlashInfer's `scripts/check_cake_build_manifests.py` builds and checks the manifest.

## Status

The API and backend are experimental and carry no compatibility guarantee. The backend does not take part in automatic backend selection, autotuning or trace-apply.
