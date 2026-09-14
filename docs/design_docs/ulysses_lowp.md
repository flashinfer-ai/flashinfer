# Low-precision Ulysses on the global quantization grid

This implementation provides INT8 Q/K and FP8 E4M3 V payload operations.
Attention and collectives belong to the caller. All legal shards use
BOUNDARY_MERGE, including shards whose length is divisible by 128.
Fused amax/pack and aligned-only receiver implementations are deferred.

## Layout and capability

| Device | Layout | Q group | K group | V storage alignment |
|---|---|---:|---:|---:|
| SM89 / SM120 | UlyssesLowpSageLayout | 32 | 64 | 64 |
| SM90 | UlyssesLowpSageLayoutSM90 | 16 | 128 | 128 |

D=64 or 128; FP16/BF16 inputs; batch and head counts are parametric.
Heads must be divisible by the Ulysses world size. This delivery targets
P=2/4/8 on SM90 and SM120; retained SM89 compatibility is not an acceptance claim.

`capability(device=None)` returns a dictionary with `supported`,
`layout_class`, `device_capability`, and compiled Q/K/D constants.
It probes only the device's own module and returns `layout_class=None`
when unsupported. Importing the module does not build or probe kernels.
Each layout's `is_supported()` checks architecture, compiled grid and selected D.
Construct either layout with `head_dim=64` or `head_dim=128` (default).
A layout rejects tensors with a different D before launching kernels.
Module-level tensor operations infer D from their inputs; geometry and unpack
functions take an explicit `head_dim`. All ranks must agree on D.
`capability()["supported_head_dims"]` reports the compiled dimensions;
`compiled_head_dim` retains the legacy default value 128.

Both dimensions compile into each existing architecture module (JIT and AOT).
The shared SM89/SM120 module requires CUDA 12.8 or newer because its fat binary
contains both `sm_89` and `sm_120a`. AOT registers it once when either target
is requested; SM90 uses its separate `sm_90a` module (CUDA 12.3 or newer).
Dimension dispatch changes the statistics CTA size, grouped-kernel thread count
and unpack transpose tile width; quantization arithmetic and Q/K token groups
are shared. D=64 halves each Q/K/V data section relative to D=128, while Q/K
scale counts and sequence padding stay unchanged. No padding of D=64 to 128
is performed.
Module-level functions remain SM89/SM120-specific; use the SM90 layout on Hopper.

The two compiled modules share the host bindings in `csrc/ulysses_lowp.cu`.
`csrc/ulysses_lowp_sm90.cu` selects Q16/K128 before including those bindings;
SM89/SM120 use Q32/K64. Shape checks, launch geometry and exported ABI stay
in one implementation. Python K boundary min/max, centered boundary merging
and live-tail repair also share helpers parameterized by the K group.
Consumers should use `layout.scale_widths(U)` instead of copying its formula.

```python
import flashinfer.comm.ulysses_lowp as lowp

cap = lowp.capability("cuda")
if not cap["supported"]:
    raise RuntimeError(cap)
layout = getattr(lowp, cap["layout_class"])(head_dim=q.shape[-1])
send, ctx = layout.local_stats(q, k, v, rank=rank, world_size=P, used_sequence=U)
# Caller AllGather: gathered is rank-major FP32 [P, send.numel()] or flat.
stats = layout.finalize_stats(gathered, ctx, k)
payload = layout.quant_and_pack(q, k, v, stats)
# Caller AllToAll: recv[source] is that source's chunk for this destination.
q8, k8, v8, qs, ks = layout.unpack_for_sage(
    recv, batch_size=B, local_sequence=L, local_heads=H // P,
    world_size=P, scale_sequence=U,
)
# Sage receives q8[:, :U], k8[:, :U], packed v8, and output[:, :U].
# V scale is stats.v_scale_global[:, rank*(H//P):(rank+1)*(H//P)].
```

## Statistics and ordering

Each rank sends K per-channel FP32 sums, V per-channel amax,
two Q boundary amax descriptors, and two raw-K boundary min/max descriptors.
For B batches, H heads and head dimension D this is B*H*(6*D+2) FP32 elements.
The layout, B/H/L/P, U and input dtype must agree across ranks.
The context is local and must be consumed with the same shard and layout.

Q grouped-amax is computed before AllGather. After the single AllGather,
finalize computes global K mean and V scale, merges Q boundaries, computes
local centered-K grouped-amax, repairs the live tail group, then merges K
boundaries. Packing consumes these final amax tensors without recomputing them.

For each global boundary group, merge all ranks touching it, including groups
spanning more than two ranks. Empty K slices use min=+inf/max=-inf and
contribute only the amax floor. Entirely padded trailing groups do not
participate in the live oracle.

## Numeric and padding contract

Let S=P*L and 0<U<=S. Padding is zero-filled only at the global sequence
tail [U,S), before sharding. Arbitrary positive L is supported.
Recommended global alignment is the layout's K group; legacy ALIGNED / 3
selects 128*P padding only, while BOUNDARY_MERGE / 2 selects the K group.
Both padding policies execute the same general path. Each layout's
aligned_length uses its own required_alignment.

K sums and global averaging use FP32 with denominator U. Round mean to the
input FP16/BF16 dtype, then convert it back to FP32 for centering.
Valid grouped-amax reduces over tokens and D with FP32 floor 1e-7.
Scale is amax/127; codes use x*(127/amax), round-to-nearest-even and signed
INT8 saturation. Independent references preserve this rounding point;
FP32 SDPA quality references use the original live Q/K/V instead.

V scale remains global channel amax/2.25 on both layouts. A zero-amax channel
encodes positive FP8 zero, with zero V scale; nonzero arithmetic is unchanged.
SM90 448 remains a separate decision requiring controlled GPU evaluation.

Scale widths for U live rows are:

| Layout | Q scale width | K scale width |
|---|---|---|
| SM90 | ceil(U/64)*4 | ceil(U/128) |
| SM89 / SM120 | ceil(U/128)*4 | ceil(U/64) |

The full widths must be allocated and initialized: Sage reads whole warp
groups. Extra Q slots beyond ceil(U/Q_GROUP) may contain finite nonzero
values because they correspond only to invalid Q rows. They need not be zero.
V is allocated to round_up(S,128) on SM90 or round_up(S,64) otherwise.
V uses Sage's global 16-token permutation, independently of rank boundaries.
Storage beyond S is zero. This does not replace caller input padding.

## Payload geometry and outputs

`payload_spec` defines all offsets. For h=H/P, each destination chunk contains:

| Section | Storage |
|---|---|
| Q | B*L*h*D INT8 bytes |
| K | B*L*h*D INT8 bytes |
| V | B*L*h*D FP8 bytes |
| Q scales | B*h*slots(L,Q_GROUP) FP32 values |
| K scales | B*h*slots(L,K_GROUP) FP32 values |
| Alignment tail | round up chunk size to 128 bytes; zero |

slots(L,G)=ceil((L+G-1)/G), an upper bound for all source offsets.
For example, L=50 and G=32 gives three slots: rank 1 owns tokens 50..99,
which intersect groups 1, 2 and 3. Using ceil(L/G)=2 would under-allocate.
A rank's unused payload scale slots and alignment tail are zero.
Data order inside each section is [local_token,batch,local_head,D].
There is no fixed extra 128-byte sentinel when the raw size is already aligned.

The general receiver returns Q/K [B,S,h,D], V [B,D,h,padded_S],
Q/K scales [B,h,width]. `scale_sequence=U` chooses consumer widths without
changing Q/K storage length. `aligned=True` only asserts L%128==0;
None/False accept any legal L. All values use the same receiver kernel.
Input projection views with dense D and 16-byte-aligned outer strides are
supported; outputs must have exact shape, dtype, device and contiguous storage.

## Verification and delivery

- Existing primitive tests: tests/comm/test_ulysses_lowp.py.
- Independent boundary and per-layout tests: tests/comm/test_ulysses_lowp_boundary.py.
- Benchmark: benchmarks/comm/bench_ulysses_lowp.py; rank-local timing excludes network.
- Distributed acceptance: benchmarks/comm/validate_ulysses_lowp.py under torchrun.

CPU structural tests do not certify CUDA behavior. Run GPU acceptance on
SM90 and SM120 separately, recording code revision, local changes, toolchain,
device, numerical reports and skips. GPU acceptance is owned by the user.
CUDA graph / torch.compile integration, training and multi-node acceptance
are outside this delivery. No performance improvement is claimed.

Both benchmark and distributed validator accept `--head-dim {64,128}`.
For example, from the FlashInfer checkout, with SageAttention installed:

```bash
torchrun --standalone --nproc-per-node=8 benchmarks/comm/validate_ulysses_lowp.py \
  --head-dim 64 --local-sequence 65 --used 511 --heads 16 --batch 2 \
  --output lowp_d64_up8.json
```

This exercises real statistics AllGather and payload AllToAll, checks exchanged
bytes and global-grid quantization, and compares Sage attention output against
FP32 SDPA and stock CUDA Sage2. Repeat on each target architecture; compiling a
module for another architecture is not hardware validation.


### D=64 extension validation (2026-09-14)

Validated on NVIDIA H20-3e (SM90), PyTorch 2.13.0+cu130, CUDA 13.0:

| P | B | H | Local L | Live U | Input dtype | V case | Result |
|---|---|---|---|---|---|---|---|
| 8 | 2 | 16 | 65 | 511 | BF16 | Random with two zero channels | Passed |
| 4 | 1 | 28 | 129 | 257 | FP16 | Random with two zero channels | Passed |
| 2 | 1 | 8 | 257 | 513 | BF16 | All zero | Passed |

The two Lowp test files passed 432 cases, with 239 skips from their hardware
and dependency guards. This includes both dimensions, short/nonaligned shards,
multiple statistics chunks, long scale arrays, output reuse and zero V.
Compute Sanitizer memcheck reported zero errors on three D=64 boundary cases.
Six FP16/BF16 D=128 cases at P=2/4/8 retained identical SHA-256 fingerprints
before/after this extension, covering every sender payload and rank-0 unpacked
outputs. The D=64 benchmark CLI also completed; no speedup is claimed here.
SM89/SM120 compilation passed for both dimensions; hardware validation of that
module remains pending. These results do not certify D=64 on SM120.
