"""SM100 NVFP4 grouped GEMM for the per-token MoE, written in CuTeDSL.

``W[N, K] @ A[rows, K].T`` over tiles of ``bm`` rows of one expert, FP4 values with E4M3 block
scales per 16 values and FP32 global scales per row (``a_gs``) and per expert (``w_gs``).

Structure:
- warp roles and ids: the epilogue is the default partition (warps 0..D-1, D = 8 token-major,
  4 weight-major), then one warp each for MMA and TMA loads; the worker warpgroup is padded to 4 warps
  (the activation scale warps of ``x_sf_gather`` take the padding warps);
- registers: ``.maxnreg 128`` at launch, workers ``setmaxnreg`` 32, the epilogue gets the rest
  (176 with 8 epilogue warps, 224 with 4);
- shared memory: operand rings, the epilogue scratch, scale rings, then barriers; in TMEM the accumulator(s)
  (``acc_stages``) followed by the weight then activation scale columns;
- synchronization: raw mbarriers with explicit counts and phases, the accumulator is released by one thread
  after a barrier of the epilogue warps;
- the load warp issues W, W_SF, A[, A_SF] per stage with one expect_tx; with PDL it prefetches
  the first ``stages`` weight tiles before ``griddepcontrol.wait``;
- weight-major epilogue: ``tcgen05.ld.32x32b.x64``, broadcast ``ld.global.v4`` of the row scales, scalar
  predicated ``st.global.b16`` (the down GEMM);
- tcgen05.cp / tcgen05.mma with explicit descriptor constants (CuTeDSL's low-level primitives);
- a plain (non-cluster) launch, except ``two_cta``, which needs CTA pairs.

The TMA goes through CuTe's TMA atoms (descriptors are kernel parameters; block-scale boxes are 3-D and the
instruction carries an L2 cache-hint operand, see HANDWRITTEN_PTX.md); ``fence.mbarrier_init`` follows the barrier
init. The MMA warp issues tcgen05.cp / mma through the low-level primitives (CuTe's tiled MMA and s2t copies spill in
the 32-register MMA warp at BK 512). Both SwiGLU epilogues quantize their output to NVFP4 with block scales only
(``act``). The gate/up rows are interleaved per 8 (``weights.interleave_gate_up``): the weight-major epilogue (short
batches) reads gate and up of an output from TMEM lanes l and l + 8 in one thread (``tcgen05.ld.16x256b``, no scratch
exchange), the token-major one (long batches) reads whole 16-output blocks per thread and stages each tile's FP4
bytes and block scales in its scratch for 16-byte stores; the token-major gate/up can run as CTA pairs (``two_cta``).
The short batches' gate/up places its tiles itself (``place_tiles``: each CTA scans the expert counts into a
shared-memory tile table, CTA 0 publishes it for the down GEMM) and, at block_m <= 32, gathers its activation rows and
block scales by token (``route_rows``: a fourth warpgroup of 16-byte and 4-byte ``cp.async``).

Inline PTX is limited to the cases listed, with their reasons, in ``HANDWRITTEN_PTX.md``.
"""

import functools
import inspect
import threading

import cutlass
import cutlass.cute as cute
import cutlass.memory
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from .activation import as_activation, is_gated
from .placement import l2_prefetch
from cutlass.tensor_utils.tensormap import TensorMapManager, TensorMapUpdateMode
from cutlass import BFloat16, Float32, Int16, Int32, Int64, Uint8, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.experimental import primitives as prims
from cutlass.cutlass_dsl.cutlass import CuTeDSL

SF_VEC = 16
# L2 cache policies of TMA loads (CUTLASS CacheHintSm90)
EVICT_NORMAL = 0x1000000000000000
EVICT_FIRST = 0x12F0000000000000
LOG2E = 1.4426950408889634


# FP8 / FP4 conversions through NVVM ops (the epilogues' quantization; also used by direct.py)
_I8X2 = lambda: ir.VectorType.get([2], ir.IntegerType.get_signless(8))  # noqa: E731
_I32 = lambda: ir.IntegerType.get_signless(32)  # noqa: E731


def cvt_e4m3x2(lo, hi):
    """``cvt.rn.satfinite.e4m3x2.f32`` (lo in byte 0), then ``cvt.rn.f16x2.e4m3x2`` and
    ``cvt.f32.f16`` back. Returns (packed pair as u32, lo and hi as rounded f32). ``.to(Float8E4M3FN)``
    on a TensorSSA converts only whole 32-bit groups (4 values), so the NVVM ops are used."""
    pair = nvvm.convert_f32x2_to_f8x2(
        _I8X2(),
        Float32(hi).ir_value(),
        Float32(lo).ir_value(),
        cutlass.Float8E4M3FN.mlir_type,
        rnd=nvvm.FPRoundingMode.RN,
        sat=nvvm.SaturationMode.SATFINITE,
    )
    h = nvvm.convert_f8x2_to_f16x2(
        ir.VectorType.get([2], ir.F16Type.get()), pair, cutlass.Float8E4M3FN.mlir_type
    )
    f = [
        Float32(
            arith.extf(
                ir.F32Type.get(), llvm.extractelement(h, arith.constant(_I32(), i))
            )
        )
        for i in range(2)
    ]
    word = Uint32(
        arith.extui(_I32(), llvm.bitcast(ir.IntegerType.get_signless(16), pair))
    )
    return word, f[0], f[1]


def cvt_e2m1x2(x0, x1):
    """One FP4 byte (x0 in the low nibble): ``cvt.rn.satfinite.e2m1x2.f32``."""
    b = nvvm.convert_f32x2_to_f4x2(
        Float32(x1).ir_value(), Float32(x0).ir_value(), cutlass.Float4E2M1FN.mlir_type
    )
    return Uint32(arith.extui(_I32(), b))


def sel_if_pos(c, a):
    """a if c > 0 else 0 (``setp.gt.f32`` + ``selp.f32``: the guarded reciprocals)."""
    return Float32(cutlass.select_(c > Float32(0.0), a, Float32(0.0)))


# --------------------------------------------------------------------------------------------
# Registers at launch. GroupedGemm's warps move registers with setmaxnreg (launch 128, the workers 32, the epilogue
# 176-256), so its kernel must start with exactly 128 per thread. CuTeDSL's LaunchConfig has no register cap: .maxnreg
# is added to the kernel's attributes by wrapping the private CuTeDSL._generate_kernel_attrs (the same in CuTeDSL
# 4.8.0 and 4.8.1). A CuTeDSL without that method (or with another signature) gets the public launch bounds instead
# (max_number_threads = 64K / 128, min_blocks_per_mp = 1, like CUTLASS's __launch_bounds__), which also gives 128
# registers. A CuTeDSL that keeps the method but no longer
# calls it is found at the first GroupedGemm's trace: MaxnregPatchUnused, after which moe._compile compiles again with
# the public bounds.
# --------------------------------------------------------------------------------------------


class _PerThread(threading.local):
    """A one-element list of each thread's own: the hook reads the value of the thread that traces the kernel, so a
    concurrent compilation in another thread (of a firefly kernel or any other CuTe DSL kernel) sees its own."""

    def __init__(self, v):
        self.v = [v]

    def __getitem__(self, i):
        return self.v[i]

    def __setitem__(self, i, x):
        self.v[i] = x


_MAXNREG = _PerThread(None)  # the cap of the kernel this thread is tracing
_MAXNREG_SET = _PerThread(0)  # kernels this thread has given .maxnreg so far
_orig_kernel_attrs = getattr(CuTeDSL, "_generate_kernel_attrs", None)


def _patchable(f):
    try:
        return list(inspect.signature(f).parameters) == ["self", "config"]
    except (TypeError, ValueError):
        return False


MAXNREG_PATCH = _patchable(
    _orig_kernel_attrs
)  # False: GroupedGemm uses the public launch bounds


def _kernel_attrs(self, config):
    ret = _orig_kernel_attrs(self, config)
    if _MAXNREG[0] is not None:
        ret["nvvm.maxnreg"] = ir.Attribute.parse(f"{_MAXNREG[0]} : i32")
        _MAXNREG_SET[0] += 1
    return ret


if MAXNREG_PATCH:
    CuTeDSL._generate_kernel_attrs = _kernel_attrs


class MaxnregPatchUnused(RuntimeError):
    """The .maxnreg patch had no effect on a kernel; MAXNREG_PATCH is False from now on: trace the kernel again."""


def launch_regs(launch, maxnreg, **kw):
    """launch(**kw) of a kernel that must start with exactly maxnreg registers per thread (see above)."""
    global MAXNREG_PATCH
    if not MAXNREG_PATCH:
        return launch(
            max_number_threads=[65536 // maxnreg, 1, 1], min_blocks_per_mp=1, **kw
        )
    done = _MAXNREG_SET[0]
    _MAXNREG[0] = maxnreg
    try:
        launch(**kw)
    finally:
        _MAXNREG[0] = None
    if _MAXNREG_SET[0] == done:
        MAXNREG_PATCH = False
        raise MaxnregPatchUnused(
            "CuTeDSL no longer calls CuTeDSL._generate_kernel_attrs: the .maxnreg patch has no "
            "effect; compile again (the public launch bounds from now on)"
        )


# --------------------------------------------------------------------------------------------
# Inline PTX helper, only where CuTeDSL's own code generation is not usable (direct.py [P1, P2],
# reduce.py [P4]; this file has none). Each case and its evidence: HANDWRITTEN_PTX.md.
# --------------------------------------------------------------------------------------------
def _asm(res, operands, asm, constraints, side_effects, loc, ip):
    return llvm.inline_asm(
        res,
        operands,
        asm,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def st_global_v4(addr, vals):
    """``st.global.v4.b32`` of 4 words through ``cute.arch.store``; callers predicate it with a DSL ``if``."""
    r = cute.make_rmem_tensor((4,), Uint32)
    for i in range(4):
        r[i] = vals[i]
    cute.arch.store(
        cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=16),
        r.load().ir_value(),
    )


# --------------------------------------------------------------------------------------------
# Floating-point instruction forms through CuTeDSL
# --------------------------------------------------------------------------------------------
def fma_zero(a, b):
    """``fma.rn.f32 x, a, b, 0`` (the form of -gs * acc)."""
    return cute.math.fma(a, b, Float32(0.0))


def ex2_approx(a):
    """``ex2.approx.f32`` without ftz (``cute.math.exp2``; ``cute.arch.exp2`` is the ftz form)."""
    return cute.math.exp2(a)


def div_full(a, b):
    """``div.full.f32`` (``/`` is the IEEE ``div.rn.f32``)."""
    return cute.math.div(a, b, full=True)


# --------------------------------------------------------------------------------------------
# tcgen05.cp / tcgen05.mma with explicit descriptors, through CuTeDSL's low-level primitives:
# CuTe's tiled MMA / s2t copies hoist the per-k scale descriptors out of the loop and spill in the
# 32-register MMA warp at BK = 512.
# --------------------------------------------------------------------------------------------
def tmem_ptr(addr):
    """LLVM TMEM pointer (address space 6) for the NVVM-level primitives (a ``cute.make_ptr(...,
    tmem)`` or its ``.llvm_ptr`` leaves an unrealized cast that libNVVM rejects)."""
    return llvm.inttoptr(llvm.PointerType.get(6), Int32(addr).ir_value())


def tcgen05_cp_32x128b_x4(taddr, sdesc, two_cta=False):
    """``tcgen05.cp.cta_group::1.warpx4.32x128b``: 512 B of block scales (128 rows x 4) to TMEM (``two_cta``:
    cta_group::2, issued by the pair's leader: each CTA's shared memory to its own TMEM)."""
    prims.tcgen05_cp(
        prims.Tcgen05CpShape.SHAPE_32X128B,
        tmem_ptr(taddr),
        sdesc,
        group=prims.CTAGroup.CTA_2 if two_cta else prims.CTAGroup.CTA_1,
        multicast=prims.Tcgen05CpMulticast.WARPX4,
    )


def tcgen05_mma_nvf4(
    d_taddr, adesc, bdesc, idesc: int, asf_taddr, bsf_taddr, accumulate, two_cta=False
):
    """``tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`` (= ``scale_vec::4X``: one E4M3
    scale per 16 values). BLOCK16 must be named: the primitive maps ``Tcgen05MMAScaleVecSize.X4`` to
    block32 (and X2 to block16), the opposite of the PTX definition."""
    prims.tcgen05_mma_block_scale(
        prims.Tcgen05MMAKind.MXF4NVF4,
        prims.CTAGroup.CTA_2 if two_cta else prims.CTAGroup.CTA_1,
        tmem_ptr(d_taddr),
        Int64(adesc),
        bdesc,
        idesc,
        accumulate,
        tmem_ptr(asf_taddr),
        tmem_ptr(bsf_taddr),
        scale_vec_size=prims.Tcgen05MMAScaleVecSize.BLOCK16,
    )


# Descriptor constants: K-major operands with 128B
# swizzle (SBO 1024 B, version 1); block scales without swizzle (SBO 128 B, version 1).
OPERAND_DESC = 0x4000404000000000
# K-major operands of 64-byte rows (K tile 128 FP4: 64B swizzle, layout type 4; SBO 8 rows x 64 B = 512 B)
OPERAND_DESC_SW64 = 0x8000402000000000
SCALE_DESC = 0x400800000000


def mma_idesc(m: int, n: int) -> int:
    """Instruction descriptor of kind::mxf4nvf4 with E2M1 A/B and K-major operands."""
    return 0x480 | (n >> 3) << 17 | (m >> 4) << 24


# --------------------------------------------------------------------------------------------
# CuTeDSL building blocks addressed by integer shared-memory / TMEM addresses.
# Plain Python: they emit IR when traced.
# --------------------------------------------------------------------------------------------
def smem_ptr(dtype, addr, align=16):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)


def _vec(n, dtype):
    return ir.VectorType.get([n], dtype.mlir_type)


def ld_shared_v4(addr, dtype=Uint32):
    """``ld.shared.v4.b32`` -> 4 values."""
    v = cute.TensorSSA(
        cute.arch.load(smem_ptr(dtype, addr), _vec(4, dtype)), (4,), dtype
    )
    return [v[i] for i in range(4)]


def st_shared_v4(addr, vals, dtype=Uint32):
    """``st.shared.v4.b32`` of 4 values."""
    r = cute.make_rmem_tensor((4,), dtype)
    for i in range(4):
        r[i] = vals[i]
    cute.arch.store(smem_ptr(dtype, addr), r.load().ir_value())


def ld_global_v4_f32(addr):
    """``ld.global.v4.b32`` of 4 FP32 at a 16-byte aligned address."""
    p = cute.make_ptr(Float32, addr, cute.AddressSpace.gmem, assumed_align=16)
    v = cute.TensorSSA(cute.arch.load(p, _vec(4, Float32)), (4,), Float32)
    return [v[i] for i in range(4)]


def ld_global_v4_u32(addr):
    """``ld.global.v4.b32`` of 4 32-bit integers at a 16-byte aligned address."""
    p = cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=16)
    v = cute.TensorSSA(cute.arch.load(p, _vec(4, Uint32)), (4,), Uint32)
    return [v[i] for i in range(4)]


def prefetch_tensormap(addr):
    """``prefetch.tensormap`` of a descriptor in global memory (generic address)."""
    cute.arch.prefetch(
        cute.make_ptr(Int64, addr, cute.AddressSpace.generic, assumed_align=64),
        tensormap=True,
    )


def cvt_bf16x2(lo, hi, packed=False):
    """``cvt.rn.bf16x2.f32`` of two FP32: the two BF16 values, or with ``packed`` one word with ``lo``
    in the low half."""
    f = cute.make_rmem_tensor((2,), Float32)
    f[0] = lo
    f[1] = hi
    b = cute.make_rmem_tensor((2,), BFloat16)
    b.store(f.load().to(BFloat16))
    return cute.recast_tensor(b, Uint32)[0] if packed else (b[0], b[1])


def gmem_bf16(addr):
    """One-element BF16 tensor at a global address (CuTe element store: ``st.global.b16``)."""
    return cute.make_tensor(
        cute.make_ptr(BFloat16, addr, cute.AddressSpace.gmem, assumed_align=2),
        cute.make_layout(1),
    )


def st_global(addr, val, dtype, align):
    """``st.global`` of one 8/32-bit integer value (a CuTe element store)."""
    p = cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)
    cute.make_tensor(p, cute.make_layout(1))[0] = dtype(val)


def u32_as_f32(w):
    """The FP32 with the bits of a Uint32 (a register reinterpretation)."""
    t = cute.make_rmem_tensor((1,), Uint32)
    t[0] = w
    return cute.recast_tensor(t, Float32)[0]


def relu2_bf16_pair(gs, a0, a1):
    """ReLU2's outputs of two accumulators, scaled by gs and rounded to BF16, as FP32: relu(gs a)^2 with packed
    multiplies (``mul.rn.f32x2``, the same products) and one ``cvt.rn.bf16x2.f32`` for the pair (the same rounding)."""
    h = mul_f32x2(gs, gs, a0, a1)
    r0 = cute.arch.fmax(h[0], Float32(0.0))
    r1 = cute.arch.fmax(h[1], Float32(0.0))
    q = mul_f32x2(r0, r1, r0, r1)
    w = cvt_bf16x2(q[0], q[1], packed=True)
    return u32_as_f32(w << 16), u32_as_f32(w & Uint32(0xFFFF0000))


def swiglu_wm_pair(sg, g0, g1, u0, u1):
    """``silu_mul_scaled`` of two outputs (gate g, up u, both times sg), rounded to BF16, as FP32: the weight-major
    epilogue's pair of a thread (TMEM lane groups j = 0, 1), with ``mul.rn.f32x2`` / ``add.rn.f32x2`` /
    ``sub.rn.f32x2`` where the scalar sequence rounds the same and one ``cvt.rn.bf16x2.f32``: the same values."""
    g = mul_f32x2(sg, sg, g0, g1)
    u = mul_f32x2(sg, sg, u0, u1)
    t = cute.arch.sub_packed_f32x2((Float32(0.0), Float32(0.0)), g)
    t = mul_f32x2(t[0], t[1], Float32(LOG2E), Float32(LOG2E))
    d = cute.arch.add_packed_f32x2(
        (ex2_approx(t[0]), ex2_approx(t[1])), (Float32(1.0), Float32(1.0))
    )
    o = mul_f32x2(u[0], u[1], div_full(g[0], d[0]), div_full(g[1], d[1]))
    w = cvt_bf16x2(o[0], o[1], packed=True)
    return u32_as_f32(w << 16), u32_as_f32(w & Uint32(0xFFFF0000))


def mul_f32x2(a0, a1, b0, b1):
    """Packed multiply: ``mul.rn.f32x2`` (CuTeDSL requires a rounding mode; ``mul.f32x2`` without one
    rounds the same and no add follows, so the products are identical)."""
    return cute.arch.mul_packed_f32x2((a0, a1), (b0, b1), rnd="rn")


def tmem_load(taddr, shape, stride, rep: int, thr):
    """``tcgen05.ld.32x32b.x{rep}`` of 128 TMEM lanes x ``shape`` columns (column ``stride``) at
    ``taddr`` through CuTe's TMEM copy (warp w reads lanes 32 (w % 4)..; ``thr`` < 128), then
    ``tcgen05.wait::ld``. Returns this thread's values in column order."""
    t = cute.make_tensor(
        cute.make_ptr(Float32, taddr, cute.AddressSpace.tmem, assumed_align=16),
        cute.make_layout((128, shape), stride=(65536, stride)),
    )
    atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{rep}")), Float32
    )
    tiled = tcgen05.make_tmem_copy(atom, t)
    thr_copy = tiled.get_slice(thr)
    dst = cute.make_rmem_tensor(thr_copy.partition_D(t).shape, Float32)
    cute.copy(tiled, thr_copy.partition_S(t), dst)
    cute.arch.fence_view_async_tmem_load()
    return [dst[i] for i in range(cute.size(dst))]


def tmem_ld_16x256b(taddr, rep: int):
    """``tcgen05.ld.sync.aligned.16x256b.x{rep}.b32`` at ``taddr`` (lane base in bits 16 and up): 4 rep FP32 values; per
    repetition r (columns 8 r ..) thread t holds (lane t / 4, columns 2 (t % 4), + 1), then (lane t / 4 + 8, the same
    columns). The caller waits (``tcgen05.wait::ld``)."""
    v = nvvm.tcgen05_ld(
        ir.VectorType.get([4 * rep], ir.IntegerType.get_signless(32)),
        nvvm.Tcgen05LdStShape.SHAPE_16X256B,
        tmem_ptr(taddr),
    )
    vf = llvm.bitcast(ir.VectorType.get([4 * rep], ir.F32Type.get()), v)
    t = cute.TensorSSA(vf, (4 * rep,), Float32)
    return [t[i] for i in range(4 * rep)]


def st_global_v2(addr, w0, w1):
    """``st.global.v2.b32`` of two words at an 8-byte aligned address."""
    r = cute.make_rmem_tensor((2,), Uint32)
    r[0] = w0
    r[1] = w1
    cute.arch.store(
        cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=8),
        r.load().ir_value(),
    )


def ld_shared_v2_f32(addr):
    """``ld.shared.v2.f32`` -> 2 values."""
    v = cute.TensorSSA(
        cute.arch.load(smem_ptr(Float32, addr, 8), _vec(2, Float32)), (2,), Float32
    )
    return [v[0], v[1]]


# Typed pointers to TMA descriptors in global memory (CuTeDSL's tensormap utility).
_TMAPS = TensorMapManager(TensorMapUpdateMode.GMEM, 128)


def bar_sync(barrier_id: int, threads: int):
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)


def bar_sync_cta(barrier_id: int):
    """CTA-wide ``barrier.sync`` without ``.aligned``, for barriers the warp roles reach from different
    instructions (``cute.arch.barrier`` emits ``bar.sync``, i.e. ``barrier.sync.aligned``)."""
    nvvm.barrier_cta_sync(Int32(barrier_id).ir_value(), aligned=False)


def silu_mul_scaled(gate, up):
    """gate / (1 + exp(-gate)) * up on scaled values (the sequence
    ``sub.f32 0, x``, ``mul`` by log2(e), ``ex2.approx``, ``+1``, ``div.full``, ``mul``)."""
    e = ex2_approx((Float32(0.0) - gate) * Float32(LOG2E))
    return up * div_full(gate, e + Float32(1.0))


# --------------------------------------------------------------------------------------------
# Host TMA descriptors for the direct GEMM (direct.py): UINT8, interleave none, no OOB fill, 256 B L2 promotion;
# dims and boxes are listed innermost first.
# --------------------------------------------------------------------------------------------
def _encode(ptr, dims, strides, box, swizzle128):
    from cuda.bindings import driver as cu

    err, tmap = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        cu.cuuint32_t(len(dims)),
        ptr,
        [cu.cuuint64_t(v) for v in dims],
        [cu.cuuint64_t(v) for v in strides],
        [cu.cuuint32_t(v) for v in box],
        [cu.cuuint32_t(1)] * len(dims),
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle128
        else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return [int(v) for v in tmap.opaque]


# --------------------------------------------------------------------------------------------
# Grouped GEMM
# --------------------------------------------------------------------------------------------
def use_pdl(op, launch_pdl=None):
    """The kernel launch's PDL attribute: the op's ``launch_pdl`` (or ``launch_pdl``) unless ``op.no_pdl`` (set by
    ``moe.CuteMoE(enable_pdl=False)``: the same kernels launched without PDL, griddepcontrol then a no-op and the
    kernels serialized by the stream)."""
    return (op.launch_pdl if launch_pdl is None else launch_pdl) and not getattr(
        op, "no_pdl", False
    )


class GroupedGemm:
    """Persistent NVFP4 grouped GEMM over expert tiles.

    Args:
        n: Weight rows per expert (2I for gate/up with SwiGLU, H for down).
        k: Reduction size.
        bm: Rows (routes) per tile; tiles are padded to bm rows of one expert.
        bn: Weight rows per tile; 256 makes the accumulator token-major (the gate/up of long batches, bm = 128,
            x_sf_gather 3, NVFP4 output: ``_epilogue_token_major``) unless ``token_major`` says otherwise.
        bk: K per pipeline stage.
        stages: K stages in flight.
        act: SwiGLU of the gate/up rows interleaved per 8 (``weights.interleave_gate_up``; output N / 2 columns as
            NVFP4, see below), else BF16 [rows, N].
        launch_pdl: Prefetch weights before ``griddepcontrol.wait`` and trigger dependents.
        evict_weights: Load an expert's weights with L2 EVICT_FIRST on its last row block (EVICT_NORMAL
            before, while other row blocks still reuse them), as FlashInfer's swap_ab MoE does.
        claim_tiles: Above 0, in a launch with at least this many tiles (``moe.claim_tiles``), after its two
            round-robin tiles (``bidx``, ``bidx + grid``) each CTA takes the next from the global counter at
            ``claims_ptr`` (zero on entry; the launch's last claim zeroes it again), one tile ahead, instead of
            ``tile += grid``; the load warp passes the tile ids to the MMA and epilogue warps through a shared-memory
            ring.
        x_sf_gather: The activation block scales are linear rows of K / 16 bytes that two scale warps
            (``_sf_warp``) load per stage for the tile's rows and store into the stage's 128x4
            block, instead of the TMA loading a tile-packed copy (``moe.sf_gather_mode``). 1: the MoE input's rows
            [tokens, K / 16] at ``sfa_ptr``, through ``rowtok_ptr`` (Int32 [rows], the token of each row), loaded by the scale warps;
            3: the scatter's row-major copy [rows, K / 16] at ``sfa_ptr`` through the TMA, a bm x bk / 16 box per
            stage into the stage's scale block (row-major, completing on ``sf_full``), which one of the two scale
            warps (by slot) then rearranges in place into the 128x4 layout (``_sf_convert_warp``: no global loads
            in the scale warps). The token-major gate/up uses mode 3 (its MMA's A scales).
        unit_a_gs: The activation row scales are all 1, neither loaded nor multiplied (acc * w_gs; acc * (1 * w_gs) is
            the same): the down GEMM after a gate/up with act, and the gate/up of per-tensor input
            (``moe.CuteMoE(per_token_scale=False)``; its epilogues then stage no row scales, ``sScales`` is empty).
        two_cta: CTA pairs (cluster [2, 1, 1]); the pair's leader (rank 0) issues the MMAs (cta_group::2). Token-major
            (the long batches' gate/up): MMAs of 256 tokens x bn weight rows; each CTA loads its 128 tokens, half of
            the bn weight rows and the weight scales of all bn rows, and stores its 128 tokens' outputs. The rows must
            be padded to 256 per expert (pair tile p: the 128-row blocks 2 p and 2 p + 1;
            ``ScatterChunks(pad_m=256)``). Token-major without act (the long batches' down GEMM, bn 256): the same
            MMAs over the pair tiles of ``pair_table``. Weight-major (``wm2``: the long batches' unfused down
            GEMM, bm 128, bn 128, n % 256 == 0): MMAs of 256 weight rows x bm tokens; pair tile q covers row block
            q // (n / 256) and N tiles 2 (q % (n / 256)) + rank; each CTA loads its 128 weight rows, half of the bm
            tokens and the activation scales of all bm, and stores its 128 weight rows' outputs for the bm tokens.
        skip_empty: Weight-major without PDL or claiming (the down GEMM after a two_cta gate/up): every role skips the
            row blocks at or past their expert's routes (the empty half of a 256-row padding), in the same sequence.
        fused_out: The down GEMM's fused finalize (``CuteMoE(use_fused_finalize=True)``): the epilogue adds
            acc * w_gs * a_gs[row] (a_gs: each row's routing weight) into the BF16 output row [tokens, N] of the row's
            token (rowtok_ptr: Int32 [rows]) instead of storing BF16 rows [rows, N]; the output is zeroed before
            (``Scatter(fused)``, ``ScatterChunks(fused)``). Short batches: weight-major with BF16 atomic adds
            (``_epilogue_fused``); long batches: token-major (``token_major``) with one bulk reduce per row
            (``_epilogue_token_major_fused``), 1-CTA or as CTA pairs (``two_cta``, ``pair_table``).
        pair_table: The token-major down GEMM's CTA pairs' tiles (fused_out, or the long batches' unfused down GEMM:
            each tile row stored into its row of out [rows, N] with one bulk copy): count_ptr holds the pair tiles'
            count, tb_ptr pair tile p's first 128-row block b0 (``ScatterChunks(pairs=True)``: each expert's row
            blocks two by two, the last alone at an odd count). The pair's CTA rank takes the rows of block b0 + rank
            of b0's expert (rank 1 past them at an odd count: computed, none stored); both CTAs load b0's expert's
            weights.
        token_major: The MMA's M is the tile's bm = 128 tokens (activations as operand A), N the bn weight rows
            (bn 256). The long batches' gate/up (act, x_sf_gather 3) and, with fused_out, the long batches'
            down GEMM (x_sf_gather 0: the TMA brings the tile-packed activation scales as the MMA's A scales).
        clear: With fused_out: Int32 count at clr_ptr zeroed after the wait (the expert route counts, which ReduceRoutes
            zeroes without fusion).

    With act: the epilogue quantizes the SwiGLU output, rounded to BF16, to NVFP4 with a unit second-level
    scale (block scale e4m3(bmax / 6) per 16 values, no row scale; FlashInfer prims_ts' per-token intermediate) and
    stores the FP4 rows [rows, N / 4 bytes] at out_ptr and the block scales packed per tile at osf_ptr: [rows / bm,
    N / 128, 512], one 128x4-swizzled block per tile and 64 output columns (the down GEMM's activation scales).
    """

    MAXNREG = 128

    def __init__(
        self,
        n,
        k,
        bm,
        bn,
        bk,
        stages,
        act,
        *,
        launch_pdl=False,
        l2_tiles=0,
        evict_weights=False,
        claim_tiles=0,
        x_sf_gather=0,
        unit_a_gs=False,
        two_cta=False,
        skip_empty=False,
        acc_stages=1,
        fused_out=False,
        clear=0,
        token_major=False,
        x_gather=False,
        activation=None,
        route_rows=0,
        place_tiles=0,
        max_tiles=0,
        tokens=0,
        zero_out=0,
        zero_rows=0,
        place_experts=0,
        k_tail=False,
        pair_table=False,
    ):
        # route_rows = topk (short batches at block_m <= 32, no Scatter): the tiles' rows are routes listed per expert
        # by SelectExperts(eroute): row i of tile t is route eroute[tbase[t] + i] (rowtok_ptr; token = route / topk).
        # The weight-major gate/up gathers its activation rows and block scales by token (x_gather with x_sf_gather 2)
        # and its epilogue the tokens' row scales (a_gs_ptr: the MoE input's [tokens]); the fused down GEMM takes each
        # row's token and routing weight (a_gs_ptr: weights [tokens * topk]) from it (tb_ptr: the published tbase).
        self.route_rows = route_rows
        # place_tiles (short batches' gate/up): every CTA computes the tile table (experts, limits, tbase, count) from
        # the expert counts (clr_ptr, place_experts of them) into its shared memory (_place_tiles), 1: after its PDL
        # wait (no Scatter: the primary is SelectExperts, which triggers at its start), 2: before it (the primary
        # Scatter triggers right after its own wait, before it copies the rows). CTA 0 also publishes the table
        # (experts_ptr, limits_ptr, count_ptr, tb_ptr) for the down GEMM; the CTAs trigger their dependents after it.
        # max_tiles: the table's size; tokens: eroute's stride (M); zero_out = H (place_tiles 1 with the fused
        # finalize): the gather warps zero the output rows (inv_ptr) of the routes in slot 0 instead of writing inverse.
        self.place_tiles, self.max_tiles, self.tokens = place_tiles, max_tiles, tokens
        self.zero_out, self.place_experts = zero_out, place_experts
        # zero_rows = H (the long batches' token-major gate/up with the fused finalize): the load warp zeroes the
        # [M, H] BF16 output (inv_ptr; M: tb_ptr's Int32) for the down GEMM's adds, after issuing each tile's stages
        # the rows tile + j * num_tiles (CTA pairs: each CTA its half of the columns); without tiles, CTA b the rows
        # b + j * grid. The load warp only issues the weights' TMA there (x_gather): it has the slack.
        self.zero_rows = zero_rows
        assert not zero_rows or (token_major and x_gather), (
            "zero_rows: the token-major gate/up"
        )
        self.place_words = 3 * max_tiles + 1 if place_tiles else 0
        assert not place_tiles or (
            act
            and bn != 256
            and max_tiles > 0
            and place_experts > 0
            and (place_tiles == 2 or (route_rows and tokens > 0))
        )
        # the gated activation of the act epilogues (activation.py; None: the plain SwiGLU, with its own sequences)
        self.activation = as_activation(activation)
        # a non-gated activation (activation.ReLU2): the n weight rows are the outputs (no gate rows,
        # no interleave), out_cols = n instead of n / 2; a weight-major tile has 128 outputs (8 blocks of 16, two
        # 128x4 scale blocks), a token-major one bn = 128 (the same 128 outputs per tile as the gated bn 256)
        self.gated = is_gated(self.activation) if act else True
        self.out_cols = (n // 2 if self.gated else n) if act else n
        assert self.gated or (bn == 128 or (bn == 256 and token_major is not False)), (
            "non-gated: unit FP4 output; 128 weight rows per weight-major tile, 128 or 256 per token-major one"
        )
        self.sfo_bytes = (512 if self.gated else 1024) if act else 0
        # the only non-gated activation is ReLU2, which both act epilogues compute with packed multiplies (scale,
        # square, quantization scale) and one BF16 conversion per pair (relu2_bf16_pair), bit-identical to the scalar
        # sequence. Its outputs are >= 0, so the block max is a plain max (no abs, no min)
        assert self.gated or type(self.activation).__name__ == "ReLU2", (
            "non-gated: ReLU2 only"
        )
        # the plain SwiGLU in the weight-major act epilogue the same way (swiglu_wm_pair; the quantization scale packed
        # too), bit-identical; the token-major epilogue keeps the scalar sequence
        self.token_major = token_major
        self.packed_g = self.gated and self.activation.plain and not token_major
        assert not self.token_major or (
            bm == 128 and x_sf_gather == (3 if act else 0)
        ), "token-major: bm 128; the gate/up with x_sf_gather 3, the down GEMM with 0"
        assert not self.token_major or act or fused_out or two_cta, (
            "the token-major down GEMM: the fused finalize's, or CTA pairs"
        )
        assert self.token_major or bn == 128
        assert bm in (16, 32, 64, 128)
        # bk 128 (K-major 64-byte operand rows, 64B swizzle): the down GEMM where 256 does not divide K = I
        # k_tail: K need not be a multiple of bk (only of 128): the K loop takes ceil(K / bk) stages,
        # the last box partly past K, which the TMA fills with zeros (data and scales: zero products). The down GEMM of
        # the non-gated shapes with I % 256 != 0 (Nemotron 3 Super, I = 2688) at K 512
        self.k_tail = k_tail
        # n_tail: the token-major non-gated gate/up with bn 256 where 256 does not divide n = I
        # (Nemotron 3 Super, I = 2688): the weights and their scales are described per expert, (n, k, E), and addressed
        # by (N tile, K tile, expert), so the last N tile's rows past n are zeros from the TMA; its epilogue stores no
        # output past n
        self.n_tail = n % bn != 0
        assert (
            (
                n % bn == 0
                or (
                    n % 128 == 0
                    and self.n_tail_ok(act, activation, bn, token_major, two_cta)
                )
            )
            and (k % bk == 0 or (k_tail and k % 128 == 0))
            and (bk % 256 == 0 or bk == 128)
        )
        self.kt = -(-k // bk)  # K stages per tile
        self.nt = -(-n // bn)  # N tiles per expert
        self.n, self.k, self.bm, self.bn, self.bk, self.stages, self.act = (
            n,
            k,
            bm,
            bn,
            bk,
            stages,
            act,
        )
        self.launch_pdl = launch_pdl
        # token-major: operand TMA through per-CTA descriptor copies with 256 B L2 promotion (``_l2_256_desc``). Needs
        # tmap_ptr (grid_ctas x 512 B).
        # l2_tiles (PDL, weight-major): before its griddepcontrol.wait the load warp also L2-prefetches the weights
        # and weight scales of the CTA's tiles 1 .. l2_tiles (cp.async.bulk.prefetch.L2 holds the kernel's completion).
        self.l2_tiles = l2_tiles
        # The token-major SwiGLU computes x * sigmoid(x) as x * rcp.approx(1 + 2^(-x log2 e)) (the weight-major one:
        # div.full). The weight-major epilogues stage the tile's row scales: each lane loads 4 of them before the
        # accumulator wait, each warp puts them in its 4 * bm bytes of shared memory after it and reads them back as
        # broadcast ld.shared.v4.
        # evict_weights: weights read for the last time enter L2 as EVICT_FIRST, so they replace each other rather
        # than other lines.
        self.evict_weights = evict_weights
        # x_sf_gather 3 also for the token-major gate/up (its activation scales are the MMA's A scales)
        assert x_sf_gather in (0, 1, 2, 3)
        assert not x_sf_gather or x_sf_gather == 3 or not self.token_major, (
            "x_sf_gather"
        )
        # x_sf_gather 2 (route_rows): the gather warps load the block scales with the rows (into the 128x4 layout);
        # the two scale warps only wait for the gathered stage, fence the async proxy and arrive on stage_ready
        assert x_sf_gather != 2 or (x_gather and route_rows)
        assert x_sf_gather != 3 or bm >= 32, (
            "x_sf_gather 3: every lane of a scale warp stores rows"
        )
        self.x_sf_gather = x_sf_gather
        # act: the SwiGLU epilogues store NVFP4 with block scales only (see the class docstring). Token-major
        # (``_epilogue_token_major``): each tile's FP4 bytes are staged in the scratch (16-byte chunks XOR-swizzled by
        # row) and leave as 16-byte stores of whole 64-byte row segments
        # two_cta (class docstring): the MMA covers 256 tokens, so each CTA loads half of the bn weight rows per stage;
        # its convert and epilogue warps arrive on the leader's barriers. Those remote arrivals use CTA scope as
        # CUTLASS's 2-CTA pipelines do (``mbarrier.arrive.shared::cluster`` with .cta semantics). The TMEM is freed
        # after a cluster-scope handshake.
        self.two_cta = two_cta
        # wm2 (weight-major two_cta: the long batches' unfused down GEMM): the pair's MMA covers 256 weight rows (each
        # CTA its 128: N tile 2 (q % TNT) + rank of pair tile q) x bm tokens (each CTA loads bm / 2 of them and the
        # activation scales of all bm); each CTA's accumulator holds its 128 weight rows x the bm tokens
        self.wm2 = two_cta and not self.token_major
        assert not two_cta or (
            (self.token_major and not launch_pdl and not claim_tiles)
            or (
                self.wm2
                and not act
                and not fused_out
                and not launch_pdl
                and not claim_tiles
                and not x_sf_gather
                and bn == 128
                and bm == 128
                and n % 256 == 0
            )
        ), "two_cta: the token-major gate/up or the weight-major unfused down GEMM"
        self.cta_bn = bn // 2 if two_cta and self.token_major else bn
        self.cta_bm = bm // 2 if self.wm2 else bm
        self.dealloc_n = (
            1 if two_cta else 0
        )  # the pair's TMEM handshake barrier (weight-major layout: wm2 only)
        # the tile enumeration: tiles per row block (wm2: pairs of N tiles) and rows per tile (token-major pairs: 2 bm)
        self.tnt = self.nt // 2 if self.wm2 else self.nt
        self.pair_rows = 2 * bm if two_cta and self.token_major else bm
        # pair_table (class docstring): the token-major fused_out pairs' tiles from the table (count_ptr: pair tiles)
        self.pair_table = pair_table
        assert pair_table == (two_cta and self.token_major and not act) and not (
            pair_table and evict_weights
        ), "pair_table: the token-major down GEMM's CTA pairs (no evict_weights)"
        if pair_table:
            self.pair_rows = 1
        # the MMA instruction's M x N (cta_group::2: both CTAs' rows)
        if self.wm2:
            self.mma_shape = (2 * bn, bm)
        elif two_cta:
            self.mma_shape = (2 * bm, bn)
        else:
            self.mma_shape = None  # mma_mn
        # stage_ready arrivals: the load warp's expect_tx (two_cta: the leader's), and the scale warps with
        # x_sf_gather (two_cta, x_sf_gather 3: both CTAs' convert warps)
        if two_cta:
            self.ready_count = 1 if self.wm2 or not x_sf_gather else 3
        else:
            self.ready_count = 2 if x_sf_gather else 1
        self.skip_empty = skip_empty
        assert not skip_empty or not (
            (self.token_major and act) or claim_tiles or launch_pdl
        )
        # unit_a_gs: no activation row scales (all 1); not with fused_out (assert below), whose row scales are the
        # routing weights
        self.unit_a_gs = unit_a_gs
        # the weight-major epilogues' row-scale staging (4 * bm FP32 per epilogue warp); none for a gate/up with
        # unit_a_gs (the down GEMM's layout is kept as it is)
        self.scales_bytes = 0 if unit_a_gs and act else 4 * 4 * bm
        # fused_out weight-major: only the short batches' down GEMM (PDL); the long batches' is token-major
        assert not fused_out or (
            not act
            and not unit_a_gs
            and (
                (self.token_major and not launch_pdl) or (launch_pdl and not skip_empty)
            )
        ), (
            "fused_out: the weight-major down GEMM of a short batch, or the token-major one of a long batch"
        )
        assert not clear or fused_out
        self.fused_out, self.clear = fused_out, clear
        # claim_tiles: each CTA's first two tiles are round-robin, the rest claimed. The load warp runs at most
        # 1 + ceil(stages / KT) tiles ahead of the epilogue, so a ring one deeper never makes it wait for a free slot.
        # (weight-major only: the token-major gate/up runs without PDL and does not claim)
        assert not claim_tiles or not self.token_major, "claim_tiles: weight-major"
        self.dynamic = claim_tiles > 0
        self.claim_tiles = claim_tiles
        # acc_stages (weight-major): TMEM accumulators that the MMA warp and the epilogue alternate between (tile i in
        # i % acc_stages), so a tile's epilogue overlaps the next tile's MMAs (FlashInfer prims_ts: 2)
        assert acc_stages in (1, 2) and (
            acc_stages == 1 or not self.token_major or not act
        )
        self.acc_stages = acc_stages
        # (the load warp runs at most acc_stages + ceil(stages / KT) tiles ahead of the epilogue)
        self.ring = (
            1 + acc_stages + (stages + self.kt - 1) // self.kt if self.dynamic else 0
        )
        self.mma_mn = (bm, self.cta_bn) if self.token_major else (bn, self.cta_bm)
        # the B scales' rows in each CTA (two_cta: all of the MMA's N: the token-major pair's bn weight rows, wm2's bm
        # tokens)
        if two_cta:
            self.sfb_rows = bn if self.token_major else bm
        else:
            self.sfb_rows = self.mma_mn[1]
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        # Warp roles: the epilogue warps, then the worker warpgroup (and the gather warpgroup with x_gather).
        self.D = 8 if self.token_major else 4
        # x_gather (the long batches' token-major gate/up): the activation rows are the MoE input's [tokens, K / 2] at
        # a_ptr, gathered per stage by token (rowtok_ptr: Int32 [rows]) with 16-byte cp.async by a fourth warpgroup
        # (thread g: chunk g % 8 of rows g / 8 + 16 i), 128B-swizzled as the TMA would, completing on a_full (one
        # cp.async.mbarrier.arrive.noinc per thread); the scale-convert warps wait for it before their proxy fence
        self.x_gather = x_gather
        assert (
            not x_gather
            or (self.token_major and act and bk == 256 and x_sf_gather == 3)
            or (not self.token_major and act and x_sf_gather == 2)
        ), "x_gather: the token-major gate/up (K 256) or route_rows"
        self.G = 4 if x_gather else 0
        order = (
            ["sfg0", "sfg1", "mma", "load"]
            if x_sf_gather
            else ["pad", "pad", "mma", "load"]
        )
        self.role_warp = {r: self.D + i for i, r in enumerate(order) if r != "pad"}
        self.threads = 32 * (self.D + 4 + self.G)
        self.epi_threads = 32 * self.D
        self.regs_epi = min(
            256,
            (self.MAXNREG * (self.D + 4 + self.G) - 32 * (4 + self.G))
            // self.D
            // 8
            * 8,
        )
        self.regs_worker = 32
        # Bytes per stage
        self.w_bytes = self.cta_bn * bk // 2
        self.a_bytes = self.cta_bm * bk // 2
        self.wsf_bytes = bn // 128 * bk // 64 * 512
        self.asf_bytes = bk // 64 * 512
        # bytes the TMA brings per stage on stage_ready (activation scales: separately with x_sf_gather)
        self.stage_bytes = (
            self.w_bytes
            + self.wsf_bytes
            + (0 if x_gather else self.a_bytes)
            + (0 if x_sf_gather else self.asf_bytes)
        )
        if self.token_major and act:
            # the epilogue stages a tile's FP4 bytes (8 KB) and block scales (1 KB): 9 KB, room for a 4th stage of
            # K 256 at 1-CTA; CTA pairs keep 16 KB
            self.scratch_bytes = 16384 if two_cta else 9216
        elif self.token_major:
            # the fused token-major down: the tile's BF16 rows, 16 bytes of padding apart
            self.row_stride = bn * 2 + 16
            self.scratch_bytes = bm * self.row_stride
        else:  # the weight-major epilogues need none (the SwiGLU pairs are in-thread: _epilogue_weight_major_act)
            self.scratch_bytes = 0
        # TMEM: accumulator, then weight scales, then activation scales.
        self.acc_cols = bn if self.token_major else bm
        # Weight-major epilogue: tcgen05.ld.32x32b.x{bm} below 64 tokens, else x64 chunks.
        self.ld_rep = min(bm, 64)
        self.wsf_cols = bn // 128 * 4 * (bk // 64)
        self.asf_cols = 4 * (bk // 64)
        used = self.acc_cols * self.acc_stages + self.wsf_cols + self.asf_cols
        self.tmem_cols = max(32, 1 << (used - 1).bit_length())
        # live_n (weight-major tiles of bm >= 32 tokens, 1-CTA and wm2): the MMAs of a tile take N = its live tokens
        # rounded up to 8 (a runtime instruction descriptor), not bm, as FlashInfer's cute_dsl swap-AB MoE does. The
        # padding columns are not computed (their accumulator columns keep stale values; the epilogues skip the
        # 8-token chunks past the tile's routes): less tensor work and fewer shared-memory operand reads, i.e. less
        # energy per call, which at the GPU's power limit is a higher clock (0.2-1.9% faster there in the padded
        # tiles, unchanged in short bursts). At bm 16 (short K loops, a few live tokens) the per-tile read of the
        # limit on the MMA warp's path costs more than it saves (up to 1% slower). wm2: the pair's N columns are
        # N / 2 rows of each CTA's tokens (bm / 2 each), so N shrinks only where the live tokens fit in the first
        # CTA's half: N = 2 x their count rounded up to 8 (the columns past N / 2 are the second CTA's padding rows;
        # columns below N / 2 stay tokens 0 .. N / 2 - 1), else bm (up to 1.5% faster in the tails of the experts).
        self.live_n = not token_major and bm >= 32 and (not two_cta or self.wm2)
        if (
            self.tmem_cols > 512
        ):  # an exception, not an assert: pick_tiles_token_major relies on it
            raise ValueError(
                f"TMEM: {used} columns of accumulators and scales exceed 512"
            )

    @staticmethod
    def n_tail_ok(act, activation, bn, token_major, two_cta):
        """n_tail: the token-major (bn 256), 1-CTA, non-gated gate/up only."""
        return (
            act
            and not is_gated(as_activation(activation))
            and bn == 256
            and token_major
            and not two_cta
        )

    def smem_bytes(self):
        """Shared memory of the kernel's Storage (the layout of ``__call__``: members in order, each at its alignment,
        the whole padded to its 1024-byte alignment); equals the compiled size."""
        s, a, r = self.stages, self.acc_stages, self.ring
        if self.token_major:
            members = [
                (1024, s * self.w_bytes),
                (1024, s * self.a_bytes),
                (1024, max(self.scratch_bytes, 16)),
                (1024, s * self.wsf_bytes),
                (1024, s * self.asf_bytes),
                (8, 8 * s),
                (8, 8 * s),
                (8, 8 * (s if self.x_sf_gather == 3 else 1)),
                (8, 8 * (s if self.x_gather else 1)),
                (8, 8 * a),
                (8, 8 * a),
                (8, 8),
                (8, 8 * r),
                (8, 8 * r),
                (4, 4 * r),
                (4, 4),
            ]
            off = 0
            for align, size in members:
                off = (off + align - 1) // align * align + size
            return (off + 1023) // 1024 * 1024
        members = [
            (1024, s * self.w_bytes),
            (1024, s * self.a_bytes),
            (1024, s * self.wsf_bytes),
            (1024, s * self.asf_bytes),
            (1024, max(self.scratch_bytes, 16)),
            (16, self.scales_bytes),
            (16, self.sfo_bytes),
            (8, 8 * s),
            (8, 8 * s),
            (8, 8 * (s if self.x_sf_gather == 3 else 1)),
            (8, 8 * s if self.x_gather else 0),
            (8, 8 * a),
            (8, 8 * a),
            (8, 8 * self.dealloc_n),
            (8, 8 * r),
            (8, 8 * r),
            (4, 4 * r),
            (4, 4),
            (4, 4 * self.place_words),
        ]
        off = 0
        for align, size in members:
            off = (off + align - 1) // align * align + size
        return (off + 1023) // 1024 * 1024

    @staticmethod
    def fit_stages(n, k, bm, bk, act, capacity, **kw):
        """The most K stages (bk each; at most 16 and 4 tiles' K loops) whose weight-major kernel fits ``capacity``
        bytes of shared memory."""
        best = 1
        for s in range(2, min(16, 4 * (-(-k // bk))) + 1):
            g = GroupedGemm(n, k, bm, 128, bk, s, act, **kw)
            if g.smem_bytes() <= capacity:
                best = s
        return best

    @staticmethod
    @functools.cache  # (a search over GroupedGemm configurations)
    def pick_tiles(
        n, k, bm, act, tile_stages=False, *, sm_smem, cta_smem, k256=False, **kw
    ):
        """(bk, stages, acc_stages) of a weight-major grouped GEMM from the shared memory alone:
        - two CTAs per SM where half of an SM's shared memory holds a whole tile's K loop in at least two stages, K 512
          first; the stages fill
          that half (``tile_stages``: just the tile's K loop, at least two);
        - else one CTA per SM with as many stages as fit (K 512 wherever it divides K: 256-byte row pieces per box);
        - two TMEM accumulators where the CTAs' TMEM allows (the epilogue of a tile overlaps the next tile's MMAs);
        - ``k256``: K 256 with as many stages as fit one CTA per SM (``moe``: long batches' down GEMMs with many
          routes per expert, with ``k_tail`` (less of the last K tile past K: K 2688 pads to 2816 instead of 3072)
          or as CTA pairs with a long K loop).
        ``sm_smem`` / ``cta_smem``: the GPU's shared memory per SM and the most a CTA can opt in to (``CuteMoE``
        reads them from the device)."""
        half = sm_smem // 2 - 1024
        tail = kw.get(
            "k_tail", False
        )  # K tiles past K allowed (k_tail): K 512 / 256 for any K % 128 == 0
        if k256:
            assert k % 256 == 0 or tail, "k256: K tiles past K need k_tail"
            return (
                256,
                GroupedGemm.fit_stages(
                    n, k, bm, 256, act, capacity=cta_smem, acc_stages=2, **kw
                ),
                2,
            )
        for bk in (
            (512, 256) if k % 256 == 0 or tail else (128,)
        ):  # K tile 128 only where 256 does not divide K
            if k % bk and not tail:
                continue
            g = GroupedGemm(n, k, bm, 128, bk, 2, act, **kw)
            acc = (
                2
                if 2
                * max(
                    32, 1 << (2 * g.acc_cols + g.wsf_cols + g.asf_cols - 1).bit_length()
                )
                <= 512
                else 1
            )
            st = GroupedGemm.fit_stages(
                n, k, bm, bk, act, capacity=half, acc_stages=acc, **kw
            )
            if st >= 2 and st * bk >= k:
                return bk, (max(2, -(-k // bk)) if tile_stages else st), acc
        bk = 512 if (k % 512 == 0 or tail) else 256 if k % 256 == 0 else 128
        return (
            bk,
            GroupedGemm.fit_stages(
                n, k, bm, bk, act, capacity=cta_smem, acc_stages=2, **kw
            ),
            2,
        )

    @staticmethod
    @functools.cache
    def pick_tiles_token_major(n, k, *, cta_smem, **kw):
        """(bn, bk, stages, acc_stages) of the long batches' token-major down GEMM (fused, or unfused as CTA pairs; one
        CTA per SM; sized with fused_out, whose shared memory the unfused pairs share): bn 256 as CTA pairs
        (``two_cta``: MMAs of 256 x 256) or where the K loop is short (k = I <= 1024), else 128; K 512 where at least 3
        stages fit, else K 256; as many stages as the shared memory holds (``cta_smem``: the most a CTA can opt in
        to); two TMEM accumulators where they fit."""
        bn = 256 if kw.get("two_cta") or k <= 1024 else 128
        cands = []
        tail = kw.get("k_tail", False)
        for bk in (
            (512, 256) if k % 256 == 0 or tail else (128,)
        ):  # K tile 128 only where 256 does not divide K
            if k % bk and not tail:
                continue
            for acc in (2, 1):
                best = 0
                for st in range(2, 17):
                    try:
                        g = GroupedGemm(
                            n,
                            k,
                            128,
                            bn,
                            bk,
                            st,
                            False,
                            token_major=True,
                            acc_stages=acc,
                            fused_out=True,
                            **kw,
                        )
                    except ValueError:  # the TMEM does not hold the accumulators
                        break
                    if g.smem_bytes() <= cta_smem:
                        best = st
                if best:
                    cands.append((bk, best, acc))
        bk, st, acc = next(
            (c for c in cands if c[1] >= 3), max(cands, key=lambda c: c[1])
        )
        return bn, bk, st, acc

    def _setup(self):
        mn = self.mma_mn
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            mn,
        )
        self.mma_sfb_mn = (
            mn[0],
            cute.round_up(self.sfb_rows, 128),
        )
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_sfb_mn,
        )
        self.mma_tiler = (mn[0], mn[1], self.bk)
        self.mma_tiler_sfb = (self.mma_sfb_mn[0], self.mma_sfb_mn[1], self.bk)
        self.a_smem = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.ab_dtype, self.stages
        )
        self.b_smem = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.ab_dtype, self.stages
        )
        self.sfa_smem = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, SF_VEC, self.stages
        )
        if self.two_cta:  # the weight scales of all bn rows in each CTA
            self.sfb_smem = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC, self.stages
            )
        else:
            self.sfb_smem = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma, self.mma_tiler, SF_VEC, self.stages
            )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,  # activations FP4 [rows, K]
        sfa_ptr: cute.Pointer,  # activation block scales: packed per tile, or rows (x_sf_gather)
        w_ptr: cute.Pointer,  # weights FP4 [E * N, K]
        sfw_ptr: cute.Pointer,  # weight block scales, 128x4-swizzled
        a_gs_ptr: cute.Pointer,  # FP32 [rows]
        w_gs_ptr: cute.Pointer,  # FP32 [E]
        experts_ptr: cute.Pointer,  # Int32 [tiles]
        count_ptr: cute.Pointer,  # Int32 [1]: rows in use
        limits_ptr: cute.Pointer,  # Int32 [tiles]: end of the tile's expert rows
        out_ptr: cute.Pointer,  # act: FP4 bytes [rows, N / 4]; else BF16 [rows, N]
        osf_ptr: cute.Pointer,  # act: the output's block scales (tile-packed); else unused
        tmap_ptr: cute.Pointer,  # UInt8 [grid_ctas * 512], 128-aligned: per-CTA TMA descriptor copies (token-major)
        claims_ptr: cute.Pointer,  # Int32 [1], zero: tiles claimed (claim_tiles)
        rowtok_ptr: cute.Pointer,  # Int32 [rows]: the token of each row (x_sf_gather, fused_out)
        clr_ptr: cute.Pointer,  # Int32 [clear]: zeroed after the wait (clear)
        tb_ptr: cute.Pointer,  # Int32 [tiles]: route_rows' first eroute index of each tile
        inv_ptr: cute.Pointer,  # Int32 [routes]: route_rows' gate/up writes each route's row (ReduceRoutes)
        rows: Int32,
        num_experts: Int32,
        grid_ctas: Int32,
        stream,
    ):
        self._setup()
        k, n = self.k, self.n
        tiles = rows // self.bm
        act_t = cute.make_tensor(
            a_ptr, cute.make_layout((rows, k, 1), stride=(k, 1, rows * k))
        )
        wrows = num_experts * n
        if cutlass.const_expr(
            self.n_tail
        ):  # per expert: (n, k, E), the rows past n of a tile are the TMA's zeros
            w_t = cute.make_tensor(
                w_ptr, cute.make_layout((n, k, num_experts), stride=(k, 1, n * k))
            )
        else:
            w_t = cute.make_tensor(
                w_ptr, cute.make_layout((wrows, k, 1), stride=(k, 1, wrows * k))
            )
        w_sf = cute.make_tensor(
            sfw_ptr, blockscaled_utils.tile_atom_to_shape_SF(w_t.shape, SF_VEC)
        )
        act_sf = cute.make_tensor(
            sfa_ptr,
            blockscaled_utils.tile_atom_to_shape_SF((tiles * 128, k, 1), SF_VEC),
        )
        if cutlass.const_expr(self.token_major):
            mma_a, mma_sfa, mma_b, mma_sfb = act_t, act_sf, w_t, w_sf
        else:
            mma_a, mma_sfa, mma_b, mma_sfb = w_t, w_sf, act_t, act_sf
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        cluster_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma_sfb.thr_id.shape,)
        )
        a_one = cute.slice_(self.a_smem, (None, None, None, 0))
        b_one = cute.slice_(self.b_smem, (None, None, None, 0))
        sfa_one = cute.slice_(self.sfa_smem, (None, None, None, 0))
        sfb_one = cute.slice_(self.sfb_smem, (None, None, None, 0))
        op_a = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        op_b = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), self.tiled_mma.thr_id)
        op_sfb = sm100_utils.cluster_shape_to_tma_atom_SFB(
            (1, 1), self.tiled_mma.thr_id
        )
        if cutlass.const_expr(self.two_cta):
            op_a = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
            op_b = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
            op_sfb = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        tma_a, tma_a_t = cute.nvgpu.make_tiled_tma_atom_A(
            op_a, mma_a, a_one, self.mma_tiler, self.tiled_mma, cluster_vmnk.shape
        )
        tma_b, tma_b_t = cute.nvgpu.make_tiled_tma_atom_B(
            op_b, mma_b, b_one, self.mma_tiler, self.tiled_mma, cluster_vmnk.shape
        )
        if cutlass.const_expr(self.x_sf_gather == 3):
            # x_sf_gather 3: the activation scales' rows [rows, K / 16], a bm x bk / 16 box per stage (token-major: the
            # MMA's A scales; weight-major: its B scales)
            srows = cute.make_tensor(
                sfa_ptr, cute.make_layout((rows, k // SF_VEC), stride=(k // SF_VEC, 1))
            )
            tma_rows = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                srows,
                cute.make_layout(
                    (self.bm, self.bk // SF_VEC), stride=(self.bk // SF_VEC, 1)
                ),
                (self.bm, self.bk // SF_VEC),
            )
        if cutlass.const_expr(self.token_major and self.x_sf_gather == 3):
            tma_sfa, tma_sfa_t = tma_rows
        else:
            tma_sfa, tma_sfa_t = cute.nvgpu.make_tiled_tma_atom_A(
                op_a,
                mma_sfa,
                sfa_one,
                self.mma_tiler,
                self.tiled_mma,
                cluster_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        if cutlass.const_expr(self.x_sf_gather == 3 and not self.token_major):
            tma_sfb, tma_sfb_t = tma_rows
        else:
            tma_sfb, tma_sfb_t = cute.nvgpu.make_tiled_tma_atom_B(
                op_sfb,
                mma_sfb,
                sfb_one,
                self.mma_tiler_sfb,
                self.tiled_mma_sfb,
                cluster_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        # Weight and activation smem/scale layouts by role.
        if cutlass.const_expr(self.token_major):
            w_smem, x_smem, wsf_smem, xsf_smem = (
                self.b_smem,
                self.a_smem,
                self.sfb_smem,
                self.sfa_smem,
            )
        else:
            w_smem, x_smem, wsf_smem, xsf_smem = (
                self.a_smem,
                self.b_smem,
                self.sfa_smem,
                self.sfb_smem,
            )
        stages = self.stages
        scratch = max(self.scratch_bytes, 16)
        ring = self.ring  # empty without claim_tiles: the same layout

        # Allocation order: operands, [scratch], scales[, scratch], barriers.
        if cutlass.const_expr(self.token_major):

            @cute.struct
            class Storage:
                sW: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(w_smem.outer)], 1024
                ]
                sX: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(x_smem.outer)], 1024
                ]
                sScratch: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, scratch], 1024
                ]
                sWSF: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(wsf_smem)], 1024
                ]
                sXSF: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(xsf_smem)], 1024
                ]
                stage_free: cute.struct.MemRange[cutlass.Int64, stages]
                stage_ready: cute.struct.MemRange[cutlass.Int64, stages]
                sf_full: cute.struct.MemRange[
                    cutlass.Int64, stages if self.x_sf_gather == 3 else 1
                ]
                a_full: cute.struct.MemRange[
                    cutlass.Int64, stages if self.x_gather else 1
                ]
                acc_free: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
                acc_ready: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
                tmem_dealloc: cute.struct.MemRange[cutlass.Int64, 1]
                tile_full: cute.struct.MemRange[cutlass.Int64, ring]
                tile_empty: cute.struct.MemRange[cutlass.Int64, ring]
                sTiles: cute.struct.MemRange[cutlass.Int32, ring]
                tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
        else:

            @cute.struct
            class Storage:  # type: ignore[no-redef]  # the other const_expr branch's layout
                sW: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(w_smem.outer)], 1024
                ]
                sX: cute.struct.Align[
                    cute.struct.MemRange[self.ab_dtype, cute.cosize(x_smem.outer)], 1024
                ]
                sWSF: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(wsf_smem)], 1024
                ]
                sXSF: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(xsf_smem)], 1024
                ]
                sScratch: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, scratch], 1024
                ]
                sScales: cute.struct.Align[
                    cute.struct.MemRange[Float32, self.scales_bytes // 4], 16
                ]
                sSFO: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, self.sfo_bytes], 16
                ]
                stage_free: cute.struct.MemRange[cutlass.Int64, stages]
                stage_ready: cute.struct.MemRange[cutlass.Int64, stages]
                sf_full: cute.struct.MemRange[
                    cutlass.Int64, stages if self.x_sf_gather == 3 else 1
                ]
                a_full: cute.struct.MemRange[
                    cutlass.Int64, stages if self.x_gather else 0
                ]
                acc_free: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
                acc_ready: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
                tmem_dealloc: cute.struct.MemRange[cutlass.Int64, self.dealloc_n]
                tile_full: cute.struct.MemRange[cutlass.Int64, ring]
                tile_empty: cute.struct.MemRange[cutlass.Int64, ring]
                sTiles: cute.struct.MemRange[cutlass.Int32, ring]
                tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
                sPlace: cute.struct.MemRange[cutlass.Int32, self.place_words]

        self.storage = Storage
        a_gs = cute.make_tensor(a_gs_ptr, cute.make_layout(rows))
        w_gs = cute.make_tensor(w_gs_ptr, cute.make_layout(num_experts))
        experts = cute.make_tensor(experts_ptr, cute.make_layout(tiles))
        limits = cute.make_tensor(limits_ptr, cute.make_layout(tiles))
        count = cute.make_tensor(count_ptr, cute.make_layout(1))
        out_cols = self.out_cols
        out = cute.make_tensor(
            out_ptr, cute.make_layout((rows, out_cols), stride=(out_cols, 1))
        )
        launch_regs(
            self.kernel(
                self.tiled_mma,
                self.tiled_mma_sfb,
                tma_a,
                tma_a_t,
                tma_b,
                tma_b_t,
                tma_sfa,
                tma_sfa_t,
                tma_sfb,
                tma_sfb_t,
                self.a_smem,
                self.b_smem,
                self.sfa_smem,
                self.sfb_smem,
                a_gs,
                w_gs,
                experts,
                count,
                limits,
                out,
                tmap_ptr,
                w_ptr,
                sfw_ptr,
                claims_ptr,
                sfa_ptr,
                rowtok_ptr,
                osf_ptr,
                clr_ptr,
                a_ptr,
                tb_ptr,
                inv_ptr,
            ).launch,
            self.MAXNREG,
            grid=[grid_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=use_pdl(self),
            **({"cluster": [2, 1, 1]} if self.two_cta else {}),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_sfb: cute.CopyAtom,
        mSFB: cute.Tensor,
        a_smem: cute.ComposedLayout,
        b_smem: cute.ComposedLayout,
        sfa_smem: cute.Layout,
        sfb_smem: cute.Layout,
        a_gs: cute.Tensor,
        w_gs: cute.Tensor,
        experts: cute.Tensor,
        count: cute.Tensor,
        limits: cute.Tensor,
        out: cute.Tensor,
        tmap: cute.Pointer,
        w_raw: cute.Pointer,
        wsf_raw: cute.Pointer,
        claims: cute.Pointer,
        xsf_raw: cute.Pointer,
        rowtok: cute.Pointer,
        osf: cute.Pointer,
        clr: cute.Pointer,
        xq_raw: cute.Pointer,
        tbp: cute.Pointer,
        invp: cute.Pointer,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()

        smem = cutlass.memory.SmemAllocator()
        st = smem.allocate(self.storage)
        sexp, slim, stb, gexp, glim, gcount = (
            experts,
            limits,
            None,
            experts,
            limits,
            count,
        )
        if cutlass.const_expr(self.place_tiles):
            # the kernel reads the CTA's own table (shared memory); the load warp writes it (and CTA 0 the global one)
            MT = self.max_tiles
            pb = st.sPlace.data_ptr()
            sexp = cute.make_tensor(pb, cute.make_layout(MT))
            slim = cute.make_tensor(pb + MT, cute.make_layout(MT))
            stb = cute.make_tensor(pb + 2 * MT, cute.make_layout(MT))
            count = cute.make_tensor(pb + 3 * MT, cute.make_layout(1))
            experts, limits = sexp, slim
        if cutlass.const_expr(self.place_tiles == 1):
            tbt = stb  # the gather warps and the epilogue read the table's tbase
        else:
            tbt = cute.make_tensor(
                cute.make_ptr(
                    Int32, tbp.toint(), cute.AddressSpace.gmem, assumed_align=4
                ),
                cute.make_layout(1 << 30),
            )
        stage_free = st.stage_free.data_ptr()
        stage_ready = st.stage_ready.data_ptr()
        acc_free = st.acc_free.data_ptr()
        acc_ready = st.acc_ready.data_ptr()
        holding = st.tmem_holding.data_ptr()
        # Each side runs from its setmaxnreg to the end without merging, so ptxas allocates each with
        # its own register budget. CTA-wide barriers use id 1; id 0 belongs to the epilogue warps.
        TWO = self.two_cta
        rank = (
            bidx % 2 if TWO else 0
        )  # CTA rank in the pair (cluster [2, 1, 1]: blocks 2 c, 2 c + 1)
        if warp < self.D:
            cute.arch.setmaxregister_increase(self.regs_epi)
            if warp == 0:
                if cutlass.const_expr(TWO):
                    cute.arch.alloc_tmem(self.tmem_cols, holding, is_two_cta=True)
                else:
                    cute.arch.alloc_tmem(self.tmem_cols, holding)
            if cutlass.const_expr(
                self.launch_pdl and self.act and not self.place_tiles
            ):
                # Dependents read only weights and routing before waiting.
                cute.arch.griddepcontrol_launch_dependents()
            if warp == 0:
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(stage_free + s, 1)
                        # The TMA loads, and the scale warps with x_sf_gather; two_cta (the leader's): its load
                        # warp's expect_tx and both CTAs' convert warps.
                        cute.arch.mbarrier_init(stage_ready + s, self.ready_count)
                        if cutlass.const_expr(self.x_sf_gather == 3):
                            cute.arch.mbarrier_init(st.sf_full.data_ptr() + s, 1)
                        if cutlass.const_expr(
                            self.x_gather
                        ):  # one cp.async completion arrival per gather thread
                            cute.arch.mbarrier_init(
                                st.a_full.data_ptr() + s, 32 * self.G
                            )
                    for a in cutlass.range_constexpr(self.acc_stages):
                        cute.arch.mbarrier_init(
                            acc_free + a, 2 if TWO else 1
                        )  # two_cta: both CTAs' epilogues
                        cute.arch.mbarrier_init(acc_ready + a, 1)
                    if cutlass.const_expr(TWO):
                        cute.arch.mbarrier_init(st.tmem_dealloc.data_ptr(), 1)
                    for s in cutlass.range_constexpr(self.ring):
                        # the load warp puts a tile id; the MMA warp and the epilogue have read it
                        cute.arch.mbarrier_init(st.tile_full.data_ptr() + s, 1)
                        cute.arch.mbarrier_init(
                            st.tile_empty.data_ptr() + s,
                            (4 if self.x_sf_gather else 2)
                            + (1 if self.x_gather and not self.token_major else 0),
                        )
                cute.arch.mbarrier_init_fence()
            bar_sync_cta(1)
            if cutlass.const_expr(
                TWO
            ):  # the peer's barriers are initialized before any remote arrive
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            tmem_base = st.tmem_holding.get_tensor(cute.make_layout(1))[0]
            if warp == 0:
                if cutlass.const_expr(TWO):
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
                else:
                    cute.arch.relinquish_tmem_alloc_permit()
            if cutlass.const_expr(self.place_tiles):
                bar_sync_cta(
                    1
                )  # the load warp has placed the tiles (and CTA 0 published them)
                cute.arch.griddepcontrol_launch_dependents()
            num_tiles = (count[0] // self.pair_rows) * self.tnt
            if cutlass.const_expr(self.launch_pdl and not self.token_major):
                # The weight-major epilogues prefetch row scales before the accumulator wait: the first
                # tile's must not be fetched before the primary grid is done.
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.clear > 0):
                if bidx == 0:
                    for i in cutlass.range_constexpr(
                        (self.clear + self.epi_threads - 1) // self.epi_threads
                    ):
                        if tidx + i * self.epi_threads < self.clear:
                            st_global(
                                clr.toint() + Int64(tidx + i * self.epi_threads) * 4,
                                0,
                                Int32,
                                4,
                            )
            # the five epilogues' common arguments (TWO: the pair's tile index and grid; only the weight-major down
            # and the token-major epilogues run as CTA pairs)
            ea = (
                st,
                tmem_base,
                warp,
                tidx,
                bidx // 2 if TWO else bidx,
                grid // 2 if TWO else grid,
                num_tiles,
                a_gs,
                w_gs,
                experts,
                limits,
                out,
                acc_free,
                acc_ready,
            )
            if cutlass.const_expr(self.token_major and not self.act):
                self._epilogue_token_major_fused(*ea, rowtok, rank, tbt)
            elif cutlass.const_expr(self.token_major):
                self._epilogue_token_major(*ea, osf, rank)
            elif cutlass.const_expr(self.act):
                self._epilogue_weight_major_act(*ea, osf, rowtok, tbt)
            elif cutlass.const_expr(self.fused_out):
                self._epilogue_fused(*ea, rowtok, tbt)
            else:
                self._epilogue_weight_major(*ea, rank)
            bar_sync_cta(1)
            if warp == 0:
                if cutlass.const_expr(TWO):
                    # both CTAs are done with the pair's TMEM before either deallocates it
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(
                            st.tmem_dealloc.data_ptr(),
                            rank ^ 1,
                            scope=nvvm.MemScopeKind.CLUSTER,
                        )
                    cute.arch.mbarrier_wait(st.tmem_dealloc.data_ptr(), 0)
                    cute.arch.dealloc_tmem(
                        cute.arch.retrieve_tmem_ptr(Float32, 16, holding),
                        self.tmem_cols,
                        is_two_cta=True,
                    )
                else:
                    cute.arch.dealloc_tmem(
                        cute.arch.retrieve_tmem_ptr(Float32, 16, holding),
                        self.tmem_cols,
                    )
        else:
            cute.arch.setmaxregister_decrease(self.regs_worker)
            bar_sync_cta(1)
            if cutlass.const_expr(TWO):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            if cutlass.const_expr(self.place_tiles):
                if warp == self.role_warp["load"]:
                    if cutlass.const_expr(self.place_tiles == 1):
                        cute.arch.griddepcontrol_wait()
                    self._place_tiles(
                        clr, sexp, slim, stb, count, gexp, glim, tbp, gcount, bidx
                    )
                bar_sync_cta(1)
            num_tiles = (count[0] // self.pair_rows) * self.tnt
            if warp == self.role_warp["mma"]:
                if cutlass.const_expr(TWO):
                    if rank == 0:  # the leader issues the pair's MMAs
                        tmem_base = st.tmem_holding.get_tensor(cute.make_layout(1))[0]
                        self._mma_warp_raw(
                            st,
                            tmem_base,
                            bidx // 2,
                            grid // 2,
                            num_tiles,
                            stage_free,
                            stage_ready,
                            acc_free,
                            acc_ready,
                            limits,
                        )
                else:
                    tmem_base = st.tmem_holding.get_tensor(cute.make_layout(1))[0]
                    self._mma_warp_raw(
                        st,
                        tmem_base,
                        bidx,
                        grid,
                        num_tiles,
                        stage_free,
                        stage_ready,
                        acc_free,
                        acc_ready,
                        limits,
                    )
            if warp == self.role_warp["load"]:
                self._load_warp(
                    st,
                    tiled_mma,
                    tiled_mma_sfb,
                    tma_a,
                    mA,
                    tma_b,
                    mB,
                    tma_sfa,
                    mSFA,
                    tma_sfb,
                    mSFB,
                    a_smem,
                    b_smem,
                    sfa_smem,
                    sfb_smem,
                    bidx // 2 if TWO else bidx,
                    grid // 2 if TWO else grid,
                    num_tiles,
                    experts,
                    stage_free,
                    stage_ready,
                    tmap.toint() + Int64(bidx) * 512,
                    w_raw,
                    wsf_raw,
                    claims,
                    rank,
                    limits,
                    tbt,
                    invp,
                )
            if cutlass.const_expr(self.x_gather and self.token_major):
                if warp >= self.D + 4:
                    self._gather_warps(
                        st,
                        tidx,
                        rowtok,
                        xq_raw,
                        limits,
                        bidx // 2 if TWO else bidx,
                        grid // 2 if TWO else grid,
                        num_tiles,
                        stage_free,
                        rank,
                    )
            if cutlass.const_expr(self.x_gather and not self.token_major):
                if warp >= self.D + 4:
                    self._gather_warps_wm(
                        st,
                        tidx,
                        rowtok,
                        xq_raw,
                        xsf_raw,
                        tbt,
                        invp,
                        limits,
                        bidx,
                        grid,
                        num_tiles,
                        stage_free,
                    )
            if cutlass.const_expr(self.x_sf_gather):
                if cutlass.const_expr(self.x_sf_gather == 2):
                    if warp == self.role_warp["sfg0"]:
                        self._sf_fence_warp(st, bidx, grid, num_tiles, stage_ready, 0)
                    if warp == self.role_warp["sfg1"]:
                        self._sf_fence_warp(st, bidx, grid, num_tiles, stage_ready, 1)
                elif cutlass.const_expr(self.x_sf_gather == 3):
                    if warp == self.role_warp["sfg0"]:
                        self._sf_convert_warp(
                            st,
                            bidx // 2 if TWO else bidx,
                            grid // 2 if TWO else grid,
                            num_tiles,
                            stage_ready,
                            0,
                        )
                    if warp == self.role_warp["sfg1"]:
                        self._sf_convert_warp(
                            st,
                            bidx // 2 if TWO else bidx,
                            grid // 2 if TWO else grid,
                            num_tiles,
                            stage_ready,
                            1,
                        )
                else:
                    if warp == self.role_warp["sfg0"]:
                        self._sf_warp(
                            st,
                            xsf_raw,
                            rowtok,
                            limits,
                            bidx,
                            grid,
                            num_tiles,
                            stage_free,
                            stage_ready,
                            0,
                        )
                    if warp == self.role_warp["sfg1"]:
                        self._sf_warp(
                            st,
                            xsf_raw,
                            rowtok,
                            limits,
                            bidx,
                            grid,
                            num_tiles,
                            stage_free,
                            stage_ready,
                            1,
                        )
            bar_sync_cta(1)

    # ---------------------------------------------------------------------------------- TMA descriptors
    @cute.jit
    def _l2_256_desc(self, atom, addr):
        """This warp's copy of ``atom``'s TMA descriptor at global ``addr`` (128 B) with the L2 promotion
        field (byte 10, bits 1-2) set to 256 B. CuTe encodes its descriptors with 128 B promotion and has
        no option for it. Returns the pointer for ``cute.copy(tma_desc_ptr=)``."""
        raw = cute.make_ptr(Uint8, addr, cute.AddressSpace.gmem, assumed_align=128)
        # the copy takes a global pointer, the TMA intrinsics a generic one
        p = _TMAPS.get_tensormap_ptr(raw, cute.AddressSpace.generic)
        with cute.arch.elect_one():
            cpasync.copy_tensormap(
                atom, _TMAPS.get_tensormap_ptr(raw, cute.AddressSpace.gmem)
            )
            b = cute.make_tensor(
                cute.make_ptr(
                    Uint8, addr + 10, cute.AddressSpace.gmem, assumed_align=1
                ),
                cute.make_layout(1),
            )
            b[0] = (b[0] & Uint8(0xF9)) | Uint8(0x06)
            cpasync.fence_tma_desc_release()
        cute.arch.sync_warp()
        cpasync.fence_tma_desc_acquire(p)
        cute.arch.prefetch(
            cute.make_ptr(Int64, addr, cute.AddressSpace.generic, assumed_align=128),
            tensormap=True,
        )
        return p

    # ---------------------------------------------------------------------------------- load
    @cute.jit
    def _weight_policy(self, mb, nrb, experts):
        """L2 policy of row block mb's weight loads (evict_weights): EVICT_FIRST unless the next of the nrb
        row blocks belongs to the same expert."""
        pol = Int64(EVICT_FIRST)
        if mb + 1 < nrb:
            if experts[mb + 1] == experts[mb]:
                pol = Int64(EVICT_NORMAL)
        return pol

    @cute.jit
    def _load_warp(
        self,
        st,
        tiled_mma,
        tiled_mma_sfb,
        tma_a,
        mA,
        tma_b,
        mB,
        tma_sfa,
        mSFA,
        tma_sfb,
        mSFB,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        bidx,
        grid,
        num_tiles,
        experts,
        stage_free,
        stage_ready,
        tmap_base,
        w_raw,
        wsf_raw,
        claims,
        rank=0,
        limits=None,
        ptab=None,
        zero_ptr=None,
    ):
        NT = self.nt
        KT = self.kt
        if cutlass.const_expr(self.token_major):
            sA = st.sX.get_tensor(a_smem.outer, swizzle=a_smem.inner)
            sB = st.sW.get_tensor(b_smem.outer, swizzle=b_smem.inner)
            sSFA = st.sXSF.get_tensor(sfa_smem)
            sSFB = st.sWSF.get_tensor(sfb_smem)
        else:
            sA = st.sW.get_tensor(a_smem.outer, swizzle=a_smem.inner)
            sB = st.sX.get_tensor(b_smem.outer, swizzle=b_smem.inner)
            sSFA = st.sWSF.get_tensor(sfa_smem)
            sSFB = st.sXSF.get_tensor(sfb_smem)
        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        one = cute.make_layout(1)
        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            0,
            one,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b,
            0,
            one,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        if cutlass.const_expr(self.x_sf_gather == 3):
            # the stage's scale block holds the bm x bk / 16 row-major box until the scale warps rearrange it
            sStg = cute.make_tensor(
                cute.make_ptr(
                    self.sf_dtype,
                    st.sXSF.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                cute.make_layout(
                    (self.bm, self.bk // 16, self.stages),
                    stride=(self.bk // 16, 1, self.asf_bytes),
                ),
            )
        if cutlass.const_expr(self.x_sf_gather == 3 and self.token_major):
            gS = cute.local_tile(mSFA, (self.bm, self.bk // 16), (None, None))
            tSs3, tSg3 = cpasync.tma_partition(
                tma_sfa,
                0,
                one,
                cute.group_modes(sStg, 0, 2),
                cute.group_modes(gS, 0, 2),
            )
        if cutlass.const_expr(self.x_sf_gather == 3 and not self.token_major):
            gS = cute.local_tile(mSFB, (self.bm, self.bk // 16), (None, None))
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_sfb,
                0,
                one,
                cute.group_modes(sStg, 0, 2),
                cute.group_modes(gS, 0, 2),
            )
        else:
            gSFB = cute.local_tile(
                mSFB,
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_sfb,
                0,
                one,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(thr_mma_sfb.partition_B(gSFB), 0, 3),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
        tAsSFA = tAsA
        tAgSFA = tAgA
        if cutlass.const_expr(self.token_major and self.x_sf_gather == 3):
            tAsSFA, tAgSFA = tSs3, tSg3
        else:
            gSFA = cute.local_tile(
                mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_sfa,
                0,
                one,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gSFA), 0, 3),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
        # Weight operand and its scales by role (token-major: B; weight-major: A).
        if cutlass.const_expr(self.token_major):
            tma_w, tWs, tWg, tma_wsf, tWSFs, tWSFg = (
                tma_b,
                tBsB,
                tBgB,
                tma_sfb,
                tBsSFB,
                tBgSFB,
            )
            tma_x, tXs, tXg, tma_xsf, tXSFs, tXSFg = (
                tma_a,
                tAsA,
                tAgA,
                tma_sfa,
                tAsSFA,
                tAgSFA,
            )
        else:
            tma_w, tWs, tWg, tma_wsf, tWSFs, tWSFg = (
                tma_a,
                tAsA,
                tAgA,
                tma_sfa,
                tAsSFA,
                tAgSFA,
            )
            tma_x, tXs, tXg, tma_xsf, tXSFs, tXSFg = (
                tma_b,
                tBsB,
                tBgB,
                tma_sfb,
                tBsSFB,
                tBgSFB,
            )

        dw, dx = {}, {}
        if cutlass.const_expr(self.token_major):
            # Operands only; the scales' loads use the kernel-parameter descriptors.
            dw = {"tma_desc_ptr": self._l2_256_desc(tma_w, tmap_base)}
            if cutlass.const_expr(not self.x_gather):
                dx = {"tma_desc_ptr": self._l2_256_desc(tma_x, tmap_base + 128)}
        TNT = self.tnt
        nrb = num_tiles // TNT

        # Weights do not depend on the producer, so with PDL fill every stage before waiting (dynamic: those of the
        # two round-robin tiles; later tiles are claimed after the wait).
        PREFETCH = (
            (min(self.stages, 2 * KT) if self.dynamic else self.stages)
            if self.launch_pdl
            else 0
        )
        for i in cutlass.range_constexpr(PREFETCH):
            start = bidx + (i // KT) * grid
            if start < num_tiles:
                wtile = experts[start // NT] * NT + start % NT
                bar = stage_ready + i
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar, self.stage_bytes)
                wp = {}  # the weights' L2 policy (evict_weights)
                if cutlass.const_expr(self.evict_weights):
                    wp = dict(
                        cache_policy=self._weight_policy(start // NT, nrb, experts)
                    )
                cute.copy(
                    tma_w,
                    tWg[(None, wtile, i % KT, 0)],
                    tWs[(None, i)],
                    tma_bar_ptr=bar,
                    **wp,
                    **dw,
                )
                cute.copy(
                    tma_wsf,
                    tWSFg[(None, wtile, i % KT, 0)],
                    tWSFs[(None, i)],
                    tma_bar_ptr=bar,
                    **wp,
                )
        if cutlass.const_expr(PREFETCH > 0 and self.l2_tiles > 0):
            # lane l: tile 1 + l // PER, part l % PER (PER - 1 data chunks of CH bytes, then the scales)
            BNK = self.bn * (self.k // 2)
            SFB = self.bn * (self.k // 16)
            CH = 16384 if BNK % 16384 == 0 else BNK
            ND = BNK // CH
            PER = ND + 1
            lane = cute.arch.lane_idx()
            ti = lane // PER
            part = lane % PER
            nxt = bidx + (ti + 1) * grid
            if ti < self.l2_tiles and nxt < num_tiles:
                wt = experts[nxt // NT] * NT + nxt % NT
                if part < ND:
                    l2_prefetch(w_raw.toint() + Int64(wt) * BNK + Int64(part) * CH, CH)
                else:
                    l2_prefetch(wsf_raw.toint() + Int64(wt) * SFB, SFB)
        if cutlass.const_expr(PREFETCH > 0):
            cute.arch.griddepcontrol_wait()
        seq = Int32(0)
        slot = Int32(0)
        phase = Int32(0)
        tile_idx = Int32(0)
        tile = bidx
        if cutlass.const_expr(self.skip_empty):
            tile = self._skip(tile, grid, num_tiles, limits)
        if cutlass.const_expr(self.dynamic):
            claiming = self._claiming(num_tiles, grid)
        while tile < num_tiles:
            if cutlass.const_expr(self.dynamic):
                # from the CTA's second tile on, claim the next now; its value is used after this tile's stages
                claim = Int32(0)
                if (cute.arch.lane_idx() == 0) & (tile >= grid) & claiming:
                    claim = cute.arch.atomic_add(
                        claims, Int32(1), sem="relaxed", scope="gpu"
                    )
            mb = tile // TNT
            wtile = experts[mb] * NT + tile % NT
            wtile_b = wtile
            xb = mb
            if cutlass.const_expr(self.wm2):
                # pair tile q: row block q // TNT (both CTAs), this CTA's 128 weight rows (N tile 2 (q % TNT) + rank),
                # its bm / 2 tokens (B tile 2 mb + rank), the activation scales of all bm tokens (mb)
                wtile = experts[mb] * NT + 2 * (tile % TNT) + rank
                wtile_b = wtile
                xb = 2 * mb + rank
            elif cutlass.const_expr(self.two_cta):
                # this CTA's 128 tokens of the pair tile and its half of the bn weight rows (128-row tiles); the
                # weight scales of all bn rows (pair_table: the tokens of block b0 + rank, b0's expert's weights)
                mb = 2 * (tile // NT) + rank
                if cutlass.const_expr(self.pair_table):
                    b0 = ptab[tile // NT]
                    mb = b0 + rank
                    wtile = experts[b0] * NT + tile % NT
                else:
                    wtile = experts[mb] * NT + tile % NT
                wtile_b = 2 * wtile + rank
                xb = mb
            pol = Int64(EVICT_NORMAL)
            if cutlass.const_expr(self.evict_weights):
                pol = self._weight_policy(mb, nrb, experts)
            for kt in cutlass.range(KT, unroll=1):
                bar = stage_ready + slot
                if seq >= PREFETCH:
                    cute.arch.mbarrier_wait(stage_free + slot, phase ^ 1)
                    if cutlass.const_expr(self.two_cta):
                        # the leader expects both CTAs' bytes; the TMA (cta_group::2) completes on the leader's barrier
                        if rank == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bar, 2 * self.stage_bytes
                                )
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                bar, self.stage_bytes
                            )
                    if cutlass.const_expr(self.n_tail):  # (N tile, K tile, expert)
                        cute.copy(
                            tma_w,
                            tWg[(None, tile % NT, kt, experts[mb])],
                            tWs[(None, slot)],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                            **dw,
                        )
                        cute.copy(
                            tma_wsf,
                            tWSFg[(None, tile % NT, kt, experts[mb])],
                            tWSFs[(None, slot)],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                        )
                    else:
                        wp = (
                            dict(cache_policy=pol)
                            if cutlass.const_expr(self.evict_weights)
                            else {}
                        )
                        cute.copy(
                            tma_w,
                            tWg[(None, wtile_b, kt, 0)],
                            tWs[(None, slot)],
                            tma_bar_ptr=bar,
                            **wp,
                            **dw,
                        )
                        cute.copy(
                            tma_wsf,
                            tWSFg[(None, wtile, kt, 0)],
                            tWSFs[(None, slot)],
                            tma_bar_ptr=bar,
                            **wp,
                        )
                if cutlass.const_expr(not self.x_gather):
                    cute.copy(
                        tma_x,
                        tXg[(None, xb, kt, 0)],
                        tXs[(None, slot)],
                        tma_bar_ptr=bar,
                        **dx,
                    )
                if cutlass.const_expr(not self.x_sf_gather):
                    cute.copy(
                        tma_xsf,
                        tXSFg[(None, mb, kt, 0)],
                        tXSFs[(None, slot)],
                        tma_bar_ptr=bar,
                    )
                if cutlass.const_expr(self.x_sf_gather == 3):
                    sfb_bar = st.sf_full.data_ptr() + slot
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sfb_bar, self.bm * self.bk // 16
                        )
                    cute.copy(
                        tma_xsf,
                        tXSFg[(None, mb, kt)],
                        tXSFs[(None, slot)],
                        tma_bar_ptr=sfb_bar,
                    )
                seq += 1
                slot += 1
                if slot == self.stages:
                    slot = Int32(0)
                    phase ^= 1
            if cutlass.const_expr(self.zero_rows):
                self._zero_rows(zero_ptr, tile, num_tiles, ptab[0], rank)
            if cutlass.const_expr(self.dynamic):
                tile = self._put_tile(
                    st,
                    claims,
                    cute.arch.shuffle_sync(claim, 0),
                    tile,
                    tile_idx,
                    grid,
                    num_tiles,
                    claiming,
                )
                tile_idx += 1
            elif cutlass.const_expr(self.skip_empty):
                tile = self._skip(tile + grid, grid, num_tiles, limits)
            else:
                tile += grid
        if cutlass.const_expr(self.zero_rows):
            if num_tiles == 0:
                self._zero_rows(zero_ptr, bidx, grid, ptab[0], rank)

    @cute.jit
    def _zero_rows(self, y_ptr, row, step, m, rank):
        """zero_rows, load warp: zero the BF16 output rows row, row + step, ... < m (16-byte stores by the 32 lanes;
        CTA pairs: this CTA's half of the columns)."""
        NC = self.zero_rows * 2 // 16  # 16-byte chunks per row
        CH = NC // 2 if self.two_cta else NC
        c0 = rank * CH if self.two_cta else 0
        lane = cute.arch.lane_idx()
        r = row
        while r < m:
            base = y_ptr.toint() + Int64(r) * (self.zero_rows * 2) + c0 * 16
            for j in cutlass.range_constexpr((CH + 31) // 32):
                c = lane + 32 * j
                if cutlass.const_expr(CH % 32 == 0):
                    st_global_v4(
                        base + c * 16, [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
                    )
                else:
                    if c < CH:
                        st_global_v4(
                            base + c * 16, [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
                        )
            r += step

    @cute.jit
    def _claiming(self, num_tiles, grid):
        """Dynamic schedule: whether this launch claims tiles: at least ``claim_tiles`` of them, past the two
        round-robin waves. Otherwise every warp follows ``tile += grid``."""
        return (num_tiles > 2 * grid) & (num_tiles >= self.claim_tiles)

    @cute.jit
    def _put_tile(self, st, claims, claim, tile, idx, grid, num_tiles, claiming):
        """Dynamic schedule, load warp: the tile after this CTA's idx-th (``tile``): ``tile + grid`` after the first
        or without claims, else from ``claim`` (the counter's value before the claim), put in the ring for the MMA
        and epilogue warps. In a claiming launch every tile but the CTAs' first claims once, so the claim of value
        ``num_tiles - grid - 1`` is the launch's last: it zeroes the counter for the next launch."""
        claimed = (tile >= grid) & claiming
        if (cute.arch.lane_idx() == 0) & claimed & (claim == num_tiles - grid - 1):
            cute.arch.atomic_exch(claims, Int32(0), sem="relaxed", scope="gpu")
        tile = Int32(cutlass.select_(claimed, 2 * grid + claim, tile + grid))
        slot = idx % self.ring
        ring = st.sTiles.get_tensor(cute.make_layout(self.ring))
        full = st.tile_full.data_ptr() + slot
        empty = st.tile_empty.data_ptr() + slot
        phase = ((idx // self.ring) % 2) ^ 1
        if claiming:
            cute.arch.mbarrier_wait(empty, phase)
            with cute.arch.elect_one():
                ring[slot] = tile
                cute.arch.mbarrier_arrive(full)
        return tile

    @cute.jit
    def _next_tile(
        self,
        st,
        tile,
        idx,
        grid,
        num_tiles,
        threads: cutlass.Constexpr,
        bar_id: cutlass.Constexpr = 0,
    ):
        """The tile after this CTA's idx-th: ``tile + grid``, or (dynamic, in a claiming launch) the one the load warp
        put in the ring, whose slot is freed once the caller's ``threads`` (the MMA warp, the epilogue warps) have
        read it."""
        nxt = tile + grid
        if cutlass.const_expr(self.dynamic):
            claiming = self._claiming(num_tiles, grid)
            slot = idx % self.ring
            ring = st.sTiles.get_tensor(cute.make_layout(self.ring))
            full = st.tile_full.data_ptr() + slot
            empty = st.tile_empty.data_ptr() + slot
            phase = (idx // self.ring) % 2
            tidx, _, _ = cute.arch.thread_idx()
            if claiming:
                cute.arch.mbarrier_wait(full, phase)
                nxt = ring[slot]
                if cutlass.const_expr(threads == 32):
                    cute.arch.sync_warp()
                else:
                    bar_sync(bar_id, threads)
                if tidx % threads == 0:
                    cute.arch.mbarrier_arrive(empty)
        return nxt

    @cute.jit
    def _skip(self, tile, grid, num_tiles, limits):
        """skip_empty: the first tile from ``tile`` on (step grid) whose row block holds routes; else ``tile``."""
        t = tile
        if cutlass.const_expr(self.skip_empty):
            TNT = self.tnt
            go = t < num_tiles
            while go:
                mb = t // TNT
                empty = limits[mb] <= mb * self.bm
                t = Int32(cutlass.select_(empty, t + grid, t))
                go = empty & (t < num_tiles)
        return t

    # ---------------------------------------------------------------------------------- activation scales
    @cute.jit
    def _sf_convert_warp(
        self, st, bidx, grid, num_tiles, stage_ready, par: cutlass.Constexpr
    ):
        """x_sf_gather 3: the stages of the slots s % 2 == ``par`` (two such warps; by slot, not by stage, so that each
        waits on every phase of its slots also with an odd stage count): once a stage's row-major box of
        activation scales has landed in its scale block (``sf_full``), rearrange it in place into the 128x4 layout
        (word q of row i at q * 512 + i % 32 * 16 + i / 32 * 4) and arrive on the stage, as its second producer. Lane l
        takes rows l + 32 j < bm (bm 16: lanes 16-31 read rows 0-15 and do not store); every lane reads before any
        writes (warp sync)."""
        KT = self.k // self.bk
        bm = self.bm
        lane = cute.arch.lane_idx()
        xsf_s = st.sXSF.data_ptr().toint()
        sf_full = st.sf_full.data_ptr()
        rows_i = [lane + 32 * j for j in range(bm // 32)] if bm >= 32 else [lane % bm]
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            for kt in cutlass.range(KT, unroll=1):
                seq = tile_idx * KT + kt
                slot = seq % self.stages
                if (
                    slot % 2 == par
                ):  # this warp's slots: it waits on every phase of each
                    self._sf_convert_stage(
                        st, xsf_s, sf_full, stage_ready, slot, seq, rows_i, lane
                    )
            tile = self._next_tile(st, tile, tile_idx, grid, num_tiles, 32)
            tile_idx += 1

    @cute.jit
    def _sf_convert_stage(
        self, st, xsf_s, sf_full, stage_ready, slot, seq, rows_i, lane
    ):
        """x_sf_gather 3: one stage of ``_sf_convert_warp`` (block_m >= 32: every lane stores rows)."""
        Q = self.bk // 64
        W = self.bk // 16
        cute.arch.mbarrier_wait(sf_full + slot, (seq // self.stages) % 2)
        base = xsf_s + slot * self.asf_bytes
        vals = [
            [
                cute.arch.load(smem_ptr(Uint32, base + i * W + 4 * q, 4), Uint32)
                for q in range(Q)
            ]
            for i in rows_i
        ]
        cute.arch.sync_warp()
        for j in cutlass.range_constexpr(len(rows_i)):
            i = rows_i[j]
            dst = base + i % 32 * 16 + i // 32 * 4
            for q in cutlass.range_constexpr(Q):
                cute.arch.store(smem_ptr(Uint32, dst + q * 512, 4), vals[j][q])
        if cutlass.const_expr(
            self.x_gather
        ):  # the stage's gathered rows have landed (all gather threads')
            cute.arch.mbarrier_wait(
                st.a_full.data_ptr() + slot, (seq // self.stages) % 2
            )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        if lane == 0:
            if cutlass.const_expr(
                self.two_cta
            ):  # the leader's MMA reads this CTA's scales (CTA scope: __init__)
                cute.arch.mbarrier_arrive(
                    stage_ready + slot, 0, scope=nvvm.MemScopeKind.CTA
                )
            else:
                cute.arch.mbarrier_arrive(stage_ready + slot)

    @cute.jit
    def _gather_warps(
        self, st, tidx, rowtok, xq, limits, bidx, grid, num_tiles, stage_free, rank
    ):
        """x_gather: the stages' activation rows from the input by token (see __init__). Thread g of the 128 takes
        16-byte chunk c = g % 8 of rows r = g / 8 + 16 i (i < 8): source row token(r), bytes kt * 128 + 16 c; in the
        stage at r * 128 + (c ^ (r % 8)) * 16 (128B swizzle; r % 8 = g / 8 % 8 for every i). Rows at or past the
        tile's limit are not loaded (padding: never stored)."""
        NT = self.nt
        KT = self.k // self.bk
        g = tidx - 32 * (self.D + 4)
        c = g % 8
        r0 = g // 8
        RB = self.k // 2  # bytes per input row
        rt = cute.make_tensor(
            cute.make_ptr(
                Int32, rowtok.toint(), cute.AddressSpace.gmem, assumed_align=4
            ),
            cute.make_layout(1 << 30),
        )
        xbase = xq.toint() + Int64(c * 16)
        sbase = st.sX.data_ptr().toint() + r0 * 128 + ((c ^ (r0 % 8)) << 4)
        a_full = st.a_full.data_ptr()
        slot = Int32(0)
        phase = Int32(0)
        tile = bidx
        while tile < num_tiles:
            mb = tile // NT
            if cutlass.const_expr(self.two_cta):
                mb = 2 * (tile // NT) + rank
            row0 = mb * self.bm + r0
            lim = limits[mb]
            toks = [rt[row0 + 16 * i] for i in range(8)]
            for kt in cutlass.range(KT, unroll=1):
                cute.arch.mbarrier_wait(stage_free + slot, phase ^ 1)
                dst = sbase + slot * self.a_bytes
                for i in cutlass.range_constexpr(8):
                    if row0 + 16 * i < lim:
                        nvvm.cp_async_shared_global(
                            smem_ptr(Uint8, dst + 2048 * i, 16).to_llvm_ptr(),
                            cute.make_ptr(
                                Uint8,
                                xbase + Int64(toks[i]) * RB + kt * 128,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ).to_llvm_ptr(),
                            16,
                            nvvm.LoadCacheModifierKind.CG,
                        )
                cute.arch.cp_async_mbarrier_arrive_noinc(a_full + slot)
                slot += 1
                if slot == self.stages:
                    slot = Int32(0)
                    phase ^= 1
            tile += grid

    @cute.jit
    def _place_tiles(
        self, ecount_ptr, sexp, slim, stb, scount, gexp, glim, gtbp, gcount, bidx
    ):
        """place_tiles, the load warp: the tile table from the expert counts (Scatter._rank_rows' scan: lane l holds
        experts EPL l .. EPL l + EPL - 1, each rounded up to block_m rows; tiles in expert order) into shared memory;
        CTA 0 also writes it to global memory (experts, limits, tbase, count) and fences before the CTA barrier."""
        E, BM, M = self.place_experts, self.bm, self.tokens
        EPL = (E + 31) // 32
        exact = E % (32 * EPL) == 0
        lane = cute.arch.lane_idx()
        ec = cute.make_tensor(
            cute.make_ptr(
                Int32, ecount_ptr.toint(), cute.AddressSpace.gmem, assumed_align=4
            ),
            cute.make_layout(1 << 30),
        )
        gtb = cute.make_tensor(
            cute.make_ptr(Int32, gtbp.toint(), cute.AddressSpace.gmem, assumed_align=4),
            cute.make_layout(1 << 30),
        )
        cnt, pc = [], []
        for j in cutlass.range_constexpr(EPL):
            e = EPL * lane + j
            c = Int32(0)
            if cutlass.const_expr(exact):
                c = ec[e]
            else:
                c = Int32(
                    cutlass.select_(
                        e < E, ec[Int32(cutlass.select_(e < E, e, E - 1))], Int32(0)
                    )
                )
            cnt.append(c)
            pc.append((c + BM - 1) // BM * BM)
        v = pc[0]
        for j in cutlass.range_constexpr(1, EPL):
            v = v + pc[j]
        incl = v
        for d in cutlass.range_constexpr(5):
            u = cute.arch.shuffle_sync_up(incl, offset=1 << d)
            incl = incl + Int32(cutlass.select_(lane >= (1 << d), u, Int32(0)))
        base = [incl - v]
        for j in cutlass.range_constexpr(1, EPL):
            base.append(base[j - 1] + pc[j - 1])
        pub = bidx == 0
        for j in cutlass.range_constexpr(EPL):
            first = base[j] // BM
            e = EPL * lane + j
            for i in cutlass.range(
                pc[j] // BM, unroll=1
            ):  # experts past E have no routes
                sexp[first + i] = e
                slim[first + i] = base[j] + cnt[j]
                stb[first + i] = e * M + i * BM
                if pub:
                    gexp[first + i] = e
                    glim[first + i] = base[j] + cnt[j]
                    gtb[first + i] = e * M + i * BM
        total = cute.arch.shuffle_sync(incl, 31)
        scount[0] = total
        if pub:
            gcount[0] = total
            cute.arch.fence_acq_rel_gpu()

    @cute.jit
    def _release_acc(self, acc_free, buf, tidx):
        """Thread 0 of the epilogue frees accumulator ``buf`` (None: the only one) for the MMA warp (after the
        epilogue's barrier); CTA pairs arrive with CTA scope on the leader's barrier (the leader's MMA reuses both
        CTAs' accumulators). The barrier's address is computed by thread 0 only."""
        if tidx == 0:
            bar = acc_free
            if cutlass.const_expr(buf is not None):
                bar = acc_free + buf
            if cutlass.const_expr(self.two_cta):
                cute.arch.mbarrier_arrive(bar, 0, scope=nvvm.MemScopeKind.CTA)
            else:
                cute.arch.mbarrier_arrive(bar)

    @cute.jit
    def _sf_fence_warp(
        self, st, bidx, grid, num_tiles, stage_ready, par: cutlass.Constexpr
    ):
        """x_sf_gather 2: the slots s % 2 == ``par``: once the gather warps' rows and scales of a stage have landed
        (a_full), fence the async proxy and arrive on the stage (its second producer, after the TMA's weights)."""
        KT = self.k // self.bk
        a_full = st.a_full.data_ptr()
        lane = cute.arch.lane_idx()
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            for kt in cutlass.range(KT, unroll=1):
                seq = tile_idx * KT + kt
                slot = seq % self.stages
                if slot % 2 == par:
                    cute.arch.mbarrier_wait(a_full + slot, (seq // self.stages) % 2)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(stage_ready + slot)
            tile = self._next_tile(st, tile, tile_idx, grid, num_tiles, 32)
            tile_idx += 1

    @cute.jit
    def _gather_warps_wm(
        self,
        st,
        tidx,
        eroute,
        xq,
        xsf,
        tbp,
        invp,
        limits,
        bidx,
        grid,
        num_tiles,
        stage_free,
    ):
        """route_rows, weight-major gate/up (x_gather, x_sf_gather 2): the stages' activation rows and block scales from
        the MoE input by token, with cp.async (completion on a_full: one noinc arrival per thread and stage). Row i of
        tile t is route eroute[tbase[t] + i] (i < the tile's rows), token = route / topk. FP4: thread g takes 16-byte
        chunk c = g % CH (CH = bk / 32 per row and stage) of rows g / CH + (128 / CH) j, stored as the TMA would (K-major
        128B swizzle: 128-byte K block c / 8 at c / 8 * bm * 128, row r at r * 128, chunk (c % 8) ^ (r % 8)); scales:
        4-byte word q = g % W (W = bk / 64) of rows g / W + (128 / W) j into the 128x4 layout (q * 512 + r % 32 * 16 +
        r / 32 * 4). Rows past the tile's routes are not loaded. The CTAs of the tiles in column 0 also write each
        route's row to inverse[] (ReduceRoutes)."""
        NT = self.nt
        KT = self.k // self.bk
        bm = self.bm
        T = self.route_rows
        g = tidx - 32 * (self.D + 4)
        CH = self.bk // 32
        W = self.bk // 64
        RPF = 128 // CH
        NF = (bm + RPF - 1) // RPF
        RPS = 128 // W
        NS = (bm + RPS - 1) // RPS
        c = g % CH
        q = g % W
        RB = self.k // 2
        SB = self.k // 16
        er = cute.make_tensor(
            cute.make_ptr(
                Int32, eroute.toint(), cute.AddressSpace.gmem, assumed_align=4
            ),
            cute.make_layout(1 << 30),
        )
        tbt = tbp
        inv = cute.make_tensor(
            cute.make_ptr(Int32, invp.toint(), cute.AddressSpace.gmem, assumed_align=4),
            cute.make_layout(1 << 30),
        )
        sX = st.sX.data_ptr().toint()
        sXSF = st.sXSF.data_ptr().toint()
        a_full = st.a_full.data_ptr()
        xbase = xq.toint()
        sbase = xsf.toint()
        cute.arch.griddepcontrol_wait()
        slot = Int32(0)
        phase = Int32(0)
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            mb = tile // NT
            lim = limits[mb] - mb * bm
            base = tbt[mb]
            rf = [g // CH + RPF * j for j in range(NF)]
            rs = [g // W + RPS * j for j in range(NS)]
            routes_f = [
                er[base + Int32(cutlass.select_(r < lim, r, Int32(0)))] for r in rf
            ]
            toks_f = [rt // T for rt in routes_f]
            toks_s = [
                er[base + Int32(cutlass.select_(r < lim, r, Int32(0)))] // T for r in rs
            ]
            if cutlass.const_expr(self.zero_out):
                # fused finalize: zero the output row of each route in slot 0 (part c of CH, 16-byte stores)
                ZB = 2 * self.zero_out // CH
                if tile % NT == 0:
                    for j in cutlass.range_constexpr(NF):
                        if (rf[j] < lim) & (routes_f[j] % T == 0):
                            za = (
                                invp.toint()
                                + Int64(toks_f[j]) * (2 * self.zero_out)
                                + c * ZB
                            )
                            for z in cutlass.range_constexpr(ZB // 16):
                                st_global_v4(
                                    za + 16 * z,
                                    [Uint32(0), Uint32(0), Uint32(0), Uint32(0)],
                                )
            else:
                if (tile % NT == 0) & (c == 0):
                    for j in cutlass.range_constexpr(NF):
                        if rf[j] < lim:
                            inv[routes_f[j]] = mb * bm + rf[j]
            for kt in cutlass.range(KT, unroll=1):
                cute.arch.mbarrier_wait(stage_free + slot, phase ^ 1)
                dstx = sX + slot * self.a_bytes + (c // 8) * (bm * 128) + ((c % 8) << 4)
                for j in cutlass.range_constexpr(NF):
                    r = rf[j]
                    if r < lim:
                        nvvm.cp_async_shared_global(
                            smem_ptr(
                                Uint8, (dstx + r * 128) ^ ((r % 8) << 4), 16
                            ).to_llvm_ptr(),
                            cute.make_ptr(
                                Uint8,
                                xbase
                                + Int64(toks_f[j]) * RB
                                + kt * (self.bk // 2)
                                + 16 * c,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ).to_llvm_ptr(),
                            16,
                            nvvm.LoadCacheModifierKind.CG,
                        )
                dsts = sXSF + slot * self.asf_bytes + q * 512
                for j in cutlass.range_constexpr(NS):
                    r = rs[j]
                    if r < lim:
                        nvvm.cp_async_shared_global(
                            smem_ptr(
                                Uint8, dsts + r % 32 * 16 + r // 32 * 4, 4
                            ).to_llvm_ptr(),
                            cute.make_ptr(
                                Uint8,
                                sbase
                                + Int64(toks_s[j]) * SB
                                + kt * (self.bk // 16)
                                + 4 * q,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ).to_llvm_ptr(),
                            4,
                            nvvm.LoadCacheModifierKind.CA,
                        )
                cute.arch.cp_async_mbarrier_arrive_noinc(a_full + slot)
                slot += 1
                if slot == self.stages:
                    slot = Int32(0)
                    phase ^= 1
            tile = self._next_tile(st, tile, tile_idx, grid, num_tiles, 128, 2)
            tile_idx += 1

    @cute.jit
    def _sf_warp(
        self,
        st,
        xsf_raw,
        rowtok,
        limits,
        bidx,
        grid,
        num_tiles,
        stage_free,
        stage_ready,
        par: cutlass.Constexpr,
    ):
        """x_sf_gather 1: the activation block scales of the stages in the slots s % 2 == ``par`` (two such warps; by
        slot so that each waits on every phase of its slots) of the tile's rows from the input. Lane l takes rows
        l + 32 j (bm 16: lanes 16-31 repeat rows 0-15 and do not store); per row the stage's K / 16 scales (bk / 64
        words) are loaded before the slot is free, then stored into the stage's 128x4 block (word q of row i at
        q * 512 + i % 32 * 16 + i / 32 * 4), the async proxy is fenced and the warp arrives on the stage, as the
        second producer next to the TMA. Rows past the tile's routes (``limits``) read the scales of its last route
        (ignored by the MMA's padded columns): their row tokens are stale (the workspace is shared by every batch
        size: another M's values)."""
        NT = self.nt
        KT = self.k // self.bk
        bm = self.bm
        Q = self.bk // 64  # 4-scale words per row and stage
        RL = (bm + 31) // 32  # rows per lane
        lane = cute.arch.lane_idx()
        rowb = self.k // 16  # input scale bytes per token
        xsf_s = st.sXSF.data_ptr().toint()
        if cutlass.const_expr(self.launch_pdl):
            cute.arch.griddepcontrol_wait()  # rowtok and the tiles come from the scatter
        rt = cute.make_tensor(
            cute.make_ptr(
                Int32, rowtok.toint(), cute.AddressSpace.gmem, assumed_align=4
            ),
            cute.make_layout(1 << 30),
        )
        rows_i = [lane + 32 * j for j in range(RL)] if bm >= 32 else [lane % bm]
        seq = Int32(0)
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            mb = tile // NT
            last = limits[mb] - 1
            srcs = [
                xsf_raw.toint() + Int64(rt[cutlass.min(mb * bm + i, last)]) * rowb
                for i in rows_i
            ]
            for kt in cutlass.range(KT, unroll=1):
                seq = tile_idx * KT + kt
                slot = seq % self.stages
                phase = (seq // self.stages) % 2
                if (
                    slot % 2 == par
                ):  # this warp's slots: it waits on every phase of each
                    vals = []
                    for src in srcs:
                        for h in cutlass.range_constexpr(Q // 4):
                            v = cute.TensorSSA(
                                cute.arch.load(
                                    cute.make_ptr(
                                        Uint32,
                                        src + kt * (4 * Q) + 16 * h,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    _vec(4, Uint32),
                                ),
                                (4,),
                                Uint32,
                            )
                            vals.append([v[e] for e in range(4)])
                    cute.arch.mbarrier_wait(stage_free + slot, phase ^ 1)
                    base = xsf_s + slot * self.asf_bytes
                    for j in cutlass.range_constexpr(len(rows_i)):
                        i = rows_i[j]
                        dst = base + i % 32 * 16 + i // 32 * 4
                        if cutlass.const_expr(bm >= 32):
                            for q in cutlass.range_constexpr(Q):
                                cute.arch.store(
                                    smem_ptr(Uint32, dst + q * 512, 4),
                                    vals[j * (Q // 4) + q // 4][q % 4],
                                )
                        else:
                            if lane < bm:
                                for q in cutlass.range_constexpr(Q):
                                    cute.arch.store(
                                        smem_ptr(Uint32, dst + q * 512, 4),
                                        vals[j * (Q // 4) + q // 4][q % 4],
                                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(stage_ready + slot)
            tile = self._next_tile(st, tile, tile_idx, grid, num_tiles, 32)
            tile_idx += 1

    # ---------------------------------------------------------------------------------- MMA
    @cute.jit
    def _mma_warp_raw(
        self,
        st,
        tmem_base,
        bidx,
        grid,
        num_tiles,
        stage_free,
        stage_ready,
        acc_free,
        acc_ready,
        limits=None,
    ):
        """MMA partition: per stage copy the weight then activation block scales to TMEM
        (``tcgen05.cp.warpx4.32x128b``, 512 B per 128 rows x 64 K), then one MMA per 64 K, then
        commit the stage; after the tile commit the accumulator.

        TMEM: accumulator at column 0, weight scales at acc_cols (column 8 kc + 4 rb for 256
        weight rows), activation scales after them (column 4 kc).

        Slot, phase and the stage's four base descriptors are carried from the previous stage, so only
        the per-instruction offsets sit between the stage wait and the copies.

        ``live_n``: each tile's instruction descriptor takes N = the tile's live tokens (its row block's routes up to
        ``limits``) rounded up to 8, read before the tile's accumulator wait (wm2: twice that where it is at most
        bm / 2, else bm).
        """
        KT = self.kt
        KC = self.bk // 64
        RB = self.bn // 128
        HALF = 4  # MMAs per 128-byte K chunk of an operand row
        TWO = self.two_cta
        idesc = mma_idesc(*(self.mma_shape if TWO else self.mma_mn))
        sW = st.sW.data_ptr().toint()
        sX = st.sX.data_ptr().toint()
        sWSF = st.sWSF.data_ptr().toint()
        sXSF = st.sXSF.data_ptr().toint()
        wsf_col = tmem_base + self.acc_cols * self.acc_stages
        xsf_col = wsf_col + self.wsf_cols
        A = self.acc_stages
        # Descriptor address fields are 16-byte units below 2^14: a stage step is a plain add.
        wsf0 = Int64((sWSF >> 4) & 0x3FFF) | SCALE_DESC
        xsf0 = Int64((sXSF >> 4) & 0x3FFF) | SCALE_DESC
        odesc = OPERAND_DESC_SW64 if self.bk == 128 else OPERAND_DESC
        w0 = Int64((sW >> 4) & 0x3FFF) | odesc
        x0 = Int64((sX >> 4) & 0x3FFF) | odesc
        last_step = self.stages - 1
        tile_idx = Int32(0)
        slot = Int32(0)
        phase = Int32(0)
        wsf_desc, xsf_desc, w_desc, x_desc = wsf0, xsf0, w0, x0
        tile = bidx
        if cutlass.const_expr(self.skip_empty):
            tile = self._skip(tile, grid, num_tiles, limits)
        while tile < num_tiles:
            if cutlass.const_expr(self.live_n):
                mb = tile // self.tnt
                live = cutlass.min(limits[mb] - mb * self.bm, self.bm)
                n = cutlass.max((live + 7) & Int32(-8), Int32(8))
                if cutlass.const_expr(TWO):  # wm2
                    n = cutlass.select_(live > self.bm // 2, Int32(self.bm), 2 * n)
                tile_idesc = Int32(
                    mma_idesc((self.mma_shape if TWO else self.mma_mn)[0], 0)
                ) | ((n >> 3) << 17)
            else:
                tile_idesc = idesc
            buf = tile_idx % A if A > 1 else 0
            acc_t = tmem_base + buf * self.acc_cols if A > 1 else tmem_base
            cute.arch.mbarrier_wait(acc_free + buf, ((tile_idx // A) % 2) ^ 1)
            for kt in cutlass.range(KT, unroll=1):
                cute.arch.mbarrier_wait(stage_ready + slot, phase)
                with cute.arch.elect_one():
                    for kc in cutlass.range_constexpr(KC):
                        for rb in cutlass.range_constexpr(RB):
                            off = rb * KC * 512 + kc * 512
                            tcgen05_cp_32x128b_x4(
                                wsf_col + 4 * (RB * kc + rb),
                                wsf_desc + ((off >> 4) | ((off >> 7) & 7) << 49),
                                TWO,
                            )
                with cute.arch.elect_one():
                    for kc in cutlass.range_constexpr(KC):
                        off = kc * 512
                        tcgen05_cp_32x128b_x4(
                            xsf_col + 4 * kc,
                            xsf_desc + ((off >> 4) | ((off >> 7) & 7) << 49),
                            TWO,
                        )
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(KC):
                        w_k = (kb % HALF) * 2 + (kb // HALF) * (self.cta_bn * 128 >> 4)
                        x_k = (kb % HALF) * 2 + (kb // HALF) * (self.cta_bm * 128 >> 4)
                        acc = (kt > 0) if kb == 0 else True
                        if cutlass.const_expr(self.token_major):
                            tcgen05_mma_nvf4(
                                acc_t,
                                x_desc + x_k,
                                w_desc + w_k,
                                idesc,
                                xsf_col + 4 * kb,
                                wsf_col + 4 * RB * kb,
                                acc,
                                TWO,
                            )
                        else:
                            tcgen05_mma_nvf4(
                                acc_t,
                                w_desc + w_k,
                                x_desc + x_k,
                                tile_idesc,
                                wsf_col + 4 * kb,
                                xsf_col + 4 * kb,
                                acc,
                                TWO,
                            )
                    if cutlass.const_expr(TWO):  # both CTAs' slot is free
                        tcgen05.commit(
                            stage_free + slot, Int16(3), tcgen05.CtaGroup.TWO
                        )
                    else:
                        tcgen05.commit(stage_free + slot)
                wrap = slot == last_step
                slot = Int32(cutlass.select_(wrap, Int32(0), slot + 1))
                phase = Int32(cutlass.select_(wrap, phase ^ 1, phase))
                wsf_desc = Int64(
                    cutlass.select_(wrap, wsf0, wsf_desc + (self.wsf_bytes >> 4))
                )
                xsf_desc = Int64(
                    cutlass.select_(wrap, xsf0, xsf_desc + (self.asf_bytes >> 4))
                )
                w_desc = Int64(cutlass.select_(wrap, w0, w_desc + (self.w_bytes >> 4)))
                x_desc = Int64(cutlass.select_(wrap, x0, x_desc + (self.a_bytes >> 4)))
            with cute.arch.elect_one():
                if cutlass.const_expr(TWO):
                    tcgen05.commit(acc_ready + buf, Int16(3), tcgen05.CtaGroup.TWO)
                else:
                    tcgen05.commit(acc_ready + buf)
            tile = self._next_tile(st, tile, tile_idx, grid, num_tiles, 32)
            if cutlass.const_expr(self.skip_empty):
                tile = self._skip(tile, grid, num_tiles, limits)
            tile_idx += 1
        if cutlass.const_expr(self.launch_pdl and not self.act):
            cute.arch.griddepcontrol_launch_dependents()

    # ---------------------------------------------------------------------------------- epilogues
    @cute.jit
    def _epilogue_token_major_fused(
        self,
        st,
        tmem_base,
        warp,
        tidx,
        bidx,
        grid,
        num_tiles,
        a_gs,
        w_gs,
        experts,
        limits,
        out,
        acc_free,
        acc_ready,
        rowtok,
        rank=0,
        ptab=None,
    ):
        """Token-major down GEMM (the long batches'; fused_out or ``pair_table``): 128 tokens x bn weight rows added into
        the tokens' output rows (fused_out), else stored into the routed rows [rows, N] (unfused pairs). Thread (warp w, lane l) owns token row r = 32 (w % 4) + l and the bn / 2 columns of half g = w // 4
        (tcgen05.ld.32x32b): its values times w_gs and the row's routing weight (a_gs) go as BF16 into row r of the
        staging (rows bn * 2 + 16 bytes apart); after a barrier, thread r (half 0) adds
        the row into its token's output row with one cp.reduce.async.bulk .add.noftz.bf16 (rows past the tile's
        routes are skipped) and, before the next tile's stores, waits until it has read the staging. two_cta
        (pair_table): pair tile p's CTA ``rank`` takes the rows of block b0 + rank and the expert and row end of b0
        (its accumulator: the pair MMA's 128 rows of this CTA); both CTAs free the leader's accumulator."""
        NT = self.nt
        HC = self.bn // 2
        g = warp // 4
        lane = tidx % 32
        r = (warp % 4) * 32 + lane
        out_base = out.iterator.toint()
        stg = st.sScratch.data_ptr().toint()
        RS = self.row_stride
        tile_idx = Int32(0)
        tile = bidx
        if cutlass.const_expr(self.skip_empty):
            tile = self._skip(tile, grid, num_tiles, limits)
        while tile < num_tiles:
            mb = tile // NT
            eb = mb  # the row block of the tile's expert and row end
            if cutlass.const_expr(self.pair_table):
                eb = ptab[tile // NT]
                mb = eb + rank
            nb = tile % NT
            wg = w_gs[experts[eb]]
            row_end = limits[eb]
            row = mb * self.bm + r
            rrd = row  # the row whose token and weight are read (pairs: rank 1's rows may lie past the routes)
            if cutlass.const_expr(self.pair_table):
                rrd = cutlass.min(row, row_end - 1)
            tok = row  # unfused: the routed row itself
            if cutlass.const_expr(self.fused_out):
                tok = cute.arch.load(
                    cute.make_ptr(
                        Int32,
                        rowtok.toint() + Int64(rrd) * 4,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    Int32,
                )
            rw = Float32(1.0)  # unit_a_gs: no row scale (acc * w_gs * 1 is acc * w_gs)
            if cutlass.const_expr(not self.unit_a_gs):
                rw = a_gs[rrd]
            A = self.acc_stages
            buf = tile_idx % A if A > 1 else 0
            cute.arch.mbarrier_wait(acc_ready + buf, (tile_idx // A) % 2)
            acc = tmem_load(
                tmem_base + buf * self.acc_cols + g * HC, HC, 1, min(HC, 64), tidx % 128
            )
            if tidx < 128:
                cute.arch.cp_async_bulk_wait_group(
                    0, read=True
                )  # the previous tile's row has left the staging
            bar_sync(0, self.epi_threads)
            self._release_acc(acc_free, buf, tidx)
            sc = wg * rw
            rbase = stg + r * RS + g * HC * 2
            for c in cutlass.range_constexpr(HC // 8):
                words = [
                    cvt_bf16x2(
                        acc[8 * c + 2 * i] * sc,
                        acc[8 * c + 2 * i + 1] * sc,
                        packed=True,
                    )
                    for i in range(4)
                ]
                st_shared_v4(rbase + 16 * c, words)
            cute.arch.fence_proxy("async.shared", space="cta")
            bar_sync(0, self.epi_threads)
            if tidx < 128:
                if row < row_end:
                    dst = cute.make_ptr(
                        BFloat16,
                        out_base + (Int64(tok) * self.n + nb * self.bn) * 2,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    src = cute.make_ptr(
                        BFloat16, stg + r * RS, cute.AddressSpace.smem, assumed_align=16
                    )
                    if cutlass.const_expr(self.fused_out):
                        prims.cp_reduce_async_bulk_global_shared_cta(
                            dst, src, self.bn * 2, noftz=True
                        )
                    else:
                        prims.cp_async_bulk_global_shared_cta(dst, src, self.bn * 2)
                    cute.arch.cp_async_bulk_commit_group()
            tile = self._next_tile(
                st, tile, tile_idx, grid, num_tiles, self.epi_threads
            )
            if cutlass.const_expr(self.skip_empty):
                tile = self._skip(tile, grid, num_tiles, limits)
            tile_idx += 1
        if tidx < 128:
            cute.arch.cp_async_bulk_wait_group(
                0
            )  # this thread's reductions are complete before the CTA ends

    @cute.jit
    def _epilogue_token_major(
        self,
        st,
        tmem_base,
        warp,
        tidx,
        bidx,
        grid,
        num_tiles,
        a_gs,
        w_gs,
        experts,
        limits,
        out,
        acc_free,
        acc_ready,
        osf,
        rank=0,
    ):
        """128 tokens x 256 gate/up columns (interleaved per 8) -> NVFP4 of the 128 x 128 SwiGLU outputs with block scales
        only (second-level scale 1; the values of ``fp4_unit_word``).

        Thread (warp w, lane l) owns token row r = 32 (w % 4) + l and accumulator columns 64 c + 32 g + {0..31}
        (g = w // 4, c < 4; one ``tcgen05.ld.32x32b.x32`` each): the gate and up columns of the 16 outputs of block
        2 c + g (output e at gate column 16 (e / 8) + e % 8 of the chunk, up 8 further), so each thread holds whole
        16-output blocks (block 4 b + 2 cc + g with c = 2 b + cc). Per block: the outputs rounded to BF16, bmax = max |y|
        in the thread, sf = e4m3(bmax / 6),
        E2M1 of y * rcp(sf). The tile's FP4 bytes (128 rows x 64) are staged in the first 8 KB of the scratch, 16-byte
        chunk c of row r at c ^ (r / 2 % 4), and stored as 16-byte chunks (rows < limit); its block scales are staged
        after them (two 512-byte 128x4 blocks) and stored as 16-byte chunks. two_cta: CTA ``rank`` of the pair takes
        the row block 2 p + rank of pair tile p.
        Non-gated (``activation.ReLU2``; bn 128 or 256): the columns are the outputs, in passes of 128; in pass p
        thread (w, l) loads columns 128 p + 64 c + 32 g + {0..31} (c < 2): blocks 4 c + 2 g + h (h < 2) of the pass
        whole, staged and stored as above."""

        NT = self.nt
        K2 = self.out_cols  # outputs per row (the down GEMM's K)
        quad = warp % 4
        g = warp // 4
        lane = tidx % 32
        r = quad * 32 + lane
        taddr = tmem_base + 32 * g
        scratch = st.sScratch.data_ptr().toint()
        q_base = out.iterator.toint()
        osf_base = osf.toint()
        SFO = 8192  # the block-scale staging blocks (after the FP4 staging)
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            mb = tile // NT
            if cutlass.const_expr(self.two_cta):
                mb = 2 * (tile // NT) + rank
            nb = tile % NT
            expert = experts[mb]
            row_end = limits[mb]
            bar_sync(0, self.epi_threads)
            cute.arch.mbarrier_wait(acc_ready, tile_idx % 2)
            acc = tmem_load(taddr, (32, self.bn // 64), (1, 64), 32, tidx % 128)
            bar_sync(0, self.epi_threads)
            self._release_acc(acc_free, None, tidx)
            if cutlass.const_expr(self.unit_a_gs):  # per-tensor input: no row scale
                gs = w_gs[expert]
            else:
                gs = a_gs[mb * self.bm + r] * w_gs[expert]
            ngs = -gs
            if cutlass.const_expr(not self.gated):
                # non-gated: bn / 128 passes of 128 outputs (columns 128 p ..) through the 9 KB scratch; pass p's
                # thread chunks cg = 2 p + c hold columns 64 cg + 32 g + {0..31}: blocks 4 c + 2 g + h of the pass
                for p in cutlass.range_constexpr(self.bn // 128):
                    if cutlass.const_expr(p > 0):
                        bar_sync(
                            0, self.epi_threads
                        )  # the previous pass's stores have read the scratch
                    for c in cutlass.range_constexpr(2):
                        for h in cutlass.range_constexpr(2):
                            yt = []
                            for e2 in cutlass.range_constexpr(8):
                                pr = relu2_bf16_pair(
                                    gs,
                                    acc[32 * (2 * p + c) + 16 * h + 2 * e2],
                                    acc[32 * (2 * p + c) + 16 * h + 2 * e2 + 1],
                                )
                                yt.append(pr[0])
                                yt.append(pr[1])
                            mt = yt[
                                0
                            ]  # ReLU2's outputs are >= 0: the block max is a plain max
                            for e in cutlass.range_constexpr(1, 16):
                                mt = cute.arch.fmax(mt, yt[e])
                            sfwt, sfft, _ = cvt_e4m3x2(
                                mt * Float32(1.0 / 6.0), Float32(0.0)
                            )
                            invt = sel_if_pos(mt, cute.arch.rcp_approx(sfft))
                            wt = []
                            for wd in cutlass.range_constexpr(2):
                                wv = Uint32(0)
                                for pp in cutlass.range_constexpr(4):
                                    e = 8 * wd + 2 * pp
                                    sq = mul_f32x2(yt[e], yt[e + 1], invt, invt)
                                    wv = wv | (cvt_e2m1x2(sq[0], sq[1]) << (8 * pp))
                                wt.append(wv)
                            # row r's 64 FP4 bytes of the pass: 16-byte chunk 2 c + g at ((2 c + g) ^ (r / 2 % 4)), half h
                            sqt = (
                                scratch
                                + r * 64
                                + (((2 * c + g) ^ ((r >> 1) & 3)) << 4)
                                + 8 * h
                            )
                            cute.arch.store(smem_ptr(Uint32, sqt, 8), wt[0])
                            cute.arch.store(smem_ptr(Uint32, sqt + 4, 4), wt[1])
                            # staging block c, byte r % 32 * 16 + r / 32 * 4 + (2 g + h)
                            cute.arch.store(
                                smem_ptr(
                                    Uint8,
                                    scratch
                                    + SFO
                                    + c * 512
                                    + lane * 16
                                    + quad * 4
                                    + 2 * g
                                    + h,
                                    1,
                                ),
                                Uint8(sfwt & 0xFF),
                            )
                    bar_sync(0, self.epi_threads)
                    # the pass's two 128x4 scale blocks (bn / 64 per tile) and its 64 FP4 bytes per row; n_tail: none
                    # for a pass past n (the last N tile's second half: computed from the TMA's zeros, not stored)
                    pv = cutlass.Boolean(True)
                    if cutlass.const_expr(self.n_tail and p > 0):
                        pv = nb * self.bn + 128 * p < self.n
                    if (tidx < 64) & pv:
                        vt = ld_shared_v4(scratch + SFO + tidx * 16)
                        st_global_v4(
                            osf_base
                            + Int64(mb) * (K2 // 64 * 512)
                            + ((self.bn // 64) * nb + 2 * p + tidx // 32) * 512
                            + (tidx % 32) * 16,
                            vt,
                        )
                    for hh in cutlass.range_constexpr(2):
                        kq = tidx + 256 * hh
                        rq = kq >> 2
                        cqq = kq & 3
                        vq = ld_shared_v4(
                            scratch + rq * 64 + ((cqq ^ ((rq >> 1) & 3)) << 4)
                        )
                        rtq = mb * self.bm + rq
                        if (rtq < row_end) & pv:
                            st_global_v4(
                                q_base
                                + Int64(rtq) * (K2 // 2)
                                + nb * (self.bn // 2)
                                + 64 * p
                                + cqq * 16,
                                vq,
                            )
            else:
                for b in cutlass.range_constexpr(2):
                    for cc in cutlass.range_constexpr(2):
                        c = 2 * b + cc  # the thread's chunk: block 2 c + g
                        y = []
                        for ep in cutlass.range_constexpr(8):
                            e2 = [
                                2 * ep + h for h in range(2)
                            ]  # outputs e of the block
                            gt = [acc[32 * c + 16 * (e // 8) + e % 8] for e in e2]
                            up = [acc[32 * c + 16 * (e // 8) + 8 + e % 8] for e in e2]
                            if cutlass.const_expr(self.activation.plain):
                                sv = []
                                for h in cutlass.range_constexpr(2):
                                    ex = ex2_approx(
                                        fma_zero(ngs, gt[h]) * Float32(LOG2E)
                                    )
                                    sv.append(
                                        (gs * gt[h])
                                        * cute.arch.rcp_approx(ex + Float32(1.0))
                                    )
                                u = mul_f32x2(gs, gs, up[0], up[1])
                                o = mul_f32x2(u[0], u[1], sv[0], sv[1])
                            else:
                                o = [
                                    self.activation.apply(gs * gt[h], gs * up[h])
                                    for h in range(2)
                                ]
                            y.append(Float32(o[0].to(BFloat16)))
                            y.append(Float32(o[1].to(BFloat16)))
                        mx = y[0]
                        mn = y[0]
                        for e in cutlass.range_constexpr(1, 16):
                            mx = cute.arch.fmax(mx, y[e])
                            mn = cute.arch.fmin(mn, y[e])
                        m = cute.arch.fmax(mx, -mn)
                        sfw, sff, _ = cvt_e4m3x2(m * Float32(1.0 / 6.0), Float32(0.0))
                        inv = sel_if_pos(m, cute.arch.rcp_approx(sff))
                        words = []
                        for wd in cutlass.range_constexpr(2):
                            wv = Uint32(0)
                            for p in cutlass.range_constexpr(4):
                                e = 8 * wd + 2 * p
                                wv = wv | (
                                    cvt_e2m1x2(y[e] * inv, y[e + 1] * inv) << (8 * p)
                                )
                            words.append(wv)
                        # row r's 64 FP4 bytes of the tile: 16-byte chunk c = 2 b + cc at (c ^ (r / 2 % 4)), half g
                        sq = (
                            scratch
                            + r * 64
                            + (((2 * b + cc) ^ ((r >> 1) & 3)) << 4)
                            + 8 * g
                        )
                        cute.arch.store(smem_ptr(Uint32, sq, 8), words[0])
                        cute.arch.store(smem_ptr(Uint32, sq + 4, 4), words[1])
                        # block kb = 8 nb + 4 b + 2 cc + g of row r: staging block b,
                        # byte r % 32 * 16 + r / 32 * 4 + kb % 4
                        cute.arch.store(
                            smem_ptr(
                                Uint8,
                                scratch
                                + SFO
                                + b * 512
                                + lane * 16
                                + quad * 4
                                + 2 * cc
                                + g,
                                1,
                            ),
                            Uint8(sfw & 0xFF),
                        )
                bar_sync(0, self.epi_threads)
                if tidx < 64:
                    v = ld_shared_v4(scratch + SFO + tidx * 16)
                    st_global_v4(
                        osf_base
                        + Int64(mb) * (K2 // 64 * 512)
                        + (2 * nb + tidx // 32) * 512
                        + (tidx % 32) * 16,
                        v,
                    )
                # 512 chunks of 16 bytes (128 rows x 4), two per thread; rows at or past the limit are not stored
                for h in cutlass.range_constexpr(2):
                    k = tidx + 256 * h
                    rr = k >> 2
                    cq = k & 3
                    v = ld_shared_v4(scratch + rr * 64 + ((cq ^ ((rr >> 1) & 3)) << 4))
                    rt = mb * self.bm + rr
                    if rt < row_end:
                        st_global_v4(
                            q_base + Int64(rt) * (K2 // 2) + nb * 64 + cq * 16, v
                        )
            tile_idx += 1
            tile += grid

    @cute.jit
    def _epilogue_weight_major(
        self,
        st,
        tmem_base,
        warp,
        tidx,
        bidx,
        grid,
        num_tiles,
        a_gs,
        w_gs,
        experts,
        limits,
        out,
        acc_free,
        acc_ready,
        rank=0,
    ):
        """128 weight rows x bm tokens: thread = weight row (TMEM lane), all bm tokens; scalar stores. wm2: pair tile q,
        this CTA's N tile 2 (q % TNT) + rank; the accumulator is freed on the leader's barrier."""
        NT = self.nt
        TNT = self.tnt
        bm = self.bm
        lane = tidx % 32
        out_base = out.iterator.toint()
        a_gs_base = a_gs.iterator.toint()
        tile_idx = Int32(0)
        tile = bidx
        if cutlass.const_expr(self.skip_empty):
            tile = self._skip(tile, grid, num_tiles, limits)
        while tile < num_tiles:
            mb = tile // TNT
            nb = tile % NT
            if cutlass.const_expr(self.wm2):
                nb = 2 * (tile % TNT) + rank
            expert = experts[mb]
            row_end = limits[mb]
            wg = w_gs[expert]
            if cutlass.const_expr(not self.unit_a_gs):
                # lane l: row scales 4 l .. 4 l + 3 of the tile (lanes past bm / 4 re-read the last ones), staged after
                # the accumulator wait (__init__)
                pre = ld_global_v4_f32(
                    a_gs_base
                    + (Int64(mb * bm) + 4 * cutlass.min(lane, bm // 4 - 1)) * 4
                )
            bar_sync(0, self.epi_threads)
            A = self.acc_stages
            buf = tile_idx % A if A > 1 else 0
            cute.arch.mbarrier_wait(acc_ready + buf, (tile_idx // A) % 2)
            acc = tmem_load(
                tmem_base + buf * self.acc_cols if A > 1 else tmem_base,
                bm,
                1,
                self.ld_rep,
                tidx,
            )
            bar_sync(0, self.epi_threads)
            self._release_acc(acc_free, buf, tidx)
            ag = []
            if cutlass.const_expr(not self.unit_a_gs):
                sc_addr = st.sScales.data_ptr().toint() + warp * (4 * bm)
                if lane < bm // 4:
                    st_shared_v4(sc_addr + lane * 16, pre, Float32)
                cute.arch.sync_warp()
                for c in cutlass.range_constexpr(bm // 4):
                    ag.extend(ld_shared_v4(sc_addr + 16 * c, Float32))
            # One base address per tile and immediate row offsets; token pairs share one cvt.
            base = (
                out_base + (Int64(mb * bm) * self.n + nb * 128 + warp * 32 + lane) * 2
            )
            lim = row_end - mb * bm
            for tp in cutlass.range_constexpr(bm // 2):
                t0, t1 = 2 * tp, 2 * tp + 1
                if cutlass.const_expr(self.unit_a_gs):
                    v0 = acc[t0] * wg
                    v1 = acc[t1] * wg
                else:
                    v0 = acc[t0] * (ag[t0] * wg)
                    v1 = acc[t1] * (ag[t1] * wg)
                h0, h1 = cvt_bf16x2(v0, v1)
                p0 = gmem_bf16(base + t0 * self.n * 2)
                p1 = gmem_bf16(base + t1 * self.n * 2)
                if t0 < lim:
                    p0[0] = h0
                if t1 < lim:
                    p1[0] = h1
            tile = self._next_tile(
                st, tile, tile_idx, grid, num_tiles, self.epi_threads
            )
            if cutlass.const_expr(self.skip_empty):
                tile = self._skip(tile, grid, num_tiles, limits)
            tile_idx += 1

    @cute.jit
    def _fused_rows(
        self,
        tile,
        num_tiles,
        lane,
        a_gs,
        rowtok,
        w_gs,
        experts,
        wv,
        tv,
        gv,
        limits=None,
        tbp=None,
    ):
        """fused_out: the tile's rows 4 l .. 4 l + 3 (lane l; lanes past bm / 4 re-read the last ones): their routing
        weights into wv, their tokens into tv, the tile expert's w_gs into gv (tile clamped to the launch's tiles)."""
        bm = self.bm
        mb = cutlass.max(cutlass.min(tile, num_tiles - 1), 0) // self.nt
        if cutlass.const_expr(self.route_rows):
            w, t = self._route_vals(
                mb, limits[mb] - mb * bm, lane, rowtok, tbp, a_gs, True
            )
            for i in cutlass.range_constexpr(4):
                wv[i] = w[i]
                tv[i] = Uint32(t[i])
        else:
            r = (Int64(mb * bm) + 4 * cutlass.min(lane, bm // 4 - 1)) * 4
            w = ld_global_v4_f32(a_gs.iterator.toint() + r)
            t = ld_global_v4_u32(rowtok.toint() + r)
            for i in cutlass.range_constexpr(4):
                wv[i] = w[i]
                tv[i] = t[i]
        gv[0] = w_gs[experts[mb]]

    @cute.jit
    def _route_vals(
        self, mb, lim, lane, rowtok, tbp, vals, by_route: cutlass.Constexpr
    ):
        """route_rows: for the tile's rows 4 l .. 4 l + 3 (lane l, as above; rows past the tile's routes read its first
        row): vals[route] (by_route; the routing weights) or vals[token] (the row scales), and the tokens."""
        er = cute.make_tensor(
            cute.make_ptr(
                Int32, rowtok.toint(), cute.AddressSpace.gmem, assumed_align=4
            ),
            cute.make_layout(1 << 30),
        )
        tbt = tbp
        base = tbt[mb]
        out, toks = [], []
        for i in cutlass.range_constexpr(4):
            r = 4 * cutlass.min(lane, self.bm // 4 - 1) + i
            route = er[base + Int32(cutlass.select_(r < lim, r, Int32(0)))]
            tok = route // self.route_rows
            toks.append(tok)
            out.append(vals[route] if by_route else vals[tok])
        return out, toks

    @cute.jit
    def _epilogue_fused(
        self,
        st,
        tmem_base,
        warp,
        tidx,
        bidx,
        grid,
        num_tiles,
        a_gs,
        w_gs,
        experts,
        limits,
        out,
        acc_free,
        acc_ready,
        rowtok,
        tbp=None,
    ):
        """fused_out: the weight-major epilogue adding each tile row's output times its routing weight into its token's
        output row (``_fused_out``). Warp 0 loads the rows' weights, tokens and the expert's w_gs one tile ahead
        (before the current tile's adds; the other warps would only repeat it: with route_rows a chain of dependent
        loads per row), and its lanes l < bm / 4 stage them before the accumulator wait in the row-scale buffer: the
        weights times w_gs (4 bm bytes), then the tokens' output row addresses (8 bm bytes; the buffer has 16 bm, so the
        shared memory is the same as without fusion); the barrier after the accumulator load publishes them."""
        NT = self.nt
        bm = self.bm
        lane = tidx % 32
        out_base = out.iterator.toint()
        fa = st.sScales.data_ptr().toint()
        wv = cute.make_rmem_tensor((4,), Float32)
        tv = cute.make_rmem_tensor((4,), Uint32)
        gv = cute.make_rmem_tensor((1,), Float32)
        tile_idx = Int32(0)
        tile = bidx
        if warp == 0:
            self._fused_rows(
                tile,
                num_tiles,
                lane,
                a_gs,
                rowtok,
                w_gs,
                experts,
                wv,
                tv,
                gv,
                limits,
                tbp,
            )
        while tile < num_tiles:
            mb = tile // NT
            nb = tile % NT
            row_end = limits[mb]
            bar_sync(0, self.epi_threads)
            if (warp == 0) & (lane < bm // 4):
                st_shared_v4(fa + lane * 16, [wv[i] * gv[0] for i in range(4)], Float32)
                rb = [Int64(out_base) + Int64(tv[i]) * (self.n * 2) for i in range(4)]
                for h in cutlass.range_constexpr(2):
                    st_shared_v4(
                        fa + 4 * bm + lane * 32 + 16 * h,
                        [
                            Uint32(rb[2 * h + (i // 2)] >> (32 * (i % 2)))
                            for i in range(4)
                        ],
                    )
            A = self.acc_stages
            buf = tile_idx % A if A > 1 else 0
            cute.arch.mbarrier_wait(acc_ready + buf, (tile_idx // A) % 2)
            acc = tmem_load(
                tmem_base + buf * self.acc_cols if A > 1 else tmem_base,
                bm,
                1,
                self.ld_rep,
                tidx,
            )
            bar_sync(0, self.epi_threads)
            self._release_acc(acc_free, buf, tidx)
            nxt = self._next_tile(st, tile, tile_idx, grid, num_tiles, self.epi_threads)
            if warp == 0:
                self._fused_rows(
                    nxt,
                    num_tiles,
                    lane,
                    a_gs,
                    rowtok,
                    w_gs,
                    experts,
                    wv,
                    tv,
                    gv,
                    limits,
                    tbp,
                )
            lim = row_end - mb * bm
            colb = Int64((nb * 128 + warp * 32 + lane - (lane & 1)) * 2)
            if lim >= bm:  # a full tile: no row checks
                self._fused_out(acc, fa, fa + 4 * bm, colb, lane, lim, False)
            else:
                self._fused_out(acc, fa, fa + 4 * bm, colb, lane, lim, True)
            tile = nxt
            tile_idx += 1

    @cute.jit
    def _fused_out(self, acc, sa, ta, colb, lane, lim, check: cutlass.Constexpr):
        """fused_out: add this thread's column of the tile's rows (``check``: rows t < lim only), times the rows' scaled
        routing weights (shared memory at ``sa``), into their tokens' output rows (row addresses at ``ta``; ``colb``:
        this lane pair's column offset in bytes). Lane pair (2j, 2j + 1) and row pair (t, t + 1): the even lane adds
        columns 2j, 2j + 1 of row t, the odd lane those of row t + 1, each with one ``red.global.add.noftz.bf16x2``
        (cute.arch.atomic_add on a bf16x2 vector; a scalar BF16 atomic add compiles to a CAS loop); each
        lane sends its partner the value it needs. Outputs: BF16 atomics in a varying order, not deterministic."""
        odd = lane & 1
        for c in cutlass.range_constexpr(self.bm // 4):
            s4 = ld_shared_v4(sa + 16 * c, Float32)
            for i in cutlass.range_constexpr(2):
                t0 = 4 * c + 2 * i
                v0 = acc[t0] * s4[2 * i]
                v1 = acc[t0 + 1] * s4[2 * i + 1]
                recv = cute.arch.shuffle_sync_bfly(
                    Float32(cutlass.select_(odd == 1, v0, v1)), offset=1
                )
                f = cute.make_rmem_tensor((2,), Float32)
                f[0] = Float32(cutlass.select_(odd == 1, recv, v0))
                f[1] = Float32(cutlass.select_(odd == 1, v1, recv))
                rb = cute.arch.load(smem_ptr(Int64, ta + 8 * t0 + 8 * odd, 8), Int64)
                p = cute.make_ptr(
                    BFloat16, rb + colb, cute.AddressSpace.gmem, assumed_align=4
                )
                if cutlass.const_expr(check):
                    if t0 + odd < lim:
                        cute.arch.atomic_add(p, f.load().to(BFloat16).ir_value())
                else:
                    cute.arch.atomic_add(p, f.load().to(BFloat16).ir_value())

    @cute.jit
    def _epilogue_weight_major_act(
        self,
        st,
        tmem_base,
        warp,
        tidx,
        bidx,
        grid,
        num_tiles,
        a_gs,
        w_gs,
        experts,
        limits,
        out,
        acc_free,
        acc_ready,
        osf,
        rowtok=None,
        tbp=None,
    ):
        """128 gate/up weight rows (interleaved per 8) x bm tokens with SwiGLU (the gate/up at bm <= 64), quantized to
        NVFP4 with block scales only (the values of ``fp4_unit_word``).

        Warp w reads its TMEM lanes 32 w + 16 j (j < 2) with ``tcgen05.ld.16x256b``: thread (i = lane / 4, c = lane % 4)
        holds gate (lane i) and up (lane i + 8) of output 16 w + 8 j + i for tokens 8 r + 2 c + {0, 1} (r < bm / 8), so
        warp w owns the 16-output block w of the tile (64 outputs) for every token: the block max over i (3 butterfly
        shuffles), E2M1 pairs of outputs i, i + 1 (lanes 4 apart), the block's 8 bytes assembled in lane i = 0 (2
        shuffles per half), one 8-byte store per token, and its block scale into the tile's staging block (stored as one
        512-byte block). Chunks of 8 tokens at or past the tile's end are skipped (uniform over the CTA).

        The row scales are loaded before the accumulator wait and staged per warp in shared memory (as in the down GEMM's
        epilogue); with unit_a_gs (per-tensor input) there are none.
        Non-gated (``activation.ReLU2``): the 128 weight rows are 128 outputs; lanes i and i + 8 of TMEM
        lane group j are outputs i and i + 8 of the tile's 16-output block 2 w + j, so each warp quantizes two blocks
        per token (the same shuffles per block), stores 8 bytes per block and token, and the tile's scales fill two
        128x4 blocks (1 KB staging)."""

        NT = self.nt
        bm = self.bm
        K = self.out_cols  # outputs per row (gated: n / 2)
        ET = self.epi_threads
        R = bm // 8
        lane = tidx % 32
        c = lane % 4
        i = lane // 4
        q_base = out.iterator.toint()
        a_gs_base = a_gs.iterator.toint()
        # the tile's block scales are collected here and stored as one 512-byte block (rows >= bm stay 0)
        sfo = st.sSFO.data_ptr().toint()
        osf_base = osf.toint()
        cute.arch.store(smem_ptr(Uint32, sfo + 4 * tidx, 4), Uint32(0))
        if cutlass.const_expr(not self.gated):  # the second 128x4 block
            cute.arch.store(smem_ptr(Uint32, sfo + 512 + 4 * tidx, 4), Uint32(0))
        sc_addr = st.sScales.data_ptr().toint() + warp * (4 * bm)
        tile_idx = Int32(0)
        tile = bidx
        while tile < num_tiles:
            mb = tile // NT
            nb = tile % NT
            expert = experts[mb]
            row_end = limits[mb]
            # lane l: row scales 4 l .. 4 l + 3 of the tile (lanes past bm / 4 re-read the last ones)
            if cutlass.const_expr(self.unit_a_gs):
                pass  # per-tensor input: none
            elif cutlass.const_expr(self.route_rows):
                pre = self._route_vals(
                    mb, row_end - mb * bm, lane, rowtok, tbp, a_gs, False
                )[0]
            else:
                pre = ld_global_v4_f32(
                    a_gs_base
                    + (Int64(mb * bm) + 4 * cutlass.min(lane, bm // 4 - 1)) * 4
                )
            wg = w_gs[expert]
            bar_sync(0, ET)
            A = self.acc_stages
            buf = tile_idx % A if A > 1 else 0
            cute.arch.mbarrier_wait(acc_ready + buf, (tile_idx // A) % 2)
            col = tmem_base + buf * self.acc_cols if A > 1 else tmem_base
            acc = [
                tmem_ld_16x256b(col + ((32 * warp + 16 * j) << 16), R) for j in range(2)
            ]
            cute.arch.fence_view_async_tmem_load()
            bar_sync(0, ET)
            self._release_acc(acc_free, buf, tidx)
            if cutlass.const_expr(not self.unit_a_gs):
                if lane < bm // 4:
                    st_shared_v4(sc_addr + lane * 16, pre, Float32)
                cute.arch.sync_warp()
            lim = row_end - mb * bm
            for r in cutlass.range_constexpr(R):
                if 8 * r < lim:
                    if cutlass.const_expr(not self.unit_a_gs):
                        agv = ld_shared_v2_f32(sc_addr + 4 * (8 * r + 2 * c))
                    for tt in cutlass.range_constexpr(2):
                        tau = 8 * r + 2 * c + tt
                        if cutlass.const_expr(self.unit_a_gs):
                            sg = wg  # (1 * wg)
                        else:
                            sg = agv[tt] * wg
                        if cutlass.const_expr(not self.gated):
                            # outputs i (h 0) and i + 8 (h 1) of block kb = 2 w + j, token tau
                            for j in cutlass.range_constexpr(2):
                                yb = list(
                                    relu2_bf16_pair(
                                        sg, acc[j][4 * r + tt], acc[j][4 * r + 2 + tt]
                                    )
                                )
                                mb_ = cute.arch.fmax(
                                    yb[0], yb[1]
                                )  # ReLU2's outputs are >= 0: a plain max
                                for sh in cutlass.range_constexpr(3):
                                    mb_ = cute.arch.fmax(
                                        mb_,
                                        cute.arch.shuffle_sync_bfly(
                                            mb_, offset=4 << sh
                                        ),
                                    )
                                sfwb, sffb, _ = cvt_e4m3x2(
                                    mb_ * Float32(1.0 / 6.0), Float32(0.0)
                                )
                                invb = sel_if_pos(mb_, cute.arch.rcp_approx(sffb))
                                wb = []
                                sqb = mul_f32x2(yb[0], yb[1], invb, invb)
                                for h in cutlass.range_constexpr(2):
                                    scb = sqb[h]
                                    bb = cvt_e2m1x2(
                                        scb, cute.arch.shuffle_sync_bfly(scb, offset=4)
                                    )
                                    hb = bb | (
                                        cute.arch.shuffle_sync_bfly(bb, offset=8) << 8
                                    )
                                    wb.append(
                                        hb
                                        | (
                                            cute.arch.shuffle_sync_bfly(hb, offset=16)
                                            << 16
                                        )
                                    )
                                if (i == 0) & (tau < lim):
                                    st_global_v2(
                                        q_base
                                        + Int64(mb * bm + tau) * (K // 2)
                                        + nb * 64
                                        + 16 * warp
                                        + 8 * j,
                                        wb[0],
                                        wb[1],
                                    )
                                if (
                                    i == 0
                                ):  # block kb: 128x4 block kb / 4 = w / 2, byte tau % 32 * 16 + tau / 32 * 4 + kb % 4
                                    cute.arch.store(
                                        smem_ptr(
                                            Uint8,
                                            sfo
                                            + (warp // 2) * 512
                                            + tau % 32 * 16
                                            + tau // 32 * 4
                                            + 2 * (warp % 2)
                                            + j,
                                            1,
                                        ),
                                        Uint8(sfwb & 0xFF),
                                    )
                        else:
                            y = []
                            if cutlass.const_expr(self.packed_g):
                                y = list(
                                    swiglu_wm_pair(
                                        sg,
                                        acc[0][4 * r + tt],
                                        acc[1][4 * r + tt],
                                        acc[0][4 * r + 2 + tt],
                                        acc[1][4 * r + 2 + tt],
                                    )
                                )
                            else:
                                for j in cutlass.range_constexpr(2):
                                    o = self._act(
                                        acc[j][4 * r + tt] * sg,
                                        acc[j][4 * r + 2 + tt] * sg,
                                    )
                                    y.append(Float32(o.to(BFloat16)))
                            m = cute.arch.fmax(
                                cute.math.absf(y[0]), cute.math.absf(y[1])
                            )
                            for sh in cutlass.range_constexpr(3):
                                m = cute.arch.fmax(
                                    m,
                                    cute.arch.shuffle_sync_bfly(m, offset=4 << sh),
                                )
                            sfw, sff, _ = cvt_e4m3x2(
                                m * Float32(1.0 / 6.0), Float32(0.0)
                            )
                            inv = sel_if_pos(m, cute.arch.rcp_approx(sff))
                            words = []
                            if cutlass.const_expr(self.packed_g):
                                scg = mul_f32x2(y[0], y[1], inv, inv)
                            for j in cutlass.range_constexpr(2):
                                if cutlass.const_expr(self.packed_g):
                                    sc = scg[j]
                                else:
                                    sc = y[j] * inv
                                byte = cvt_e2m1x2(
                                    sc, cute.arch.shuffle_sync_bfly(sc, offset=4)
                                )  # (i, i + 1) at even i
                                half = byte | (
                                    cute.arch.shuffle_sync_bfly(byte, offset=8) << 8
                                )
                                words.append(
                                    half
                                    | (
                                        cute.arch.shuffle_sync_bfly(half, offset=16)
                                        << 16
                                    )
                                )
                            if (i == 0) & (tau < lim):
                                st_global_v2(
                                    q_base
                                    + Int64(mb * bm + tau) * (K // 2)
                                    + nb * 32
                                    + 8 * warp,
                                    words[0],
                                    words[1],
                                )
                            if (
                                i == 0
                            ):  # block w of row tau in the 128x4 layout: tau % 32 * 16 + tau / 32 * 4 + w
                                cute.arch.store(
                                    smem_ptr(
                                        Uint8,
                                        sfo + tau % 32 * 16 + tau // 32 * 4 + warp,
                                        1,
                                    ),
                                    Uint8(sfw & 0xFF),
                                )
            # the tile's block scales (rows mb * 128 .., K blocks nb * 4 ..): the 16-byte chunks of rows
            # 0 .. min(bm, 32) - 1 of the 128x4 block (full sectors)
            if cutlass.const_expr(
                not self.gated
            ):  # non-gated: the tile's two 128x4 blocks 2 nb, 2 nb + 1
                bar_sync(0, ET)
                if tidx < min(bm, 32) * 4:
                    for sb in cutlass.range_constexpr(2):
                        word = cute.arch.load(
                            smem_ptr(Uint32, sfo + 512 * sb + 4 * tidx, 4), Uint32
                        )
                        st_global(
                            osf_base
                            + Int64(mb) * (K // 64 * 512)
                            + (2 * nb + sb) * 512
                            + 4 * tidx,
                            word,
                            Uint32,
                            4,
                        )
            else:
                bar_sync(0, ET)
                if tidx < min(bm, 32) * 4:
                    word = cute.arch.load(smem_ptr(Uint32, sfo + 4 * tidx, 4), Uint32)
                    st_global(
                        osf_base + Int64(mb) * (K // 64 * 512) + nb * 512 + 4 * tidx,
                        word,
                        Uint32,
                        4,
                    )
            tile = self._next_tile(
                st, tile, tile_idx, grid, num_tiles, self.epi_threads
            )
            tile_idx += 1

    def _act(self, gate, up):
        """The gated activation of scaled gate / up values: the plain SwiGLU's own sequence or ``activation.py``."""
        if self.activation.plain:
            return silu_mul_scaled(gate, up)
        return self.activation.apply(gate, up)


def fp4_unit_word(o, lane):
    """NVFP4 with a unit second-level scale of a warp's 32 consecutive outputs (one per lane; the direct gate/up's
    unit_out), o rounded to BF16 first: the block max of each 16 lanes from two full-warp ``redux.sync.max.abs.f32``,
    sf = e4m3(bmax * (1 / 6)), the values times rcp(sf) to E2M1 pairs (lanes 2j, 2j + 1; 2j in the low nibble), 4 pairs
    to a word in lane 8k (2 butterfly shuffles). Returns (word: lanes 8k hold the FP4 of lanes 8k .. 8k + 7, sfw: the
    block's E4M3 scale in the low byte); the grouped gate/up epilogues compute the same values in their own layouts.
    A plain function (no dynamic control flow): inlined where it is called."""

    y = Float32(o.to(BFloat16))
    # block max: two full-warp redux.sync, one over each half-warp's values (the other half contributes 0)
    lo = lane < 16
    m_lo = cute.arch.warp_redux_sync(
        Float32(cutlass.select_(lo, y, Float32(0.0))), "fmax", abs=True
    )
    m_hi = cute.arch.warp_redux_sync(
        Float32(cutlass.select_(lo, Float32(0.0), y)), "fmax", abs=True
    )
    m = Float32(cutlass.select_(lo, m_lo, m_hi))
    sfw, sff, _ = cvt_e4m3x2(m * Float32(1.0 / 6.0), Float32(0.0))
    inv = sel_if_pos(m, cute.arch.rcp_approx(sff))
    sc = y * inv
    nbr = cute.arch.shuffle_sync_bfly(sc, offset=1)
    even = (lane & 1) == 0
    byte = cvt_e2m1x2(
        Float32(cutlass.select_(even, sc, nbr)), Float32(cutlass.select_(even, nbr, sc))
    )
    half = byte | (cute.arch.shuffle_sync_bfly(byte, offset=2) << 8)
    word = half | (cute.arch.shuffle_sync_bfly(half, offset=4) << 16)
    return word, sfw
