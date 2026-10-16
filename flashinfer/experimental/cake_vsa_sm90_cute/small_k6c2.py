# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generated CuTe DSL source for the Hopper VSA route.  Do not edit."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as cutlass_ir
from cutlass._mlir.dialects import arith as cutlass_arith
from cutlass._mlir.dialects import llvm as cutlass_llvm
from cutlass._mlir.dialects import nvvm as cutlass_nvvm
from cutlass._mlir.dialects import cuda as cutlass_cuda
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives import nvvm_wrapper as prims_nvvm
from cutlass.utils import blackwell_helpers as cutlass_blackwell
from cutlass.utils.layout import LayoutEnum as CutlassLayout
from cutlass.utils import blockscaled_layout as cutlass_blockscaled
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapDataFormat,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)
from cutlass._mlir.dialects.nvvm import CTAGroupKind as DialectCTAGroup
from cutlass._mlir.dialects.nvvm import TMALoadMode as DialectTMALoadMode

NUM_MAIN_STAGES = 1
SMEM_MSTATS_OFF = 214016
SMEM_MSTATS_STAGE_BYTES = 1024
SMEM_MSTATS_STRIDE = 1024
SMEM_PART_LO_OFF = 99328
SMEM_PART_LO_STAGE_BYTES = 16384
SMEM_PART_LO_STRIDE = 16384
SMEM_PART_HI_OFF = 197632
SMEM_PART_HI_STAGE_BYTES = 16384
SMEM_PART_HI_STRIDE = 16384
SMEM_PART_HI0_OFF = 82944
SMEM_PART_HI0_STAGE_BYTES = 16384
SMEM_PART_HI0_STRIDE = 16384
SMEM_RECV_SMEM_OFF = 215040
SMEM_RECV_SMEM_STAGE_BYTES = 17408
SMEM_RECV_SMEM_STRIDE = 17408
SMEM_Q_SMEM_OFF = 1024
SMEM_Q_SMEM_STAGE_BYTES = 16384
SMEM_Q_SMEM_STRIDE = 16384
SMEM_K_SMEM_OFF = 17408
SMEM_K_SMEM_STAGE_BYTES = 16384
SMEM_K_SMEM_STRIDE = 16384
SMEM_VT_SMEM_OFF = 115712
SMEM_VT_SMEM_STAGE_BYTES = 16384
SMEM_VT_SMEM_STRIDE = 16384
SMEM_TOTAL = 232448
THREADS = 256
CAKE_TARGET_ARCH = 'sm_90a'
CAKE_SMEM_BYTES = 232448

def _cake_ldparam_b32(addr, ptx_type):
    return cutlass.Int32(cutlass_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [cutlass.Uint64(addr).ir_value()],
        '{ .reg .u64 %pa; cvta.to.param::entry.u64 %pa, $1; ld.param::entry.' + ptx_type + ' $0, [%pa]; }', '=r,l',
        has_side_effects=False, is_align_stack=False,
        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
    ))

def _cake_ldparam_b64(addr, ptx_type):
    return cutlass.Uint64(cutlass_llvm.inline_asm(
        cutlass.Uint64.mlir_type,
        [cutlass.Uint64(addr).ir_value()],
        '{ .reg .u64 %pa; cvta.to.param::entry.u64 %pa, $1; ld.param::entry.' + ptx_type + ' $0, [%pa]; }', '=l,l',
        has_side_effects=False, is_align_stack=False,
        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
    ))

def _cake_launch_cluster_spread(kernel_launcher, **kwargs):
    block = cutlass_ir.InsertionPoint.current.block
    first_new_op = len(block.operations)
    kernel_launcher.launch(**kwargs)
    launches = [op for op in list(block.operations)[first_new_op:] if op.operation.name == 'cuda.launch_ex']
    if len(launches) != 1:
        raise RuntimeError('expected exactly one typed CUDA launch for cluster scheduling')
    launch = launches[0]
    with cutlass_ir.InsertionPoint(launch):
        cutlass_cuda.launch_cfg_cluster_scheduling_policy(
            launch.operands[0],
            cutlass_cuda.CudaClusterSchedulingPolicy.cudaClusterSchedulingPolicySpread,
        )

@cute.kernel
def kernel_vsa_sm90_bf16_small_k6c2(Q: cutlass.GridConstant[TensorMap], K: cutlass.GridConstant[TensorMap], Vt: cutlass.GridConstant[TensorMap], plan__slot_0: cutlass.Uint64, plan__slot_1: cutlass.Uint64, plan__slot_2: cutlass.Uint64, plan__slot_3: cutlass.Uint64, plan__slot_4: cutlass.Uint64, plan__slot_5: cutlass.Uint64, plan__slot_6: cutlass.Uint64, plan__slot_7: cutlass.Uint64, plan__slot_8: cutlass.Uint64, plan__slot_9: cutlass.Uint64, plan__slot_10: cutlass.Uint64, plan__slot_11: cutlass.Uint64, plan__slot_12: cutlass.Uint64, plan__slot_13: cutlass.Uint64, plan__slot_14: cutlass.Uint64, plan__slot_15: cutlass.Uint64, plan__slot_16: cutlass.Uint64, plan__slot_17: cutlass.Uint64, plan__slot_18: cutlass.Uint64, plan__slot_19: cutlass.Uint64, plan__slot_20: cutlass.Uint64, plan__slot_21: cutlass.Uint64, plan__slot_22: cutlass.Uint64, plan__slot_23: cutlass.Uint64, plan__slot_24: cutlass.Uint64, plan__slot_25: cutlass.Uint64, plan__slot_26: cutlass.Uint64, plan__slot_27: cutlass.Uint64, plan__slot_28: cutlass.Uint64, plan__slot_29: cutlass.Uint64, plan__slot_30: cutlass.Uint64, plan__slot_31: cutlass.Uint64, plan__slot_32: cutlass.Uint64, plan__slot_33: cutlass.Uint64, plan__slot_34: cutlass.Uint64, plan__slot_35: cutlass.Uint64, plan__slot_36: cutlass.Uint64, plan__slot_37: cutlass.Uint64, plan__slot_38: cutlass.Uint64, plan__slot_39: cutlass.Uint64, plan__slot_40: cutlass.Uint64, plan__slot_41: cutlass.Uint64, plan__slot_42: cutlass.Uint64, plan__slot_43: cutlass.Uint64, plan__slot_44: cutlass.Uint64, plan__slot_45: cutlass.Uint64, plan__slot_46: cutlass.Uint64, plan__slot_47: cutlass.Uint64, plan__slot_48: cutlass.Uint64, plan__slot_49: cutlass.Uint64, plan__slot_50: cutlass.Uint64, plan__slot_51: cutlass.Uint64, plan__slot_52: cutlass.Uint64, plan__slot_53: cutlass.Uint64, plan__slot_54: cutlass.Uint64, plan__slot_55: cutlass.Uint64, plan__slot_56: cutlass.Uint64, plan__slot_57: cutlass.Uint64, plan__slot_58: cutlass.Uint64, plan__slot_59: cutlass.Uint64, plan__slot_60: cutlass.Uint64, plan__slot_61: cutlass.Uint64, plan__slot_62: cutlass.Uint64, plan__slot_63: cutlass.Uint64, plan__slot_64: cutlass.Uint64, plan__slot_65: cutlass.Uint64, plan__slot_66: cutlass.Uint64, plan__slot_67: cutlass.Uint64, plan__slot_68: cutlass.Uint64, plan__slot_69: cutlass.Uint64, plan__slot_70: cutlass.Uint64, plan__slot_71: cutlass.Uint64, plan__slot_72: cutlass.Uint64, plan__slot_73: cutlass.Uint64, plan__slot_74: cutlass.Uint64, plan__slot_75: cutlass.Uint64, plan__slot_76: cutlass.Uint64, plan__slot_77: cutlass.Uint64, plan__slot_78: cutlass.Uint64, plan__slot_79: cutlass.Uint64, plan__slot_80: cutlass.Uint64, plan__slot_81: cutlass.Uint64, plan__slot_82: cutlass.Uint64, plan__slot_83: cutlass.Uint64, plan__slot_84: cutlass.Uint64, plan__slot_85: cutlass.Uint64, plan__slot_86: cutlass.Uint64, plan__slot_87: cutlass.Uint64, plan__slot_88: cutlass.Uint64, plan__slot_89: cutlass.Uint64, plan__slot_90: cutlass.Uint64, plan__slot_91: cutlass.Uint64, plan__slot_92: cutlass.Uint64, plan__slot_93: cutlass.Uint64, plan__slot_94: cutlass.Uint64, plan__slot_95: cutlass.Uint64, plan__slot_96: cutlass.Uint64, plan__slot_97: cutlass.Uint64, plan__slot_98: cutlass.Uint64, plan__slot_99: cutlass.Uint64, plan__slot_100: cutlass.Uint64, plan__slot_101: cutlass.Uint64, plan__slot_102: cutlass.Uint64, plan__slot_103: cutlass.Uint64, plan__slot_104: cutlass.Uint64, plan__slot_105: cutlass.Uint64, plan__slot_106: cutlass.Uint64, plan__slot_107: cutlass.Uint64, plan__slot_108: cutlass.Uint64, plan__slot_109: cutlass.Uint64, plan__slot_110: cutlass.Uint64, plan__slot_111: cutlass.Uint64, plan__slot_112: cutlass.Uint64, plan__slot_113: cutlass.Uint64, plan__slot_114: cutlass.Uint64, plan__slot_115: cutlass.Uint64, plan__slot_116: cutlass.Uint64, plan__slot_117: cutlass.Uint64, plan__slot_118: cutlass.Uint64, plan__slot_119: cutlass.Uint64, plan__slot_120: cutlass.Uint64, plan__slot_121: cutlass.Uint64, plan__slot_122: cutlass.Uint64, plan__slot_123: cutlass.Uint64, plan__slot_124: cutlass.Uint64, plan__slot_125: cutlass.Uint64, plan__slot_126: cutlass.Uint64, plan__slot_127: cutlass.Uint64, plan__slot_128: cutlass.Uint64, plan__slot_129: cutlass.Uint64, plan__slot_130: cutlass.Uint64, plan__slot_131: cutlass.Uint64, plan__slot_132: cutlass.Uint64, plan__slot_133: cutlass.Uint64, plan__slot_134: cutlass.Uint64, plan__slot_135: cutlass.Uint64, plan__slot_136: cutlass.Uint64, plan__slot_137: cutlass.Uint64, plan__slot_138: cutlass.Uint64, plan__slot_139: cutlass.Uint64, plan__slot_140: cutlass.Uint64, plan__slot_141: cutlass.Uint64, plan__slot_142: cutlass.Uint64, plan__slot_143: cutlass.Uint64, plan__slot_144: cutlass.Uint64, plan__slot_145: cutlass.Uint64, plan__slot_146: cutlass.Uint64, plan__slot_147: cutlass.Uint64, plan__slot_148: cutlass.Uint64, plan__slot_149: cutlass.Uint64, plan__slot_150: cutlass.Uint64, plan__slot_151: cutlass.Uint64, plan__slot_152: cutlass.Uint64, plan__slot_153: cutlass.Uint64, plan__slot_154: cutlass.Uint64, plan__slot_155: cutlass.Uint64, plan__slot_156: cutlass.Uint64, plan__slot_157: cutlass.Uint64, plan__slot_158: cutlass.Uint64, plan__slot_159: cutlass.Uint64, plan__slot_160: cutlass.Uint64, plan__slot_161: cutlass.Uint64, plan__slot_162: cutlass.Uint64, plan__slot_163: cutlass.Uint64, plan__slot_164: cutlass.Uint64, plan__slot_165: cutlass.Uint64, plan__slot_166: cutlass.Uint64, plan__slot_167: cutlass.Uint64, plan__slot_168: cutlass.Uint64, plan__slot_169: cutlass.Uint64, plan__slot_170: cutlass.Uint64, plan__slot_171: cutlass.Uint64, plan__slot_172: cutlass.Uint64, plan__slot_173: cutlass.Uint64, plan__slot_174: cutlass.Uint64, plan__slot_175: cutlass.Uint64, plan__slot_176: cutlass.Uint64, plan__slot_177: cutlass.Uint64, plan__slot_178: cutlass.Uint64, plan__slot_179: cutlass.Uint64, plan__slot_180: cutlass.Uint64, plan__slot_181: cutlass.Uint64, plan__slot_182: cutlass.Uint64, plan__slot_183: cutlass.Uint64, plan__slot_184: cutlass.Uint64, plan__slot_185: cutlass.Uint64, plan__slot_186: cutlass.Uint64, plan__slot_187: cutlass.Uint64, plan__slot_188: cutlass.Uint64, plan__slot_189: cutlass.Uint64, plan__slot_190: cutlass.Uint64, plan__slot_191: cutlass.Uint64, plan__slot_192: cutlass.Uint64, plan__slot_193: cutlass.Uint64, plan__slot_194: cutlass.Uint64, plan__slot_195: cutlass.Uint64, plan__slot_196: cutlass.Uint64, plan__slot_197: cutlass.Uint64, plan__slot_198: cutlass.Uint64, plan__slot_199: cutlass.Uint64, plan__slot_200: cutlass.Uint64, plan__slot_201: cutlass.Uint64, plan__slot_202: cutlass.Uint64, plan__slot_203: cutlass.Uint64, plan__slot_204: cutlass.Uint64, plan__slot_205: cutlass.Uint64, plan__slot_206: cutlass.Uint64, plan__slot_207: cutlass.Uint64, plan__slot_208: cutlass.Uint64, plan__slot_209: cutlass.Uint64, plan__slot_210: cutlass.Uint64, plan__slot_211: cutlass.Uint64, plan__slot_212: cutlass.Uint64, plan__slot_213: cutlass.Uint64, plan__slot_214: cutlass.Uint64, plan__slot_215: cutlass.Uint64, plan__slot_216: cutlass.Uint64, plan__slot_217: cutlass.Uint64, plan__slot_218: cutlass.Uint64, plan__slot_219: cutlass.Uint64, plan__slot_220: cutlass.Uint64, plan__slot_221: cutlass.Uint64, plan__slot_222: cutlass.Uint64, plan__slot_223: cutlass.Uint64, plan__slot_224: cutlass.Uint64, plan__slot_225: cutlass.Uint64, plan__slot_226: cutlass.Uint64, plan__slot_227: cutlass.Uint64, plan__slot_228: cutlass.Uint64, plan__slot_229: cutlass.Uint64, plan__slot_230: cutlass.Uint64, plan__slot_231: cutlass.Uint64, plan__slot_232: cutlass.Uint64, plan__slot_233: cutlass.Uint64, plan__slot_234: cutlass.Uint64, plan__slot_235: cutlass.Uint64, plan__slot_236: cutlass.Uint64, plan__slot_237: cutlass.Uint64, plan__slot_238: cutlass.Uint64, plan__slot_239: cutlass.Uint64, plan__slot_240: cutlass.Uint64, plan__slot_241: cutlass.Uint64, plan__slot_242: cutlass.Uint64, plan__slot_243: cutlass.Uint64, plan__slot_244: cutlass.Uint64, plan__slot_245: cutlass.Uint64, plan__slot_246: cutlass.Uint64, plan__slot_247: cutlass.Uint64, plan__slot_248: cutlass.Uint64, plan__slot_249: cutlass.Uint64, plan__slot_250: cutlass.Uint64, plan__slot_251: cutlass.Uint64, plan__slot_252: cutlass.Uint64, plan__slot_253: cutlass.Uint64, plan__slot_254: cutlass.Uint64, plan__slot_255: cutlass.Uint64, plan__slot_256: cutlass.Uint64, plan__slot_257: cutlass.Uint64, plan__slot_258: cutlass.Uint64, plan__slot_259: cutlass.Uint64, plan__slot_260: cutlass.Uint64, plan__slot_261: cutlass.Uint64, plan__slot_262: cutlass.Uint64, plan__slot_263: cutlass.Uint64, plan__slot_264: cutlass.Uint64, plan__slot_265: cutlass.Uint64, plan__slot_266: cutlass.Uint64, plan__slot_267: cutlass.Uint64, plan__slot_268: cutlass.Uint64, plan__slot_269: cutlass.Uint64, plan__slot_270: cutlass.Uint64, plan__slot_271: cutlass.Uint64, plan__slot_272: cutlass.Uint64, plan__slot_273: cutlass.Uint64, plan__slot_274: cutlass.Uint64, plan__slot_275: cutlass.Uint64, plan__slot_276: cutlass.Uint64, plan__slot_277: cutlass.Uint64, plan__slot_278: cutlass.Uint64, plan__slot_279: cutlass.Uint64, plan__slot_280: cutlass.Uint64, plan__slot_281: cutlass.Uint64, plan__slot_282: cutlass.Uint64, plan__slot_283: cutlass.Uint64, plan__slot_284: cutlass.Uint64, plan__slot_285: cutlass.Uint64, plan__slot_286: cutlass.Uint64, plan__slot_287: cutlass.Uint64, plan__slot_288: cutlass.Uint64, plan__slot_289: cutlass.Uint64, plan__slot_290: cutlass.Uint64, plan__slot_291: cutlass.Uint64, plan__slot_292: cutlass.Uint64, plan__slot_293: cutlass.Uint64, plan__slot_294: cutlass.Uint64, plan__slot_295: cutlass.Uint64, plan__slot_296: cutlass.Uint64, plan__slot_297: cutlass.Uint64, plan__slot_298: cutlass.Uint64, plan__slot_299: cutlass.Uint64, plan__slot_300: cutlass.Uint64, plan__slot_301: cutlass.Uint64, plan__slot_302: cutlass.Uint64, plan__slot_303: cutlass.Uint64, plan__slot_304: cutlass.Uint64, plan__slot_305: cutlass.Uint64, plan__slot_306: cutlass.Uint64, plan__slot_307: cutlass.Uint64, plan__slot_308: cutlass.Uint64, plan__slot_309: cutlass.Uint64, plan__slot_310: cutlass.Uint64, plan__slot_311: cutlass.Uint64, plan__slot_312: cutlass.Uint64, plan__slot_313: cutlass.Uint64, plan__slot_314: cutlass.Uint64, plan__slot_315: cutlass.Uint64, plan__slot_316: cutlass.Uint64, plan__slot_317: cutlass.Uint64, plan__slot_318: cutlass.Uint64, plan__slot_319: cutlass.Uint64, plan__slot_320: cutlass.Uint64, plan__slot_321: cutlass.Uint64, plan__slot_322: cutlass.Uint64, plan__slot_323: cutlass.Uint64, plan__slot_324: cutlass.Uint64, plan__slot_325: cutlass.Uint64, plan__slot_326: cutlass.Uint64, plan__slot_327: cutlass.Uint64, plan__slot_328: cutlass.Uint64, plan__slot_329: cutlass.Uint64, plan__slot_330: cutlass.Uint64, plan__slot_331: cutlass.Uint64, plan__slot_332: cutlass.Uint64, plan__slot_333: cutlass.Uint64, plan__slot_334: cutlass.Uint64, plan__slot_335: cutlass.Uint64, plan__slot_336: cutlass.Uint64, plan__slot_337: cutlass.Uint64, plan__slot_338: cutlass.Uint64, plan__slot_339: cutlass.Uint64, plan__slot_340: cutlass.Uint64, plan__slot_341: cutlass.Uint64, plan__slot_342: cutlass.Uint64, plan__slot_343: cutlass.Uint64, plan__slot_344: cutlass.Uint64, plan__slot_345: cutlass.Uint64, plan__slot_346: cutlass.Uint64, plan__slot_347: cutlass.Uint64, plan__slot_348: cutlass.Uint64, plan__slot_349: cutlass.Uint64, plan__slot_350: cutlass.Uint64, plan__slot_351: cutlass.Uint64, plan__slot_352: cutlass.Uint64, plan__slot_353: cutlass.Uint64, plan__slot_354: cutlass.Uint64, plan__slot_355: cutlass.Uint64, plan__slot_356: cutlass.Uint64, plan__slot_357: cutlass.Uint64, plan__slot_358: cutlass.Uint64, plan__slot_359: cutlass.Uint64, plan__slot_360: cutlass.Uint64, plan__slot_361: cutlass.Uint64, plan__slot_362: cutlass.Uint64, plan__slot_363: cutlass.Uint64, plan__slot_364: cutlass.Uint64, plan__slot_365: cutlass.Uint64, plan__slot_366: cutlass.Uint64, plan__slot_367: cutlass.Uint64, plan__slot_368: cutlass.Uint64, plan__slot_369: cutlass.Uint64, plan__slot_370: cutlass.Uint64, plan__slot_371: cutlass.Uint64, plan__slot_372: cutlass.Uint64, plan__slot_373: cutlass.Uint64, plan__slot_374: cutlass.Uint64, plan__slot_375: cutlass.Uint64, plan__slot_376: cutlass.Uint64, plan__slot_377: cutlass.Uint64, plan__slot_378: cutlass.Uint64, plan__slot_379: cutlass.Uint64, plan__slot_380: cutlass.Uint64, plan__slot_381: cutlass.Uint64, plan__slot_382: cutlass.Uint64, plan__slot_383: cutlass.Uint64, plan__slot_384: cutlass.Uint64, plan__slot_385: cutlass.Uint64, plan__slot_386: cutlass.Uint64, plan__slot_387: cutlass.Uint64, plan__slot_388: cutlass.Uint64, plan__slot_389: cutlass.Uint64, plan__slot_390: cutlass.Uint64, plan__slot_391: cutlass.Uint64, plan__slot_392: cutlass.Uint64, plan__slot_393: cutlass.Uint64, plan__slot_394: cutlass.Uint64, plan__slot_395: cutlass.Uint64, plan__slot_396: cutlass.Uint64, plan__slot_397: cutlass.Uint64, plan__slot_398: cutlass.Uint64, plan__slot_399: cutlass.Uint64, plan__slot_400: cutlass.Uint64, plan__slot_401: cutlass.Uint64, plan__slot_402: cutlass.Uint64, plan__slot_403: cutlass.Uint64, plan__slot_404: cutlass.Uint64, plan__slot_405: cutlass.Uint64, plan__slot_406: cutlass.Uint64, plan__slot_407: cutlass.Uint64, plan__slot_408: cutlass.Uint64, plan__slot_409: cutlass.Uint64, plan__slot_410: cutlass.Uint64, plan__slot_411: cutlass.Uint64, plan__slot_412: cutlass.Uint64, plan__slot_413: cutlass.Uint64, plan__slot_414: cutlass.Uint64, plan__slot_415: cutlass.Uint64, plan__slot_416: cutlass.Uint64, plan__slot_417: cutlass.Uint64, plan__slot_418: cutlass.Uint64, plan__slot_419: cutlass.Uint64, plan__slot_420: cutlass.Uint64, plan__slot_421: cutlass.Uint64, plan__slot_422: cutlass.Uint64, plan__slot_423: cutlass.Uint64, plan__slot_424: cutlass.Uint64, plan__slot_425: cutlass.Uint64, plan__slot_426: cutlass.Uint64, plan__slot_427: cutlass.Uint64, plan__slot_428: cutlass.Uint64, plan__slot_429: cutlass.Uint64, plan__slot_430: cutlass.Uint64, plan__slot_431: cutlass.Uint64, plan__slot_432: cutlass.Uint64, plan__slot_433: cutlass.Uint64, plan__slot_434: cutlass.Uint64, plan__slot_435: cutlass.Uint64, plan__slot_436: cutlass.Uint64, plan__slot_437: cutlass.Uint64, O: cute.Pointer, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    clusterIdx = (cutlass.Int32(cutlass.Uint32(blockIdx[0]) >> 1), blockIdx[1], blockIdx[2])
    clusterDim = (cutlass.Int32(cutlass.Uint32(gridDim[0]) >> 1), gridDim[1], gridDim[2])
    cluster_id = ((clusterIdx[2] * clusterDim[1] + clusterIdx[1]) * clusterDim[0]) + clusterIdx[0]
    num_clusters = clusterDim[0] * clusterDim[1] * clusterDim[2]
    cta_rank = cute.arch.block_idx_in_cluster()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _O = cute.make_tensor(O, _flat_layout)
    _cake_param_slots_base = cutlass.Uint64(Vt.get_ptr().toint()) + 128
    _plan__base = _cake_param_slots_base + 0
    mstats = cute.recast_ptr(smem_raw + 214016, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _mstats = cute.make_tensor(mstats, _flat_layout)
    mstats_addr = smem + 214016
    part_lo = cute.recast_ptr(smem_raw + 99328, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_lo = cute.make_tensor(part_lo, _flat_layout)
    part_lo_addr = smem + 99328
    part_hi = cute.recast_ptr(smem_raw + 197632, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_hi = cute.make_tensor(part_hi, _flat_layout)
    part_hi_addr = smem + 197632
    part_hi0 = cute.recast_ptr(smem_raw + 82944, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_hi0 = cute.make_tensor(part_hi0, _flat_layout)
    part_hi0_addr = smem + 82944
    recv_smem = cute.recast_ptr(smem_raw + 215040, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _recv_smem = cute.make_tensor(recv_smem, _flat_layout)
    recv_smem_addr = smem + 215040
    q_smem = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _q_smem = cute.make_tensor(q_smem, _flat_layout)
    q_smem_addr = smem + 1024
    k_smem = cute.recast_ptr(smem_raw + 17408, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _k_smem = cute.make_tensor(k_smem, _flat_layout)
    k_smem_addr = smem + 17408
    vt_smem = cute.recast_ptr(smem_raw + 115712, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _vt_smem = cute.make_tensor(vt_smem, _flat_layout)
    vt_smem_addr = smem + 115712
    q_full_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    k_full0_addr = cute.recast_ptr(smem_raw + 8, dtype=cutlass.Uint64)
    k_full1_addr = cute.recast_ptr(smem_raw + 16, dtype=cutlass.Uint64)
    k_full2_addr = cute.recast_ptr(smem_raw + 24, dtype=cutlass.Uint64)
    k_full3_addr = cute.recast_ptr(smem_raw + 32, dtype=cutlass.Uint64)
    k_full4_addr = cute.recast_ptr(smem_raw + 40, dtype=cutlass.Uint64)
    k_full5_addr = cute.recast_ptr(smem_raw + 48, dtype=cutlass.Uint64)
    v_full0_addr = cute.recast_ptr(smem_raw + 56, dtype=cutlass.Uint64)
    v_full1_addr = cute.recast_ptr(smem_raw + 64, dtype=cutlass.Uint64)
    v_full2_addr = cute.recast_ptr(smem_raw + 72, dtype=cutlass.Uint64)
    v_full3_addr = cute.recast_ptr(smem_raw + 80, dtype=cutlass.Uint64)
    v_full4_addr = cute.recast_ptr(smem_raw + 88, dtype=cutlass.Uint64)
    v_full5_addr = cute.recast_ptr(smem_raw + 96, dtype=cutlass.Uint64)
    recv_full_addr = cute.recast_ptr(smem_raw + 104, dtype=cutlass.Uint64)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_full_addr + 0, 1)
            cute.arch.mbarrier_init(k_full0_addr + 0, 1)
            cute.arch.mbarrier_init(k_full1_addr + 0, 1)
            cute.arch.mbarrier_init(k_full2_addr + 0, 1)
            cute.arch.mbarrier_init(k_full3_addr + 0, 1)
            cute.arch.mbarrier_init(k_full4_addr + 0, 1)
            cute.arch.mbarrier_init(k_full5_addr + 0, 1)
            cute.arch.mbarrier_init(v_full0_addr + 0, 1)
            cute.arch.mbarrier_init(v_full1_addr + 0, 1)
            cute.arch.mbarrier_init(v_full2_addr + 0, 1)
            cute.arch.mbarrier_init(v_full3_addr + 0, 1)
            cute.arch.mbarrier_init(v_full4_addr + 0, 1)
            cute.arch.mbarrier_init(v_full5_addr + 0, 1)
            cute.arch.mbarrier_init(recv_full_addr + 0, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    cute.arch.cluster_arrive(aligned=True)
    # barrier.cluster.wait deferred (WarpConfig.cluster_init_wait_warps=()): the schedule's ClusterSyncWait
    lim_even = cute.make_rmem_tensor((1,), cutlass.Int32)
    lim_odd = cute.make_rmem_tensor((1,), cutlass.Int32)
    row_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_q_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_k_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full1_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full1_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full2_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full2_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full3_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full3_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_3 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_3 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full4_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full4_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_4 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_4 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full5_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full5_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_5 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_5 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_recv_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    pslot = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    merged_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    merged_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    fold = cute.make_rmem_tensor((1,), cutlass.Int32)
    fold_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_2 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_3 = cute.make_rmem_tensor((1,), cutlass.Int32)
    merged_max0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    merged_max1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    fold_2 = cute.make_rmem_tensor((1,), cutlass.Int32)
    fold_1_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    cute.arch.prefetch(Q.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(K.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(Vt.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    item = cutlass.Int32(bid)
    mb = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(seqlen_q).ir_value(), cutlass.Int32(64).ir_value())))
    plan_base = cutlass.Int32((item * 8))
    meta = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + (plan_base) * 2, 's16')))
    cnt = cutlass.Int32((meta & 15))
    tile = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((plan_base + 1)) * 2, 's16')))
    blk_base = cutlass.Int32((plan_base + 2))
    head = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(tile).ir_value(), cutlass.Int32(mb).ir_value())))
    qb = cutlass.Int32((tile - (head * mb)))
    q_row = cutlass.Int32(((head * seqlen_q) + (qb * 64)))
    kv_base = cutlass.Int32((head * seqlen_k))
    if (warp == 0):
        if prims.elect_sync():
            if (cta_rank < 2):
                nsets = 2
                recv_tx = ((nsets * 8192) + 512)
                cute.arch.mbarrier_arrive_and_expect_tx(recv_full_addr, recv_tx)
            blk_pre = cute.make_rmem_tensor((6,), cutlass.Int32)
            blk_pre[0] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + (blk_base) * 2, 's16')))
            blk_pre[1] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((blk_base + 1)) * 2, 's16')))
            blk_pre[2] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((blk_base + 2)) * 2, 's16')))
            blk_pre[3] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((blk_base + 3)) * 2, 's16')))
            blk_pre[4] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((blk_base + 4)) * 2, 's16')))
            blk_pre[5] = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_plan__base + ((blk_base + 5)) * 2, 's16')))
            cute.arch.mbarrier_arrive_and_expect_tx(q_full_addr, 16384)
            prims.cp_async_bulk_tensor_shared_cta_global(
                cute.make_ptr(cutlass.Uint8, cutlass.Uint32(q_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                Q.get_ptr(),
                [cutlass.Int32(0), cutlass.Int32(q_row), cutlass.Int32(0)],
                q_full_addr,
                mode=prims.TMALoadMode.TILE,
            )
            if (cnt > 0):
                blk_row = cutlass.Int32((kv_base + (blk_pre[0] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full0_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32(k_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row), cutlass.Int32(0)],
                    k_full0_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full0_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32(vt_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full0_addr,
                    mode=prims.TMALoadMode.TILE,
                )
            if (cnt > 1):
                blk_row_1 = cutlass.Int32((kv_base + (blk_pre[1] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full1_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_1), cutlass.Int32(0)],
                    k_full1_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full1_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_1).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full1_addr,
                    mode=prims.TMALoadMode.TILE,
                )
            if (cnt > 2):
                blk_row_2 = cutlass.Int32((kv_base + (blk_pre[2] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full2_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 32768)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_2), cutlass.Int32(0)],
                    k_full2_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full2_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 32768)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_2).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full2_addr,
                    mode=prims.TMALoadMode.TILE,
                )
            if (cnt > 3):
                blk_row_3 = cutlass.Int32((kv_base + (blk_pre[3] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full3_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 49152)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_3), cutlass.Int32(0)],
                    k_full3_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full3_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 49152)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_3).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full3_addr,
                    mode=prims.TMALoadMode.TILE,
                )
            if (cnt > 4):
                blk_row_4 = cutlass.Int32((kv_base + (blk_pre[4] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full4_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 65536)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_4), cutlass.Int32(0)],
                    k_full4_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full4_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 65536)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_4).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full4_addr,
                    mode=prims.TMALoadMode.TILE,
                )
            if (cnt > 5):
                blk_row_5 = cutlass.Int32((kv_base + (blk_pre[5] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full5_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 81920)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_5), cutlass.Int32(0)],
                    k_full5_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full5_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 81920)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_5).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full5_addr,
                    mode=prims.TMALoadMode.TILE,
                )
    warp_raw = cutlass.Int32(warp)
    _shfl_0 = cute.arch.shuffle_sync(warp_raw, 0, mask=4294967295, mask_and_clamp=31)
    warp_u = cutlass.Int32(_shfl_0)
    wg = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_u).ir_value(), cutlass.Int32(4).ir_value())))
    warp_in_wg = cutlass.Int32((warp_u - (wg * 4)))
    tid_wg = cutlass.Int32(((warp_in_wg * 32) + lane))
    quad = cutlass.Int32(((warp_in_wg * 8) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
    lim_even[0] = cutlass.Int32(0)
    lim_odd[0] = cutlass.Int32(0)
    if (wg == 0):
        lim_even[0] = cutlass.Int32(cnt)
    else:
        lim_odd[0] = cutlass.Int32(cnt)
    m0_local = cutlass.Int32(((warp_in_wg * 16) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
    m1_local = cutlass.Int32((m0_local + 8))
    d_o = cute.make_rmem_tensor((64,), cutlass.Float32)
    d_qk = cute.make_rmem_tensor((32,), cutlass.Float32)
    p_bf16 = cute.make_rmem_tensor((16,), cutlass.Uint32)
    row_max0[0] = cutlass.Float32((0 - float("inf")))
    row_max1[0] = cutlass.Float32((0 - float("inf")))
    row_sum0[0] = cutlass.Float32(0.0)
    row_sum1[0] = cutlass.Float32(0.0)
    if (cnt <= wg):
        d_o[0] = cutlass.Float32(0.0)
        d_o[1] = cutlass.Float32(0.0)
        d_o[2] = cutlass.Float32(0.0)
        d_o[3] = cutlass.Float32(0.0)
        d_o[4] = cutlass.Float32(0.0)
        d_o[5] = cutlass.Float32(0.0)
        d_o[6] = cutlass.Float32(0.0)
        d_o[7] = cutlass.Float32(0.0)
        d_o[8] = cutlass.Float32(0.0)
        d_o[9] = cutlass.Float32(0.0)
        d_o[10] = cutlass.Float32(0.0)
        d_o[11] = cutlass.Float32(0.0)
        d_o[12] = cutlass.Float32(0.0)
        d_o[13] = cutlass.Float32(0.0)
        d_o[14] = cutlass.Float32(0.0)
        d_o[15] = cutlass.Float32(0.0)
        d_o[16] = cutlass.Float32(0.0)
        d_o[17] = cutlass.Float32(0.0)
        d_o[18] = cutlass.Float32(0.0)
        d_o[19] = cutlass.Float32(0.0)
        d_o[20] = cutlass.Float32(0.0)
        d_o[21] = cutlass.Float32(0.0)
        d_o[22] = cutlass.Float32(0.0)
        d_o[23] = cutlass.Float32(0.0)
        d_o[24] = cutlass.Float32(0.0)
        d_o[25] = cutlass.Float32(0.0)
        d_o[26] = cutlass.Float32(0.0)
        d_o[27] = cutlass.Float32(0.0)
        d_o[28] = cutlass.Float32(0.0)
        d_o[29] = cutlass.Float32(0.0)
        d_o[30] = cutlass.Float32(0.0)
        d_o[31] = cutlass.Float32(0.0)
        d_o[32] = cutlass.Float32(0.0)
        d_o[33] = cutlass.Float32(0.0)
        d_o[34] = cutlass.Float32(0.0)
        d_o[35] = cutlass.Float32(0.0)
        d_o[36] = cutlass.Float32(0.0)
        d_o[37] = cutlass.Float32(0.0)
        d_o[38] = cutlass.Float32(0.0)
        d_o[39] = cutlass.Float32(0.0)
        d_o[40] = cutlass.Float32(0.0)
        d_o[41] = cutlass.Float32(0.0)
        d_o[42] = cutlass.Float32(0.0)
        d_o[43] = cutlass.Float32(0.0)
        d_o[44] = cutlass.Float32(0.0)
        d_o[45] = cutlass.Float32(0.0)
        d_o[46] = cutlass.Float32(0.0)
        d_o[47] = cutlass.Float32(0.0)
        d_o[48] = cutlass.Float32(0.0)
        d_o[49] = cutlass.Float32(0.0)
        d_o[50] = cutlass.Float32(0.0)
        d_o[51] = cutlass.Float32(0.0)
        d_o[52] = cutlass.Float32(0.0)
        d_o[53] = cutlass.Float32(0.0)
        d_o[54] = cutlass.Float32(0.0)
        d_o[55] = cutlass.Float32(0.0)
        d_o[56] = cutlass.Float32(0.0)
        d_o[57] = cutlass.Float32(0.0)
        d_o[58] = cutlass.Float32(0.0)
        d_o[59] = cutlass.Float32(0.0)
        d_o[60] = cutlass.Float32(0.0)
        d_o[61] = cutlass.Float32(0.0)
        d_o[62] = cutlass.Float32(0.0)
        d_o[63] = cutlass.Float32(0.0)
    _phase_q_full_0[0] = cutlass.Uint32(0)
    while not prims.mbarrier_wait_parity(q_full_addr, _phase_q_full_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
        pass
    _phase_q_full_0[0] ^= cutlass.Uint32(1)
    _phase_k_full0_0[0] = cutlass.Uint32(0)
    _wgmma_a_0_0_raw = ((cutlass.Uint64(cutlass.Uint32(q_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_a_0_0 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_0_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_0_raw)))
    _wgmma_b_0_1_raw = ((cutlass.Uint64(cutlass.Uint32(k_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_1 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw)))
    _wgmma_a_0_2_raw = ((cutlass.Uint64(cutlass.Uint32((q_smem_addr + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_a_0_2 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_2_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_2_raw)))
    _wgmma_b_0_3_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_3 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw)))
    _phase_v_full0_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_4_raw = ((cutlass.Uint64(cutlass.Uint32(vt_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_4 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw)))
    if (lim_even[0] > 0):
        while not prims.mbarrier_wait_parity(k_full0_addr, _phase_k_full0_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full0_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_0_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_0 = cutlass_llvm.inline_asm(
            _wgmma_0_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_1).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_0, position=[31]))
        _wgmma_1_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_1 = cutlass_llvm.inline_asm(
            _wgmma_1_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_1 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_1, position=[31]))
        _wgmma_2_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_2 = cutlass_llvm.inline_asm(
            _wgmma_2_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_1 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_2, position=[31]))
        _wgmma_3_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_3 = cutlass_llvm.inline_asm(
            _wgmma_3_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_1 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_3, position=[31]))
        _wgmma_4_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_4 = cutlass_llvm.inline_asm(
            _wgmma_4_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_3).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_4, position=[31]))
        _wgmma_5_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_5 = cutlass_llvm.inline_asm(
            _wgmma_5_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_3 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_5, position=[31]))
        _wgmma_6_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_6 = cutlass_llvm.inline_asm(
            _wgmma_6_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_3 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_6, position=[31]))
        _wgmma_7_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_7 = cutlass_llvm.inline_asm(
            _wgmma_7_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_3 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_7, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0[0] = cutlass.Float32((0 - float("inf")))
        new_max1[0] = cutlass.Float32((0 - float("inf")))
        _max_0 = cute.arch.fmax(new_max0[0], d_qk[0], ftz=False)
        new_max0[0] = cutlass.Float32(_max_0)
        _max_1 = cute.arch.fmax(new_max0[0], d_qk[1], ftz=False)
        new_max0[0] = cutlass.Float32(_max_1)
        _max_2 = cute.arch.fmax(new_max0[0], d_qk[4], ftz=False)
        new_max0[0] = cutlass.Float32(_max_2)
        _max_3 = cute.arch.fmax(new_max0[0], d_qk[5], ftz=False)
        new_max0[0] = cutlass.Float32(_max_3)
        _max_4 = cute.arch.fmax(new_max0[0], d_qk[8], ftz=False)
        new_max0[0] = cutlass.Float32(_max_4)
        _max_5 = cute.arch.fmax(new_max0[0], d_qk[9], ftz=False)
        new_max0[0] = cutlass.Float32(_max_5)
        _max_6 = cute.arch.fmax(new_max0[0], d_qk[12], ftz=False)
        new_max0[0] = cutlass.Float32(_max_6)
        _max_7 = cute.arch.fmax(new_max0[0], d_qk[13], ftz=False)
        new_max0[0] = cutlass.Float32(_max_7)
        _max_8 = cute.arch.fmax(new_max0[0], d_qk[16], ftz=False)
        new_max0[0] = cutlass.Float32(_max_8)
        _max_9 = cute.arch.fmax(new_max0[0], d_qk[17], ftz=False)
        new_max0[0] = cutlass.Float32(_max_9)
        _max_10 = cute.arch.fmax(new_max0[0], d_qk[20], ftz=False)
        new_max0[0] = cutlass.Float32(_max_10)
        _max_11 = cute.arch.fmax(new_max0[0], d_qk[21], ftz=False)
        new_max0[0] = cutlass.Float32(_max_11)
        _max_12 = cute.arch.fmax(new_max0[0], d_qk[24], ftz=False)
        new_max0[0] = cutlass.Float32(_max_12)
        _max_13 = cute.arch.fmax(new_max0[0], d_qk[25], ftz=False)
        new_max0[0] = cutlass.Float32(_max_13)
        _max_14 = cute.arch.fmax(new_max0[0], d_qk[28], ftz=False)
        new_max0[0] = cutlass.Float32(_max_14)
        _max_15 = cute.arch.fmax(new_max0[0], d_qk[29], ftz=False)
        new_max0[0] = cutlass.Float32(_max_15)
        _max_16 = cute.arch.fmax(new_max1[0], d_qk[2], ftz=False)
        new_max1[0] = cutlass.Float32(_max_16)
        _max_17 = cute.arch.fmax(new_max1[0], d_qk[3], ftz=False)
        new_max1[0] = cutlass.Float32(_max_17)
        _max_18 = cute.arch.fmax(new_max1[0], d_qk[6], ftz=False)
        new_max1[0] = cutlass.Float32(_max_18)
        _max_19 = cute.arch.fmax(new_max1[0], d_qk[7], ftz=False)
        new_max1[0] = cutlass.Float32(_max_19)
        _max_20 = cute.arch.fmax(new_max1[0], d_qk[10], ftz=False)
        new_max1[0] = cutlass.Float32(_max_20)
        _max_21 = cute.arch.fmax(new_max1[0], d_qk[11], ftz=False)
        new_max1[0] = cutlass.Float32(_max_21)
        _max_22 = cute.arch.fmax(new_max1[0], d_qk[14], ftz=False)
        new_max1[0] = cutlass.Float32(_max_22)
        _max_23 = cute.arch.fmax(new_max1[0], d_qk[15], ftz=False)
        new_max1[0] = cutlass.Float32(_max_23)
        _max_24 = cute.arch.fmax(new_max1[0], d_qk[18], ftz=False)
        new_max1[0] = cutlass.Float32(_max_24)
        _max_25 = cute.arch.fmax(new_max1[0], d_qk[19], ftz=False)
        new_max1[0] = cutlass.Float32(_max_25)
        _max_26 = cute.arch.fmax(new_max1[0], d_qk[22], ftz=False)
        new_max1[0] = cutlass.Float32(_max_26)
        _max_27 = cute.arch.fmax(new_max1[0], d_qk[23], ftz=False)
        new_max1[0] = cutlass.Float32(_max_27)
        _max_28 = cute.arch.fmax(new_max1[0], d_qk[26], ftz=False)
        new_max1[0] = cutlass.Float32(_max_28)
        _max_29 = cute.arch.fmax(new_max1[0], d_qk[27], ftz=False)
        new_max1[0] = cutlass.Float32(_max_29)
        _max_30 = cute.arch.fmax(new_max1[0], d_qk[30], ftz=False)
        new_max1[0] = cutlass.Float32(_max_30)
        _max_31 = cute.arch.fmax(new_max1[0], d_qk[31], ftz=False)
        new_max1[0] = cutlass.Float32(_max_31)
        _shfl_xor_0 = cute.arch.shuffle_sync_bfly(new_max0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_32 = cute.arch.fmax(new_max0[0], _shfl_xor_0, ftz=False)
        new_max0[0] = cutlass.Float32(_max_32)
        _shfl_xor_1 = cute.arch.shuffle_sync_bfly(new_max0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_33 = cute.arch.fmax(new_max0[0], _shfl_xor_1, ftz=False)
        new_max0[0] = cutlass.Float32(_max_33)
        _shfl_xor_2 = cute.arch.shuffle_sync_bfly(new_max1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_34 = cute.arch.fmax(new_max1[0], _shfl_xor_2, ftz=False)
        new_max1[0] = cutlass.Float32(_max_34)
        _shfl_xor_3 = cute.arch.shuffle_sync_bfly(new_max1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_35 = cute.arch.fmax(new_max1[0], _shfl_xor_3, ftz=False)
        new_max1[0] = cutlass.Float32(_max_35)
        row_max0[0] = cutlass.Float32(new_max0[0])
        row_max1[0] = cutlass.Float32(new_max1[0])
        _exp2_2 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_3 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_2)
        d_qk[1] = cutlass.Float32(_exp2_3)
        row_sum0[0] += cutlass.Float32((_exp2_2 + _exp2_3))
        _exp2_4 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_5 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_4)
        d_qk[3] = cutlass.Float32(_exp2_5)
        row_sum1[0] += cutlass.Float32((_exp2_4 + _exp2_5))
        _exp2_6 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_7 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_6)
        d_qk[5] = cutlass.Float32(_exp2_7)
        row_sum0[0] += cutlass.Float32((_exp2_6 + _exp2_7))
        _exp2_8 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_9 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_8)
        d_qk[7] = cutlass.Float32(_exp2_9)
        row_sum1[0] += cutlass.Float32((_exp2_8 + _exp2_9))
        _exp2_10 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_11 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_10)
        d_qk[9] = cutlass.Float32(_exp2_11)
        row_sum0[0] += cutlass.Float32((_exp2_10 + _exp2_11))
        _exp2_12 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_13 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_12)
        d_qk[11] = cutlass.Float32(_exp2_13)
        row_sum1[0] += cutlass.Float32((_exp2_12 + _exp2_13))
        _exp2_14 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_15 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_14)
        d_qk[13] = cutlass.Float32(_exp2_15)
        row_sum0[0] += cutlass.Float32((_exp2_14 + _exp2_15))
        _exp2_16 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_17 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_16)
        d_qk[15] = cutlass.Float32(_exp2_17)
        row_sum1[0] += cutlass.Float32((_exp2_16 + _exp2_17))
        _exp2_18 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_19 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_18)
        d_qk[17] = cutlass.Float32(_exp2_19)
        row_sum0[0] += cutlass.Float32((_exp2_18 + _exp2_19))
        _exp2_20 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_21 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_20)
        d_qk[19] = cutlass.Float32(_exp2_21)
        row_sum1[0] += cutlass.Float32((_exp2_20 + _exp2_21))
        _exp2_22 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_23 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_22)
        d_qk[21] = cutlass.Float32(_exp2_23)
        row_sum0[0] += cutlass.Float32((_exp2_22 + _exp2_23))
        _exp2_24 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_25 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_24)
        d_qk[23] = cutlass.Float32(_exp2_25)
        row_sum1[0] += cutlass.Float32((_exp2_24 + _exp2_25))
        _exp2_26 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_27 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_26)
        d_qk[25] = cutlass.Float32(_exp2_27)
        row_sum0[0] += cutlass.Float32((_exp2_26 + _exp2_27))
        _exp2_28 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_29 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_28)
        d_qk[27] = cutlass.Float32(_exp2_29)
        row_sum1[0] += cutlass.Float32((_exp2_28 + _exp2_29))
        _exp2_30 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_31 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_30)
        d_qk[29] = cutlass.Float32(_exp2_31)
        row_sum0[0] += cutlass.Float32((_exp2_30 + _exp2_31))
        _exp2_32 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_33 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_32)
        d_qk[31] = cutlass.Float32(_exp2_33)
        row_sum1[0] += cutlass.Float32((_exp2_32 + _exp2_33))
        _bf16x2_0 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_0)
        _bf16x2_1 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_1)
        _bf16x2_2 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_2)
        _bf16x2_3 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_3)
        _bf16x2_4 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_4)
        _bf16x2_5 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_5)
        _bf16x2_6 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_6)
        _bf16x2_7 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_7)
        _bf16x2_8 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_8)
        _bf16x2_9 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_9)
        _bf16x2_10 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_10)
        _bf16x2_11 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_11)
        _bf16x2_12 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_12)
        _bf16x2_13 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_13)
        _bf16x2_14 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_14)
        _bf16x2_15 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_15)
        while not prims.mbarrier_wait_parity(v_full0_addr, _phase_v_full0_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full0_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_8_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_8 = cutlass_llvm.inline_asm(
            _wgmma_8_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_4).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((False) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_8, position=[63]))
        _wgmma_9_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_9 = cutlass_llvm.inline_asm(
            _wgmma_9_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_4 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_9, position=[63]))
        _wgmma_10_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_10 = cutlass_llvm.inline_asm(
            _wgmma_10_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_4 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_10, position=[63]))
        _wgmma_11_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_11 = cutlass_llvm.inline_asm(
            _wgmma_11_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_4 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_11, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _phase_k_full1_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_5_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_5 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw)))
    _wgmma_b_0_6_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 16384) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_6 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw)))
    _phase_v_full1_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_7_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_7 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw)))
    if (lim_odd[0] > 1):
        while not prims.mbarrier_wait_parity(k_full1_addr, _phase_k_full1_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full1_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_12_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_12 = cutlass_llvm.inline_asm(
            _wgmma_12_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_5).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_12, position=[31]))
        _wgmma_13_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_13 = cutlass_llvm.inline_asm(
            _wgmma_13_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_5 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_13, position=[31]))
        _wgmma_14_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_14 = cutlass_llvm.inline_asm(
            _wgmma_14_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_5 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[31]))
        _wgmma_15_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_15 = cutlass_llvm.inline_asm(
            _wgmma_15_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_5 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[31]))
        _wgmma_16_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_16 = cutlass_llvm.inline_asm(
            _wgmma_16_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_6).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[31]))
        _wgmma_17_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_17 = cutlass_llvm.inline_asm(
            _wgmma_17_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_6 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[31]))
        _wgmma_18_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_18 = cutlass_llvm.inline_asm(
            _wgmma_18_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_6 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[31]))
        _wgmma_19_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_19 = cutlass_llvm.inline_asm(
            _wgmma_19_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_6 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0_1[0] = cutlass.Float32((0 - float("inf")))
        new_max1_1[0] = cutlass.Float32((0 - float("inf")))
        _max_38 = cute.arch.fmax(new_max0_1[0], d_qk[0], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_38)
        _max_39 = cute.arch.fmax(new_max0_1[0], d_qk[1], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_39)
        _max_40 = cute.arch.fmax(new_max0_1[0], d_qk[4], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_40)
        _max_41 = cute.arch.fmax(new_max0_1[0], d_qk[5], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_41)
        _max_42 = cute.arch.fmax(new_max0_1[0], d_qk[8], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_42)
        _max_43 = cute.arch.fmax(new_max0_1[0], d_qk[9], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_43)
        _max_44 = cute.arch.fmax(new_max0_1[0], d_qk[12], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_44)
        _max_45 = cute.arch.fmax(new_max0_1[0], d_qk[13], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_45)
        _max_46 = cute.arch.fmax(new_max0_1[0], d_qk[16], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_46)
        _max_47 = cute.arch.fmax(new_max0_1[0], d_qk[17], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_47)
        _max_48 = cute.arch.fmax(new_max0_1[0], d_qk[20], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_48)
        _max_49 = cute.arch.fmax(new_max0_1[0], d_qk[21], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_49)
        _max_50 = cute.arch.fmax(new_max0_1[0], d_qk[24], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_50)
        _max_51 = cute.arch.fmax(new_max0_1[0], d_qk[25], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_51)
        _max_52 = cute.arch.fmax(new_max0_1[0], d_qk[28], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_52)
        _max_53 = cute.arch.fmax(new_max0_1[0], d_qk[29], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_53)
        _max_54 = cute.arch.fmax(new_max1_1[0], d_qk[2], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_54)
        _max_55 = cute.arch.fmax(new_max1_1[0], d_qk[3], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_55)
        _max_56 = cute.arch.fmax(new_max1_1[0], d_qk[6], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_56)
        _max_57 = cute.arch.fmax(new_max1_1[0], d_qk[7], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_57)
        _max_58 = cute.arch.fmax(new_max1_1[0], d_qk[10], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_58)
        _max_59 = cute.arch.fmax(new_max1_1[0], d_qk[11], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_59)
        _max_60 = cute.arch.fmax(new_max1_1[0], d_qk[14], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_60)
        _max_61 = cute.arch.fmax(new_max1_1[0], d_qk[15], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_61)
        _max_62 = cute.arch.fmax(new_max1_1[0], d_qk[18], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_62)
        _max_63 = cute.arch.fmax(new_max1_1[0], d_qk[19], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_63)
        _max_64 = cute.arch.fmax(new_max1_1[0], d_qk[22], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_64)
        _max_65 = cute.arch.fmax(new_max1_1[0], d_qk[23], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_65)
        _max_66 = cute.arch.fmax(new_max1_1[0], d_qk[26], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_66)
        _max_67 = cute.arch.fmax(new_max1_1[0], d_qk[27], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_67)
        _max_68 = cute.arch.fmax(new_max1_1[0], d_qk[30], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_68)
        _max_69 = cute.arch.fmax(new_max1_1[0], d_qk[31], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_69)
        _shfl_xor_4 = cute.arch.shuffle_sync_bfly(new_max0_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_70 = cute.arch.fmax(new_max0_1[0], _shfl_xor_4, ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_70)
        _shfl_xor_5 = cute.arch.shuffle_sync_bfly(new_max0_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_71 = cute.arch.fmax(new_max0_1[0], _shfl_xor_5, ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_71)
        _shfl_xor_6 = cute.arch.shuffle_sync_bfly(new_max1_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_72 = cute.arch.fmax(new_max1_1[0], _shfl_xor_6, ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_72)
        _shfl_xor_7 = cute.arch.shuffle_sync_bfly(new_max1_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_73 = cute.arch.fmax(new_max1_1[0], _shfl_xor_7, ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_73)
        _max_74 = cute.arch.fmax(row_max0[0], new_max0_1[0], ftz=False)
        merged_max0_2 = cutlass.Float32(_max_74)
        _max_75 = cute.arch.fmax(row_max1[0], new_max1_1[0], ftz=False)
        merged_max1_2 = cutlass.Float32(_max_75)
        _exp2_34 = cute.math.exp2((row_max0[0] - merged_max0_2), approx=True, ftz=True)
        _exp2_35 = cute.math.exp2((row_max1[0] - merged_max1_2), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_34))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_34))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_34))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_34))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_34))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_34))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_34))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_34))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_34))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_34))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_34))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_34))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_34))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_34))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_34))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_34))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_34))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_34))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_34))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_34))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_34))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_34))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_34))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_34))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_34))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_34))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_34))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_34))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_34))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_34))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_34))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_34))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_35))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_35))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_35))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_35))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_35))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_35))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_35))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_35))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_35))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_35))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_35))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_35))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_35))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_35))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_35))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_35))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_35))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_35))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_35))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_35))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_35))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_35))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_35))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_35))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_35))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_35))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_35))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_35))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_35))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_35))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_35))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_35))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_34))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_35))
        row_max0[0] = cutlass.Float32(merged_max0_2)
        row_max1[0] = cutlass.Float32(merged_max1_2)
        _exp2_36 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_37 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_36)
        d_qk[1] = cutlass.Float32(_exp2_37)
        row_sum0[0] += cutlass.Float32((_exp2_36 + _exp2_37))
        _exp2_38 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_39 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_38)
        d_qk[3] = cutlass.Float32(_exp2_39)
        row_sum1[0] += cutlass.Float32((_exp2_38 + _exp2_39))
        _exp2_40 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_41 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_40)
        d_qk[5] = cutlass.Float32(_exp2_41)
        row_sum0[0] += cutlass.Float32((_exp2_40 + _exp2_41))
        _exp2_42 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_43 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_42)
        d_qk[7] = cutlass.Float32(_exp2_43)
        row_sum1[0] += cutlass.Float32((_exp2_42 + _exp2_43))
        _exp2_44 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_45 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_44)
        d_qk[9] = cutlass.Float32(_exp2_45)
        row_sum0[0] += cutlass.Float32((_exp2_44 + _exp2_45))
        _exp2_46 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_47 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_46)
        d_qk[11] = cutlass.Float32(_exp2_47)
        row_sum1[0] += cutlass.Float32((_exp2_46 + _exp2_47))
        _exp2_48 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_49 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_48)
        d_qk[13] = cutlass.Float32(_exp2_49)
        row_sum0[0] += cutlass.Float32((_exp2_48 + _exp2_49))
        _exp2_50 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_51 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_50)
        d_qk[15] = cutlass.Float32(_exp2_51)
        row_sum1[0] += cutlass.Float32((_exp2_50 + _exp2_51))
        _exp2_52 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_53 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_52)
        d_qk[17] = cutlass.Float32(_exp2_53)
        row_sum0[0] += cutlass.Float32((_exp2_52 + _exp2_53))
        _exp2_54 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_55 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_54)
        d_qk[19] = cutlass.Float32(_exp2_55)
        row_sum1[0] += cutlass.Float32((_exp2_54 + _exp2_55))
        _exp2_56 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_57 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_56)
        d_qk[21] = cutlass.Float32(_exp2_57)
        row_sum0[0] += cutlass.Float32((_exp2_56 + _exp2_57))
        _exp2_58 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_59 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_58)
        d_qk[23] = cutlass.Float32(_exp2_59)
        row_sum1[0] += cutlass.Float32((_exp2_58 + _exp2_59))
        _exp2_60 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_61 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_60)
        d_qk[25] = cutlass.Float32(_exp2_61)
        row_sum0[0] += cutlass.Float32((_exp2_60 + _exp2_61))
        _exp2_62 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_63 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_62)
        d_qk[27] = cutlass.Float32(_exp2_63)
        row_sum1[0] += cutlass.Float32((_exp2_62 + _exp2_63))
        _exp2_64 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_65 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_64)
        d_qk[29] = cutlass.Float32(_exp2_65)
        row_sum0[0] += cutlass.Float32((_exp2_64 + _exp2_65))
        _exp2_66 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_67 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_66)
        d_qk[31] = cutlass.Float32(_exp2_67)
        row_sum1[0] += cutlass.Float32((_exp2_66 + _exp2_67))
        _bf16x2_16 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_16)
        _bf16x2_17 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_17)
        _bf16x2_18 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_18)
        _bf16x2_19 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_19)
        _bf16x2_20 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_20)
        _bf16x2_21 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_21)
        _bf16x2_22 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_22)
        _bf16x2_23 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_23)
        _bf16x2_24 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_24)
        _bf16x2_25 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_25)
        _bf16x2_26 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_26)
        _bf16x2_27 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_27)
        _bf16x2_28 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_28)
        _bf16x2_29 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_29)
        _bf16x2_30 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_30)
        _bf16x2_31 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_31)
        while not prims.mbarrier_wait_parity(v_full1_addr, _phase_v_full1_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full1_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_20_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_20 = cutlass_llvm.inline_asm(
            _wgmma_20_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_7).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[63]))
        _wgmma_21_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_21 = cutlass_llvm.inline_asm(
            _wgmma_21_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_7 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[63]))
        _wgmma_22_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_22 = cutlass_llvm.inline_asm(
            _wgmma_22_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_7 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[63]))
        _wgmma_23_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_23 = cutlass_llvm.inline_asm(
            _wgmma_23_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_7 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _phase_k_full2_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_8_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 32768)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_8 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_8_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_8_raw)))
    _wgmma_b_0_9_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 32768) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_9 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_9_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_9_raw)))
    _phase_v_full2_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_10_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 32768)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_10 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_10_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_10_raw)))
    if (lim_even[0] > 2):
        while not prims.mbarrier_wait_parity(k_full2_addr, _phase_k_full2_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full2_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_24_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_24 = cutlass_llvm.inline_asm(
            _wgmma_24_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_8).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[31]))
        _wgmma_25_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_25 = cutlass_llvm.inline_asm(
            _wgmma_25_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_8 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[31]))
        _wgmma_26_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_26 = cutlass_llvm.inline_asm(
            _wgmma_26_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_8 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[31]))
        _wgmma_27_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_27 = cutlass_llvm.inline_asm(
            _wgmma_27_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_8 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[31]))
        _wgmma_28_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_28 = cutlass_llvm.inline_asm(
            _wgmma_28_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_9).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[31]))
        _wgmma_29_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_29 = cutlass_llvm.inline_asm(
            _wgmma_29_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_9 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[31]))
        _wgmma_30_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_30 = cutlass_llvm.inline_asm(
            _wgmma_30_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_9 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[31]))
        _wgmma_31_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_31 = cutlass_llvm.inline_asm(
            _wgmma_31_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_9 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0_2[0] = cutlass.Float32((0 - float("inf")))
        new_max1_2[0] = cutlass.Float32((0 - float("inf")))
        _max_76 = cute.arch.fmax(new_max0_2[0], d_qk[0], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_76)
        _max_77 = cute.arch.fmax(new_max0_2[0], d_qk[1], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_77)
        _max_78 = cute.arch.fmax(new_max0_2[0], d_qk[4], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_78)
        _max_79 = cute.arch.fmax(new_max0_2[0], d_qk[5], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_79)
        _max_80 = cute.arch.fmax(new_max0_2[0], d_qk[8], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_80)
        _max_81 = cute.arch.fmax(new_max0_2[0], d_qk[9], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_81)
        _max_82 = cute.arch.fmax(new_max0_2[0], d_qk[12], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_82)
        _max_83 = cute.arch.fmax(new_max0_2[0], d_qk[13], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_83)
        _max_84 = cute.arch.fmax(new_max0_2[0], d_qk[16], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_84)
        _max_85 = cute.arch.fmax(new_max0_2[0], d_qk[17], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_85)
        _max_86 = cute.arch.fmax(new_max0_2[0], d_qk[20], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_86)
        _max_87 = cute.arch.fmax(new_max0_2[0], d_qk[21], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_87)
        _max_88 = cute.arch.fmax(new_max0_2[0], d_qk[24], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_88)
        _max_89 = cute.arch.fmax(new_max0_2[0], d_qk[25], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_89)
        _max_90 = cute.arch.fmax(new_max0_2[0], d_qk[28], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_90)
        _max_91 = cute.arch.fmax(new_max0_2[0], d_qk[29], ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_91)
        _max_92 = cute.arch.fmax(new_max1_2[0], d_qk[2], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_92)
        _max_93 = cute.arch.fmax(new_max1_2[0], d_qk[3], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_93)
        _max_94 = cute.arch.fmax(new_max1_2[0], d_qk[6], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_94)
        _max_95 = cute.arch.fmax(new_max1_2[0], d_qk[7], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_95)
        _max_96 = cute.arch.fmax(new_max1_2[0], d_qk[10], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_96)
        _max_97 = cute.arch.fmax(new_max1_2[0], d_qk[11], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_97)
        _max_98 = cute.arch.fmax(new_max1_2[0], d_qk[14], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_98)
        _max_99 = cute.arch.fmax(new_max1_2[0], d_qk[15], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_99)
        _max_100 = cute.arch.fmax(new_max1_2[0], d_qk[18], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_100)
        _max_101 = cute.arch.fmax(new_max1_2[0], d_qk[19], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_101)
        _max_102 = cute.arch.fmax(new_max1_2[0], d_qk[22], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_102)
        _max_103 = cute.arch.fmax(new_max1_2[0], d_qk[23], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_103)
        _max_104 = cute.arch.fmax(new_max1_2[0], d_qk[26], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_104)
        _max_105 = cute.arch.fmax(new_max1_2[0], d_qk[27], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_105)
        _max_106 = cute.arch.fmax(new_max1_2[0], d_qk[30], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_106)
        _max_107 = cute.arch.fmax(new_max1_2[0], d_qk[31], ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_107)
        _shfl_xor_8 = cute.arch.shuffle_sync_bfly(new_max0_2[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_108 = cute.arch.fmax(new_max0_2[0], _shfl_xor_8, ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_108)
        _shfl_xor_9 = cute.arch.shuffle_sync_bfly(new_max0_2[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_109 = cute.arch.fmax(new_max0_2[0], _shfl_xor_9, ftz=False)
        new_max0_2[0] = cutlass.Float32(_max_109)
        _shfl_xor_10 = cute.arch.shuffle_sync_bfly(new_max1_2[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_110 = cute.arch.fmax(new_max1_2[0], _shfl_xor_10, ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_110)
        _shfl_xor_11 = cute.arch.shuffle_sync_bfly(new_max1_2[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_111 = cute.arch.fmax(new_max1_2[0], _shfl_xor_11, ftz=False)
        new_max1_2[0] = cutlass.Float32(_max_111)
        _max_112 = cute.arch.fmax(row_max0[0], new_max0_2[0], ftz=False)
        merged_max0_3 = cutlass.Float32(_max_112)
        _max_113 = cute.arch.fmax(row_max1[0], new_max1_2[0], ftz=False)
        merged_max1_3 = cutlass.Float32(_max_113)
        _exp2_68 = cute.math.exp2((row_max0[0] - merged_max0_3), approx=True, ftz=True)
        _exp2_69 = cute.math.exp2((row_max1[0] - merged_max1_3), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_68))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_68))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_68))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_68))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_68))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_68))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_68))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_68))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_68))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_68))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_68))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_68))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_68))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_68))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_68))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_68))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_68))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_68))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_68))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_68))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_68))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_68))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_68))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_68))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_68))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_68))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_68))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_68))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_68))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_68))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_68))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_68))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_69))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_69))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_69))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_69))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_69))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_69))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_69))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_69))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_69))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_69))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_69))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_69))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_69))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_69))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_69))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_69))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_69))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_69))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_69))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_69))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_69))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_69))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_69))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_69))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_69))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_69))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_69))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_69))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_69))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_69))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_69))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_69))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_68))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_69))
        row_max0[0] = cutlass.Float32(merged_max0_3)
        row_max1[0] = cutlass.Float32(merged_max1_3)
        _exp2_70 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_71 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_70)
        d_qk[1] = cutlass.Float32(_exp2_71)
        row_sum0[0] += cutlass.Float32((_exp2_70 + _exp2_71))
        _exp2_72 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_73 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_72)
        d_qk[3] = cutlass.Float32(_exp2_73)
        row_sum1[0] += cutlass.Float32((_exp2_72 + _exp2_73))
        _exp2_74 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_75 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_74)
        d_qk[5] = cutlass.Float32(_exp2_75)
        row_sum0[0] += cutlass.Float32((_exp2_74 + _exp2_75))
        _exp2_76 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_77 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_76)
        d_qk[7] = cutlass.Float32(_exp2_77)
        row_sum1[0] += cutlass.Float32((_exp2_76 + _exp2_77))
        _exp2_78 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_79 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_78)
        d_qk[9] = cutlass.Float32(_exp2_79)
        row_sum0[0] += cutlass.Float32((_exp2_78 + _exp2_79))
        _exp2_80 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_81 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_80)
        d_qk[11] = cutlass.Float32(_exp2_81)
        row_sum1[0] += cutlass.Float32((_exp2_80 + _exp2_81))
        _exp2_82 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_83 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_82)
        d_qk[13] = cutlass.Float32(_exp2_83)
        row_sum0[0] += cutlass.Float32((_exp2_82 + _exp2_83))
        _exp2_84 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_85 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_84)
        d_qk[15] = cutlass.Float32(_exp2_85)
        row_sum1[0] += cutlass.Float32((_exp2_84 + _exp2_85))
        _exp2_86 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_87 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_86)
        d_qk[17] = cutlass.Float32(_exp2_87)
        row_sum0[0] += cutlass.Float32((_exp2_86 + _exp2_87))
        _exp2_88 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_89 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_88)
        d_qk[19] = cutlass.Float32(_exp2_89)
        row_sum1[0] += cutlass.Float32((_exp2_88 + _exp2_89))
        _exp2_90 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_91 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_90)
        d_qk[21] = cutlass.Float32(_exp2_91)
        row_sum0[0] += cutlass.Float32((_exp2_90 + _exp2_91))
        _exp2_92 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_93 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_92)
        d_qk[23] = cutlass.Float32(_exp2_93)
        row_sum1[0] += cutlass.Float32((_exp2_92 + _exp2_93))
        _exp2_94 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_95 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_94)
        d_qk[25] = cutlass.Float32(_exp2_95)
        row_sum0[0] += cutlass.Float32((_exp2_94 + _exp2_95))
        _exp2_96 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_97 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_96)
        d_qk[27] = cutlass.Float32(_exp2_97)
        row_sum1[0] += cutlass.Float32((_exp2_96 + _exp2_97))
        _exp2_98 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_99 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_98)
        d_qk[29] = cutlass.Float32(_exp2_99)
        row_sum0[0] += cutlass.Float32((_exp2_98 + _exp2_99))
        _exp2_100 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_101 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_100)
        d_qk[31] = cutlass.Float32(_exp2_101)
        row_sum1[0] += cutlass.Float32((_exp2_100 + _exp2_101))
        _bf16x2_32 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_32)
        _bf16x2_33 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_33)
        _bf16x2_34 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_34)
        _bf16x2_35 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_35)
        _bf16x2_36 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_36)
        _bf16x2_37 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_37)
        _bf16x2_38 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_38)
        _bf16x2_39 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_39)
        _bf16x2_40 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_40)
        _bf16x2_41 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_41)
        _bf16x2_42 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_42)
        _bf16x2_43 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_43)
        _bf16x2_44 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_44)
        _bf16x2_45 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_45)
        _bf16x2_46 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_46)
        _bf16x2_47 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_47)
        while not prims.mbarrier_wait_parity(v_full2_addr, _phase_v_full2_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full2_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_32_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_32 = cutlass_llvm.inline_asm(
            _wgmma_32_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_10).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_32, position=[63]))
        _wgmma_33_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_33 = cutlass_llvm.inline_asm(
            _wgmma_33_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_10 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_33, position=[63]))
        _wgmma_34_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_34 = cutlass_llvm.inline_asm(
            _wgmma_34_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_10 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_34, position=[63]))
        _wgmma_35_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_35 = cutlass_llvm.inline_asm(
            _wgmma_35_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_10 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_35, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _phase_k_full3_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_11_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 49152)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_11 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_11_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_11_raw)))
    _wgmma_b_0_12_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 49152) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_12 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_12_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_12_raw)))
    _phase_v_full3_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_13_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 49152)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_13 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_13_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_13_raw)))
    if (lim_odd[0] > 3):
        while not prims.mbarrier_wait_parity(k_full3_addr, _phase_k_full3_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full3_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_36_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_36 = cutlass_llvm.inline_asm(
            _wgmma_36_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_11).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[31]))
        _wgmma_37_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_37 = cutlass_llvm.inline_asm(
            _wgmma_37_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_11 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[31]))
        _wgmma_38_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_38 = cutlass_llvm.inline_asm(
            _wgmma_38_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_11 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[31]))
        _wgmma_39_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_39 = cutlass_llvm.inline_asm(
            _wgmma_39_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_11 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[31]))
        _wgmma_40_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_40 = cutlass_llvm.inline_asm(
            _wgmma_40_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_12).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[31]))
        _wgmma_41_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_41 = cutlass_llvm.inline_asm(
            _wgmma_41_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_12 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[31]))
        _wgmma_42_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_42 = cutlass_llvm.inline_asm(
            _wgmma_42_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_12 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[31]))
        _wgmma_43_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_43 = cutlass_llvm.inline_asm(
            _wgmma_43_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_12 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0_3[0] = cutlass.Float32((0 - float("inf")))
        new_max1_3[0] = cutlass.Float32((0 - float("inf")))
        _max_114 = cute.arch.fmax(new_max0_3[0], d_qk[0], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_114)
        _max_115 = cute.arch.fmax(new_max0_3[0], d_qk[1], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_115)
        _max_116 = cute.arch.fmax(new_max0_3[0], d_qk[4], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_116)
        _max_117 = cute.arch.fmax(new_max0_3[0], d_qk[5], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_117)
        _max_118 = cute.arch.fmax(new_max0_3[0], d_qk[8], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_118)
        _max_119 = cute.arch.fmax(new_max0_3[0], d_qk[9], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_119)
        _max_120 = cute.arch.fmax(new_max0_3[0], d_qk[12], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_120)
        _max_121 = cute.arch.fmax(new_max0_3[0], d_qk[13], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_121)
        _max_122 = cute.arch.fmax(new_max0_3[0], d_qk[16], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_122)
        _max_123 = cute.arch.fmax(new_max0_3[0], d_qk[17], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_123)
        _max_124 = cute.arch.fmax(new_max0_3[0], d_qk[20], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_124)
        _max_125 = cute.arch.fmax(new_max0_3[0], d_qk[21], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_125)
        _max_126 = cute.arch.fmax(new_max0_3[0], d_qk[24], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_126)
        _max_127 = cute.arch.fmax(new_max0_3[0], d_qk[25], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_127)
        _max_128 = cute.arch.fmax(new_max0_3[0], d_qk[28], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_128)
        _max_129 = cute.arch.fmax(new_max0_3[0], d_qk[29], ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_129)
        _max_130 = cute.arch.fmax(new_max1_3[0], d_qk[2], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_130)
        _max_131 = cute.arch.fmax(new_max1_3[0], d_qk[3], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_131)
        _max_132 = cute.arch.fmax(new_max1_3[0], d_qk[6], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_132)
        _max_133 = cute.arch.fmax(new_max1_3[0], d_qk[7], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_133)
        _max_134 = cute.arch.fmax(new_max1_3[0], d_qk[10], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_134)
        _max_135 = cute.arch.fmax(new_max1_3[0], d_qk[11], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_135)
        _max_136 = cute.arch.fmax(new_max1_3[0], d_qk[14], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_136)
        _max_137 = cute.arch.fmax(new_max1_3[0], d_qk[15], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_137)
        _max_138 = cute.arch.fmax(new_max1_3[0], d_qk[18], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_138)
        _max_139 = cute.arch.fmax(new_max1_3[0], d_qk[19], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_139)
        _max_140 = cute.arch.fmax(new_max1_3[0], d_qk[22], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_140)
        _max_141 = cute.arch.fmax(new_max1_3[0], d_qk[23], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_141)
        _max_142 = cute.arch.fmax(new_max1_3[0], d_qk[26], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_142)
        _max_143 = cute.arch.fmax(new_max1_3[0], d_qk[27], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_143)
        _max_144 = cute.arch.fmax(new_max1_3[0], d_qk[30], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_144)
        _max_145 = cute.arch.fmax(new_max1_3[0], d_qk[31], ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_145)
        _shfl_xor_12 = cute.arch.shuffle_sync_bfly(new_max0_3[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_146 = cute.arch.fmax(new_max0_3[0], _shfl_xor_12, ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_146)
        _shfl_xor_13 = cute.arch.shuffle_sync_bfly(new_max0_3[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_147 = cute.arch.fmax(new_max0_3[0], _shfl_xor_13, ftz=False)
        new_max0_3[0] = cutlass.Float32(_max_147)
        _shfl_xor_14 = cute.arch.shuffle_sync_bfly(new_max1_3[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_148 = cute.arch.fmax(new_max1_3[0], _shfl_xor_14, ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_148)
        _shfl_xor_15 = cute.arch.shuffle_sync_bfly(new_max1_3[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_149 = cute.arch.fmax(new_max1_3[0], _shfl_xor_15, ftz=False)
        new_max1_3[0] = cutlass.Float32(_max_149)
        _max_150 = cute.arch.fmax(row_max0[0], new_max0_3[0], ftz=False)
        merged_max0_4 = cutlass.Float32(_max_150)
        _max_151 = cute.arch.fmax(row_max1[0], new_max1_3[0], ftz=False)
        merged_max1_4 = cutlass.Float32(_max_151)
        _exp2_102 = cute.math.exp2((row_max0[0] - merged_max0_4), approx=True, ftz=True)
        _exp2_103 = cute.math.exp2((row_max1[0] - merged_max1_4), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_102))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_102))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_102))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_102))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_102))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_102))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_102))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_102))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_102))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_102))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_102))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_102))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_102))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_102))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_102))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_102))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_102))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_102))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_102))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_102))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_102))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_102))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_102))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_102))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_102))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_102))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_102))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_102))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_102))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_102))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_102))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_102))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_103))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_103))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_103))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_103))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_103))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_103))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_103))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_103))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_103))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_103))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_103))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_103))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_103))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_103))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_103))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_103))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_103))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_103))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_103))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_103))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_103))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_103))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_103))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_103))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_103))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_103))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_103))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_103))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_103))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_103))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_103))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_103))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_102))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_103))
        row_max0[0] = cutlass.Float32(merged_max0_4)
        row_max1[0] = cutlass.Float32(merged_max1_4)
        _exp2_104 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_105 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_104)
        d_qk[1] = cutlass.Float32(_exp2_105)
        row_sum0[0] += cutlass.Float32((_exp2_104 + _exp2_105))
        _exp2_106 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_107 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_106)
        d_qk[3] = cutlass.Float32(_exp2_107)
        row_sum1[0] += cutlass.Float32((_exp2_106 + _exp2_107))
        _exp2_108 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_109 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_108)
        d_qk[5] = cutlass.Float32(_exp2_109)
        row_sum0[0] += cutlass.Float32((_exp2_108 + _exp2_109))
        _exp2_110 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_111 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_110)
        d_qk[7] = cutlass.Float32(_exp2_111)
        row_sum1[0] += cutlass.Float32((_exp2_110 + _exp2_111))
        _exp2_112 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_113 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_112)
        d_qk[9] = cutlass.Float32(_exp2_113)
        row_sum0[0] += cutlass.Float32((_exp2_112 + _exp2_113))
        _exp2_114 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_115 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_114)
        d_qk[11] = cutlass.Float32(_exp2_115)
        row_sum1[0] += cutlass.Float32((_exp2_114 + _exp2_115))
        _exp2_116 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_117 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_116)
        d_qk[13] = cutlass.Float32(_exp2_117)
        row_sum0[0] += cutlass.Float32((_exp2_116 + _exp2_117))
        _exp2_118 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_119 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_118)
        d_qk[15] = cutlass.Float32(_exp2_119)
        row_sum1[0] += cutlass.Float32((_exp2_118 + _exp2_119))
        _exp2_120 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_121 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_120)
        d_qk[17] = cutlass.Float32(_exp2_121)
        row_sum0[0] += cutlass.Float32((_exp2_120 + _exp2_121))
        _exp2_122 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_123 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_122)
        d_qk[19] = cutlass.Float32(_exp2_123)
        row_sum1[0] += cutlass.Float32((_exp2_122 + _exp2_123))
        _exp2_124 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_125 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_124)
        d_qk[21] = cutlass.Float32(_exp2_125)
        row_sum0[0] += cutlass.Float32((_exp2_124 + _exp2_125))
        _exp2_126 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_127 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_126)
        d_qk[23] = cutlass.Float32(_exp2_127)
        row_sum1[0] += cutlass.Float32((_exp2_126 + _exp2_127))
        _exp2_128 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_129 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_128)
        d_qk[25] = cutlass.Float32(_exp2_129)
        row_sum0[0] += cutlass.Float32((_exp2_128 + _exp2_129))
        _exp2_130 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_131 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_130)
        d_qk[27] = cutlass.Float32(_exp2_131)
        row_sum1[0] += cutlass.Float32((_exp2_130 + _exp2_131))
        _exp2_132 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_133 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_132)
        d_qk[29] = cutlass.Float32(_exp2_133)
        row_sum0[0] += cutlass.Float32((_exp2_132 + _exp2_133))
        _exp2_134 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_135 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_134)
        d_qk[31] = cutlass.Float32(_exp2_135)
        row_sum1[0] += cutlass.Float32((_exp2_134 + _exp2_135))
        _bf16x2_48 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_48)
        _bf16x2_49 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_49)
        _bf16x2_50 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_50)
        _bf16x2_51 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_51)
        _bf16x2_52 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_52)
        _bf16x2_53 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_53)
        _bf16x2_54 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_54)
        _bf16x2_55 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_55)
        _bf16x2_56 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_56)
        _bf16x2_57 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_57)
        _bf16x2_58 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_58)
        _bf16x2_59 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_59)
        _bf16x2_60 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_60)
        _bf16x2_61 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_61)
        _bf16x2_62 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_62)
        _bf16x2_63 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_63)
        while not prims.mbarrier_wait_parity(v_full3_addr, _phase_v_full3_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full3_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_44_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_44 = cutlass_llvm.inline_asm(
            _wgmma_44_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_13).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_44, position=[63]))
        _wgmma_45_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_45 = cutlass_llvm.inline_asm(
            _wgmma_45_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_13 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_45, position=[63]))
        _wgmma_46_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_46 = cutlass_llvm.inline_asm(
            _wgmma_46_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_13 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_46, position=[63]))
        _wgmma_47_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_47 = cutlass_llvm.inline_asm(
            _wgmma_47_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_13 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_47, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _phase_k_full4_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_14_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 65536)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_14 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_14_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_14_raw)))
    _wgmma_b_0_15_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 65536) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_15 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_15_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_15_raw)))
    _phase_v_full4_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_16_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 65536)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_16 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_16_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_16_raw)))
    if (lim_even[0] > 4):
        while not prims.mbarrier_wait_parity(k_full4_addr, _phase_k_full4_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full4_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_48_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_48 = cutlass_llvm.inline_asm(
            _wgmma_48_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_14).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_48, position=[31]))
        _wgmma_49_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_49 = cutlass_llvm.inline_asm(
            _wgmma_49_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_14 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_49, position=[31]))
        _wgmma_50_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_50 = cutlass_llvm.inline_asm(
            _wgmma_50_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_14 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_50, position=[31]))
        _wgmma_51_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_51 = cutlass_llvm.inline_asm(
            _wgmma_51_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_14 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_51, position=[31]))
        _wgmma_52_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_52 = cutlass_llvm.inline_asm(
            _wgmma_52_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_15).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_52, position=[31]))
        _wgmma_53_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_53 = cutlass_llvm.inline_asm(
            _wgmma_53_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_15 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_53, position=[31]))
        _wgmma_54_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_54 = cutlass_llvm.inline_asm(
            _wgmma_54_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_15 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_54, position=[31]))
        _wgmma_55_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_55 = cutlass_llvm.inline_asm(
            _wgmma_55_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_15 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_55, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0_4[0] = cutlass.Float32((0 - float("inf")))
        new_max1_4[0] = cutlass.Float32((0 - float("inf")))
        _max_152 = cute.arch.fmax(new_max0_4[0], d_qk[0], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_152)
        _max_153 = cute.arch.fmax(new_max0_4[0], d_qk[1], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_153)
        _max_154 = cute.arch.fmax(new_max0_4[0], d_qk[4], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_154)
        _max_155 = cute.arch.fmax(new_max0_4[0], d_qk[5], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_155)
        _max_156 = cute.arch.fmax(new_max0_4[0], d_qk[8], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_156)
        _max_157 = cute.arch.fmax(new_max0_4[0], d_qk[9], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_157)
        _max_158 = cute.arch.fmax(new_max0_4[0], d_qk[12], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_158)
        _max_159 = cute.arch.fmax(new_max0_4[0], d_qk[13], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_159)
        _max_160 = cute.arch.fmax(new_max0_4[0], d_qk[16], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_160)
        _max_161 = cute.arch.fmax(new_max0_4[0], d_qk[17], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_161)
        _max_162 = cute.arch.fmax(new_max0_4[0], d_qk[20], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_162)
        _max_163 = cute.arch.fmax(new_max0_4[0], d_qk[21], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_163)
        _max_164 = cute.arch.fmax(new_max0_4[0], d_qk[24], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_164)
        _max_165 = cute.arch.fmax(new_max0_4[0], d_qk[25], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_165)
        _max_166 = cute.arch.fmax(new_max0_4[0], d_qk[28], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_166)
        _max_167 = cute.arch.fmax(new_max0_4[0], d_qk[29], ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_167)
        _max_168 = cute.arch.fmax(new_max1_4[0], d_qk[2], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_168)
        _max_169 = cute.arch.fmax(new_max1_4[0], d_qk[3], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_169)
        _max_170 = cute.arch.fmax(new_max1_4[0], d_qk[6], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_170)
        _max_171 = cute.arch.fmax(new_max1_4[0], d_qk[7], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_171)
        _max_172 = cute.arch.fmax(new_max1_4[0], d_qk[10], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_172)
        _max_173 = cute.arch.fmax(new_max1_4[0], d_qk[11], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_173)
        _max_174 = cute.arch.fmax(new_max1_4[0], d_qk[14], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_174)
        _max_175 = cute.arch.fmax(new_max1_4[0], d_qk[15], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_175)
        _max_176 = cute.arch.fmax(new_max1_4[0], d_qk[18], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_176)
        _max_177 = cute.arch.fmax(new_max1_4[0], d_qk[19], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_177)
        _max_178 = cute.arch.fmax(new_max1_4[0], d_qk[22], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_178)
        _max_179 = cute.arch.fmax(new_max1_4[0], d_qk[23], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_179)
        _max_180 = cute.arch.fmax(new_max1_4[0], d_qk[26], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_180)
        _max_181 = cute.arch.fmax(new_max1_4[0], d_qk[27], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_181)
        _max_182 = cute.arch.fmax(new_max1_4[0], d_qk[30], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_182)
        _max_183 = cute.arch.fmax(new_max1_4[0], d_qk[31], ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_183)
        _shfl_xor_16 = cute.arch.shuffle_sync_bfly(new_max0_4[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_184 = cute.arch.fmax(new_max0_4[0], _shfl_xor_16, ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_184)
        _shfl_xor_17 = cute.arch.shuffle_sync_bfly(new_max0_4[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_185 = cute.arch.fmax(new_max0_4[0], _shfl_xor_17, ftz=False)
        new_max0_4[0] = cutlass.Float32(_max_185)
        _shfl_xor_18 = cute.arch.shuffle_sync_bfly(new_max1_4[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_186 = cute.arch.fmax(new_max1_4[0], _shfl_xor_18, ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_186)
        _shfl_xor_19 = cute.arch.shuffle_sync_bfly(new_max1_4[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_187 = cute.arch.fmax(new_max1_4[0], _shfl_xor_19, ftz=False)
        new_max1_4[0] = cutlass.Float32(_max_187)
        _max_188 = cute.arch.fmax(row_max0[0], new_max0_4[0], ftz=False)
        merged_max0_5 = cutlass.Float32(_max_188)
        _max_189 = cute.arch.fmax(row_max1[0], new_max1_4[0], ftz=False)
        merged_max1_5 = cutlass.Float32(_max_189)
        _exp2_136 = cute.math.exp2((row_max0[0] - merged_max0_5), approx=True, ftz=True)
        _exp2_137 = cute.math.exp2((row_max1[0] - merged_max1_5), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_136))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_136))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_136))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_136))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_136))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_136))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_136))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_136))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_136))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_136))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_136))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_136))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_136))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_136))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_136))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_136))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_136))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_136))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_136))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_136))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_136))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_136))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_136))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_136))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_136))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_136))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_136))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_136))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_136))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_136))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_136))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_136))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_137))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_137))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_137))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_137))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_137))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_137))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_137))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_137))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_137))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_137))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_137))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_137))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_137))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_137))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_137))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_137))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_137))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_137))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_137))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_137))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_137))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_137))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_137))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_137))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_137))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_137))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_137))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_137))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_137))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_137))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_137))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_137))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_136))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_137))
        row_max0[0] = cutlass.Float32(merged_max0_5)
        row_max1[0] = cutlass.Float32(merged_max1_5)
        _exp2_138 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_139 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_138)
        d_qk[1] = cutlass.Float32(_exp2_139)
        row_sum0[0] += cutlass.Float32((_exp2_138 + _exp2_139))
        _exp2_140 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_141 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_140)
        d_qk[3] = cutlass.Float32(_exp2_141)
        row_sum1[0] += cutlass.Float32((_exp2_140 + _exp2_141))
        _exp2_142 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_143 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_142)
        d_qk[5] = cutlass.Float32(_exp2_143)
        row_sum0[0] += cutlass.Float32((_exp2_142 + _exp2_143))
        _exp2_144 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_145 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_144)
        d_qk[7] = cutlass.Float32(_exp2_145)
        row_sum1[0] += cutlass.Float32((_exp2_144 + _exp2_145))
        _exp2_146 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_147 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_146)
        d_qk[9] = cutlass.Float32(_exp2_147)
        row_sum0[0] += cutlass.Float32((_exp2_146 + _exp2_147))
        _exp2_148 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_149 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_148)
        d_qk[11] = cutlass.Float32(_exp2_149)
        row_sum1[0] += cutlass.Float32((_exp2_148 + _exp2_149))
        _exp2_150 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_151 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_150)
        d_qk[13] = cutlass.Float32(_exp2_151)
        row_sum0[0] += cutlass.Float32((_exp2_150 + _exp2_151))
        _exp2_152 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_153 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_152)
        d_qk[15] = cutlass.Float32(_exp2_153)
        row_sum1[0] += cutlass.Float32((_exp2_152 + _exp2_153))
        _exp2_154 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_155 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_154)
        d_qk[17] = cutlass.Float32(_exp2_155)
        row_sum0[0] += cutlass.Float32((_exp2_154 + _exp2_155))
        _exp2_156 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_157 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_156)
        d_qk[19] = cutlass.Float32(_exp2_157)
        row_sum1[0] += cutlass.Float32((_exp2_156 + _exp2_157))
        _exp2_158 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_159 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_158)
        d_qk[21] = cutlass.Float32(_exp2_159)
        row_sum0[0] += cutlass.Float32((_exp2_158 + _exp2_159))
        _exp2_160 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_161 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_160)
        d_qk[23] = cutlass.Float32(_exp2_161)
        row_sum1[0] += cutlass.Float32((_exp2_160 + _exp2_161))
        _exp2_162 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_163 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_162)
        d_qk[25] = cutlass.Float32(_exp2_163)
        row_sum0[0] += cutlass.Float32((_exp2_162 + _exp2_163))
        _exp2_164 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_165 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_164)
        d_qk[27] = cutlass.Float32(_exp2_165)
        row_sum1[0] += cutlass.Float32((_exp2_164 + _exp2_165))
        _exp2_166 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_167 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_166)
        d_qk[29] = cutlass.Float32(_exp2_167)
        row_sum0[0] += cutlass.Float32((_exp2_166 + _exp2_167))
        _exp2_168 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_169 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_168)
        d_qk[31] = cutlass.Float32(_exp2_169)
        row_sum1[0] += cutlass.Float32((_exp2_168 + _exp2_169))
        _bf16x2_64 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_64)
        _bf16x2_65 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_65)
        _bf16x2_66 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_66)
        _bf16x2_67 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_67)
        _bf16x2_68 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_68)
        _bf16x2_69 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_69)
        _bf16x2_70 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_70)
        _bf16x2_71 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_71)
        _bf16x2_72 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_72)
        _bf16x2_73 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_73)
        _bf16x2_74 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_74)
        _bf16x2_75 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_75)
        _bf16x2_76 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_76)
        _bf16x2_77 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_77)
        _bf16x2_78 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_78)
        _bf16x2_79 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_79)
        while not prims.mbarrier_wait_parity(v_full4_addr, _phase_v_full4_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full4_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_56_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_56 = cutlass_llvm.inline_asm(
            _wgmma_56_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_16).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_56, position=[63]))
        _wgmma_57_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_57 = cutlass_llvm.inline_asm(
            _wgmma_57_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_16 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_57, position=[63]))
        _wgmma_58_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_58 = cutlass_llvm.inline_asm(
            _wgmma_58_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_16 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_58, position=[63]))
        _wgmma_59_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_59 = cutlass_llvm.inline_asm(
            _wgmma_59_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_16 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_59, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _phase_k_full5_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_17_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 81920)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_17 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_17_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_17_raw)))
    _wgmma_b_0_18_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 81920) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_18 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_18_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_18_raw)))
    _phase_v_full5_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_19_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 81920)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_19 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_19_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_19_raw)))
    if (lim_odd[0] > 5):
        while not prims.mbarrier_wait_parity(k_full5_addr, _phase_k_full5_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full5_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_60_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_60 = cutlass_llvm.inline_asm(
            _wgmma_60_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_17).ir_value(),
                cutlass.Uint64(_wgmma_a_0_0).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 0, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_60, position=[31]))
        _wgmma_61_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_61 = cutlass_llvm.inline_asm(
            _wgmma_61_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_17 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_61, position=[31]))
        _wgmma_62_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_62 = cutlass_llvm.inline_asm(
            _wgmma_62_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_17 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_62, position=[31]))
        _wgmma_63_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_63 = cutlass_llvm.inline_asm(
            _wgmma_63_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_17 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_0 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_63, position=[31]))
        _wgmma_64_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_64 = cutlass_llvm.inline_asm(
            _wgmma_64_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_18).ir_value(),
                cutlass.Uint64(_wgmma_a_0_2).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_64, position=[31]))
        _wgmma_65_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_65 = cutlass_llvm.inline_asm(
            _wgmma_65_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_18 + 2)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 2)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_65, position=[31]))
        _wgmma_66_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_66 = cutlass_llvm.inline_asm(
            _wgmma_66_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_18 + 4)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 4)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_66, position=[31]))
        _wgmma_67_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 32)
        _wgmma_67 = cutlass_llvm.inline_asm(
            _wgmma_67_ty,
            [
                cutlass.Float32(d_qk[(0) + 0]).ir_value(),
                cutlass.Float32(d_qk[(0) + 1]).ir_value(),
                cutlass.Float32(d_qk[(0) + 2]).ir_value(),
                cutlass.Float32(d_qk[(0) + 3]).ir_value(),
                cutlass.Float32(d_qk[(0) + 4]).ir_value(),
                cutlass.Float32(d_qk[(0) + 5]).ir_value(),
                cutlass.Float32(d_qk[(0) + 6]).ir_value(),
                cutlass.Float32(d_qk[(0) + 7]).ir_value(),
                cutlass.Float32(d_qk[(0) + 8]).ir_value(),
                cutlass.Float32(d_qk[(0) + 9]).ir_value(),
                cutlass.Float32(d_qk[(0) + 10]).ir_value(),
                cutlass.Float32(d_qk[(0) + 11]).ir_value(),
                cutlass.Float32(d_qk[(0) + 12]).ir_value(),
                cutlass.Float32(d_qk[(0) + 13]).ir_value(),
                cutlass.Float32(d_qk[(0) + 14]).ir_value(),
                cutlass.Float32(d_qk[(0) + 15]).ir_value(),
                cutlass.Float32(d_qk[(0) + 16]).ir_value(),
                cutlass.Float32(d_qk[(0) + 17]).ir_value(),
                cutlass.Float32(d_qk[(0) + 18]).ir_value(),
                cutlass.Float32(d_qk[(0) + 19]).ir_value(),
                cutlass.Float32(d_qk[(0) + 20]).ir_value(),
                cutlass.Float32(d_qk[(0) + 21]).ir_value(),
                cutlass.Float32(d_qk[(0) + 22]).ir_value(),
                cutlass.Float32(d_qk[(0) + 23]).ir_value(),
                cutlass.Float32(d_qk[(0) + 24]).ir_value(),
                cutlass.Float32(d_qk[(0) + 25]).ir_value(),
                cutlass.Float32(d_qk[(0) + 26]).ir_value(),
                cutlass.Float32(d_qk[(0) + 27]).ir_value(),
                cutlass.Float32(d_qk[(0) + 28]).ir_value(),
                cutlass.Float32(d_qk[(0) + 29]).ir_value(),
                cutlass.Float32(d_qk[(0) + 30]).ir_value(),
                cutlass.Float32(d_qk[(0) + 31]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_18 + 6)).ir_value(),
                cutlass.Uint64((_wgmma_a_0_2 + 6)).ir_value(),
            ],
            asm_string='{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $65, $64, 1, 1, 1, 0, 0;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[0]))
        d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[1]))
        d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[2]))
        d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[3]))
        d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[4]))
        d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[5]))
        d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[6]))
        d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[7]))
        d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[8]))
        d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[9]))
        d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[10]))
        d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[11]))
        d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[12]))
        d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[13]))
        d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[14]))
        d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[15]))
        d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[16]))
        d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[17]))
        d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[18]))
        d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[19]))
        d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[20]))
        d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[21]))
        d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[22]))
        d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[23]))
        d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[24]))
        d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[25]))
        d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[26]))
        d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[27]))
        d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[28]))
        d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[29]))
        d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[30]))
        d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_67, position=[31]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0_5[0] = cutlass.Float32((0 - float("inf")))
        new_max1_5[0] = cutlass.Float32((0 - float("inf")))
        _max_190 = cute.arch.fmax(new_max0_5[0], d_qk[0], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_190)
        _max_191 = cute.arch.fmax(new_max0_5[0], d_qk[1], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_191)
        _max_192 = cute.arch.fmax(new_max0_5[0], d_qk[4], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_192)
        _max_193 = cute.arch.fmax(new_max0_5[0], d_qk[5], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_193)
        _max_194 = cute.arch.fmax(new_max0_5[0], d_qk[8], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_194)
        _max_195 = cute.arch.fmax(new_max0_5[0], d_qk[9], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_195)
        _max_196 = cute.arch.fmax(new_max0_5[0], d_qk[12], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_196)
        _max_197 = cute.arch.fmax(new_max0_5[0], d_qk[13], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_197)
        _max_198 = cute.arch.fmax(new_max0_5[0], d_qk[16], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_198)
        _max_199 = cute.arch.fmax(new_max0_5[0], d_qk[17], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_199)
        _max_200 = cute.arch.fmax(new_max0_5[0], d_qk[20], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_200)
        _max_201 = cute.arch.fmax(new_max0_5[0], d_qk[21], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_201)
        _max_202 = cute.arch.fmax(new_max0_5[0], d_qk[24], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_202)
        _max_203 = cute.arch.fmax(new_max0_5[0], d_qk[25], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_203)
        _max_204 = cute.arch.fmax(new_max0_5[0], d_qk[28], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_204)
        _max_205 = cute.arch.fmax(new_max0_5[0], d_qk[29], ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_205)
        _max_206 = cute.arch.fmax(new_max1_5[0], d_qk[2], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_206)
        _max_207 = cute.arch.fmax(new_max1_5[0], d_qk[3], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_207)
        _max_208 = cute.arch.fmax(new_max1_5[0], d_qk[6], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_208)
        _max_209 = cute.arch.fmax(new_max1_5[0], d_qk[7], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_209)
        _max_210 = cute.arch.fmax(new_max1_5[0], d_qk[10], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_210)
        _max_211 = cute.arch.fmax(new_max1_5[0], d_qk[11], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_211)
        _max_212 = cute.arch.fmax(new_max1_5[0], d_qk[14], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_212)
        _max_213 = cute.arch.fmax(new_max1_5[0], d_qk[15], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_213)
        _max_214 = cute.arch.fmax(new_max1_5[0], d_qk[18], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_214)
        _max_215 = cute.arch.fmax(new_max1_5[0], d_qk[19], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_215)
        _max_216 = cute.arch.fmax(new_max1_5[0], d_qk[22], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_216)
        _max_217 = cute.arch.fmax(new_max1_5[0], d_qk[23], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_217)
        _max_218 = cute.arch.fmax(new_max1_5[0], d_qk[26], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_218)
        _max_219 = cute.arch.fmax(new_max1_5[0], d_qk[27], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_219)
        _max_220 = cute.arch.fmax(new_max1_5[0], d_qk[30], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_220)
        _max_221 = cute.arch.fmax(new_max1_5[0], d_qk[31], ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_221)
        _shfl_xor_20 = cute.arch.shuffle_sync_bfly(new_max0_5[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_222 = cute.arch.fmax(new_max0_5[0], _shfl_xor_20, ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_222)
        _shfl_xor_21 = cute.arch.shuffle_sync_bfly(new_max0_5[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_223 = cute.arch.fmax(new_max0_5[0], _shfl_xor_21, ftz=False)
        new_max0_5[0] = cutlass.Float32(_max_223)
        _shfl_xor_22 = cute.arch.shuffle_sync_bfly(new_max1_5[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_224 = cute.arch.fmax(new_max1_5[0], _shfl_xor_22, ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_224)
        _shfl_xor_23 = cute.arch.shuffle_sync_bfly(new_max1_5[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_225 = cute.arch.fmax(new_max1_5[0], _shfl_xor_23, ftz=False)
        new_max1_5[0] = cutlass.Float32(_max_225)
        _max_226 = cute.arch.fmax(row_max0[0], new_max0_5[0], ftz=False)
        merged_max0_6 = cutlass.Float32(_max_226)
        _max_227 = cute.arch.fmax(row_max1[0], new_max1_5[0], ftz=False)
        merged_max1_6 = cutlass.Float32(_max_227)
        _exp2_170 = cute.math.exp2((row_max0[0] - merged_max0_6), approx=True, ftz=True)
        _exp2_171 = cute.math.exp2((row_max1[0] - merged_max1_6), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_170))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_170))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_170))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_170))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_170))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_170))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_170))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_170))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_170))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_170))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_170))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_170))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_170))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_170))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_170))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_170))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_170))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_170))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_170))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_170))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_170))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_170))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_170))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_170))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_170))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_170))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_170))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_170))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_170))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_170))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_170))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_170))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_171))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_171))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_171))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_171))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_171))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_171))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_171))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_171))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_171))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_171))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_171))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_171))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_171))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_171))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_171))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_171))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_171))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_171))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_171))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_171))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_171))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_171))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_171))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_171))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_171))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_171))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_171))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_171))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_171))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_171))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_171))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_171))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_170))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_171))
        row_max0[0] = cutlass.Float32(merged_max0_6)
        row_max1[0] = cutlass.Float32(merged_max1_6)
        _exp2_172 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_173 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_172)
        d_qk[1] = cutlass.Float32(_exp2_173)
        row_sum0[0] += cutlass.Float32((_exp2_172 + _exp2_173))
        _exp2_174 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_175 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_174)
        d_qk[3] = cutlass.Float32(_exp2_175)
        row_sum1[0] += cutlass.Float32((_exp2_174 + _exp2_175))
        _exp2_176 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_177 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_176)
        d_qk[5] = cutlass.Float32(_exp2_177)
        row_sum0[0] += cutlass.Float32((_exp2_176 + _exp2_177))
        _exp2_178 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_179 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_178)
        d_qk[7] = cutlass.Float32(_exp2_179)
        row_sum1[0] += cutlass.Float32((_exp2_178 + _exp2_179))
        _exp2_180 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_181 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_180)
        d_qk[9] = cutlass.Float32(_exp2_181)
        row_sum0[0] += cutlass.Float32((_exp2_180 + _exp2_181))
        _exp2_182 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_183 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_182)
        d_qk[11] = cutlass.Float32(_exp2_183)
        row_sum1[0] += cutlass.Float32((_exp2_182 + _exp2_183))
        _exp2_184 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_185 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_184)
        d_qk[13] = cutlass.Float32(_exp2_185)
        row_sum0[0] += cutlass.Float32((_exp2_184 + _exp2_185))
        _exp2_186 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_187 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_186)
        d_qk[15] = cutlass.Float32(_exp2_187)
        row_sum1[0] += cutlass.Float32((_exp2_186 + _exp2_187))
        _exp2_188 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_189 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_188)
        d_qk[17] = cutlass.Float32(_exp2_189)
        row_sum0[0] += cutlass.Float32((_exp2_188 + _exp2_189))
        _exp2_190 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_191 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_190)
        d_qk[19] = cutlass.Float32(_exp2_191)
        row_sum1[0] += cutlass.Float32((_exp2_190 + _exp2_191))
        _exp2_192 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_193 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_192)
        d_qk[21] = cutlass.Float32(_exp2_193)
        row_sum0[0] += cutlass.Float32((_exp2_192 + _exp2_193))
        _exp2_194 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_195 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_194)
        d_qk[23] = cutlass.Float32(_exp2_195)
        row_sum1[0] += cutlass.Float32((_exp2_194 + _exp2_195))
        _exp2_196 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_197 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_196)
        d_qk[25] = cutlass.Float32(_exp2_197)
        row_sum0[0] += cutlass.Float32((_exp2_196 + _exp2_197))
        _exp2_198 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_199 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_198)
        d_qk[27] = cutlass.Float32(_exp2_199)
        row_sum1[0] += cutlass.Float32((_exp2_198 + _exp2_199))
        _exp2_200 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_201 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_200)
        d_qk[29] = cutlass.Float32(_exp2_201)
        row_sum0[0] += cutlass.Float32((_exp2_200 + _exp2_201))
        _exp2_202 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_203 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_202)
        d_qk[31] = cutlass.Float32(_exp2_203)
        row_sum1[0] += cutlass.Float32((_exp2_202 + _exp2_203))
        _bf16x2_80 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_80)
        _bf16x2_81 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_81)
        _bf16x2_82 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_82)
        _bf16x2_83 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_83)
        _bf16x2_84 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_84)
        _bf16x2_85 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_85)
        _bf16x2_86 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_86)
        _bf16x2_87 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_87)
        _bf16x2_88 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_88)
        _bf16x2_89 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_89)
        _bf16x2_90 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_90)
        _bf16x2_91 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_91)
        _bf16x2_92 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_92)
        _bf16x2_93 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_93)
        _bf16x2_94 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_94)
        _bf16x2_95 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_95)
        while not prims.mbarrier_wait_parity(v_full5_addr, _phase_v_full5_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full5_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_68_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_68 = cutlass_llvm.inline_asm(
            _wgmma_68_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64(_wgmma_b_0_19).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_68, position=[63]))
        _wgmma_69_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_69 = cutlass_llvm.inline_asm(
            _wgmma_69_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_19 + 128)).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_69, position=[63]))
        _wgmma_70_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_70 = cutlass_llvm.inline_asm(
            _wgmma_70_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_19 + 256)).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_70, position=[63]))
        _wgmma_71_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
        _wgmma_71 = cutlass_llvm.inline_asm(
            _wgmma_71_ty,
            [
                cutlass.Float32(d_o[(0) + 0]).ir_value(),
                cutlass.Float32(d_o[(0) + 1]).ir_value(),
                cutlass.Float32(d_o[(0) + 2]).ir_value(),
                cutlass.Float32(d_o[(0) + 3]).ir_value(),
                cutlass.Float32(d_o[(0) + 4]).ir_value(),
                cutlass.Float32(d_o[(0) + 5]).ir_value(),
                cutlass.Float32(d_o[(0) + 6]).ir_value(),
                cutlass.Float32(d_o[(0) + 7]).ir_value(),
                cutlass.Float32(d_o[(0) + 8]).ir_value(),
                cutlass.Float32(d_o[(0) + 9]).ir_value(),
                cutlass.Float32(d_o[(0) + 10]).ir_value(),
                cutlass.Float32(d_o[(0) + 11]).ir_value(),
                cutlass.Float32(d_o[(0) + 12]).ir_value(),
                cutlass.Float32(d_o[(0) + 13]).ir_value(),
                cutlass.Float32(d_o[(0) + 14]).ir_value(),
                cutlass.Float32(d_o[(0) + 15]).ir_value(),
                cutlass.Float32(d_o[(0) + 16]).ir_value(),
                cutlass.Float32(d_o[(0) + 17]).ir_value(),
                cutlass.Float32(d_o[(0) + 18]).ir_value(),
                cutlass.Float32(d_o[(0) + 19]).ir_value(),
                cutlass.Float32(d_o[(0) + 20]).ir_value(),
                cutlass.Float32(d_o[(0) + 21]).ir_value(),
                cutlass.Float32(d_o[(0) + 22]).ir_value(),
                cutlass.Float32(d_o[(0) + 23]).ir_value(),
                cutlass.Float32(d_o[(0) + 24]).ir_value(),
                cutlass.Float32(d_o[(0) + 25]).ir_value(),
                cutlass.Float32(d_o[(0) + 26]).ir_value(),
                cutlass.Float32(d_o[(0) + 27]).ir_value(),
                cutlass.Float32(d_o[(0) + 28]).ir_value(),
                cutlass.Float32(d_o[(0) + 29]).ir_value(),
                cutlass.Float32(d_o[(0) + 30]).ir_value(),
                cutlass.Float32(d_o[(0) + 31]).ir_value(),
                cutlass.Float32(d_o[(0) + 32]).ir_value(),
                cutlass.Float32(d_o[(0) + 33]).ir_value(),
                cutlass.Float32(d_o[(0) + 34]).ir_value(),
                cutlass.Float32(d_o[(0) + 35]).ir_value(),
                cutlass.Float32(d_o[(0) + 36]).ir_value(),
                cutlass.Float32(d_o[(0) + 37]).ir_value(),
                cutlass.Float32(d_o[(0) + 38]).ir_value(),
                cutlass.Float32(d_o[(0) + 39]).ir_value(),
                cutlass.Float32(d_o[(0) + 40]).ir_value(),
                cutlass.Float32(d_o[(0) + 41]).ir_value(),
                cutlass.Float32(d_o[(0) + 42]).ir_value(),
                cutlass.Float32(d_o[(0) + 43]).ir_value(),
                cutlass.Float32(d_o[(0) + 44]).ir_value(),
                cutlass.Float32(d_o[(0) + 45]).ir_value(),
                cutlass.Float32(d_o[(0) + 46]).ir_value(),
                cutlass.Float32(d_o[(0) + 47]).ir_value(),
                cutlass.Float32(d_o[(0) + 48]).ir_value(),
                cutlass.Float32(d_o[(0) + 49]).ir_value(),
                cutlass.Float32(d_o[(0) + 50]).ir_value(),
                cutlass.Float32(d_o[(0) + 51]).ir_value(),
                cutlass.Float32(d_o[(0) + 52]).ir_value(),
                cutlass.Float32(d_o[(0) + 53]).ir_value(),
                cutlass.Float32(d_o[(0) + 54]).ir_value(),
                cutlass.Float32(d_o[(0) + 55]).ir_value(),
                cutlass.Float32(d_o[(0) + 56]).ir_value(),
                cutlass.Float32(d_o[(0) + 57]).ir_value(),
                cutlass.Float32(d_o[(0) + 58]).ir_value(),
                cutlass.Float32(d_o[(0) + 59]).ir_value(),
                cutlass.Float32(d_o[(0) + 60]).ir_value(),
                cutlass.Float32(d_o[(0) + 61]).ir_value(),
                cutlass.Float32(d_o[(0) + 62]).ir_value(),
                cutlass.Float32(d_o[(0) + 63]).ir_value(),
                cutlass.Uint64((_wgmma_b_0_19 + 384)).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                cutlass_arith.extui(cutlass.Int32.mlir_type, cutlass.Boolean((True) != 0).ir_value()),
            ],
            asm_string='{\n.reg .pred p;\nsetp.ne.b32 p, $133, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, p, 1, 1, 1;\n}\n',
            constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,r,~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
        d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[0]))
        d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[1]))
        d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[2]))
        d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[3]))
        d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[4]))
        d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[5]))
        d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[6]))
        d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[7]))
        d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[8]))
        d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[9]))
        d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[10]))
        d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[11]))
        d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[12]))
        d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[13]))
        d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[14]))
        d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[15]))
        d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[16]))
        d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[17]))
        d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[18]))
        d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[19]))
        d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[20]))
        d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[21]))
        d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[22]))
        d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[23]))
        d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[24]))
        d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[25]))
        d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[26]))
        d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[27]))
        d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[28]))
        d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[29]))
        d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[30]))
        d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[31]))
        d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[32]))
        d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[33]))
        d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[34]))
        d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[35]))
        d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[36]))
        d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[37]))
        d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[38]))
        d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[39]))
        d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[40]))
        d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[41]))
        d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[42]))
        d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[43]))
        d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[44]))
        d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[45]))
        d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[46]))
        d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[47]))
        d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[48]))
        d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[49]))
        d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[50]))
        d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[51]))
        d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[52]))
        d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[53]))
        d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[54]))
        d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[55]))
        d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[56]))
        d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[57]))
        d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[58]))
        d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[59]))
        d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[60]))
        d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[61]))
        d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[62]))
        d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_71, position=[63]))
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _shfl_xor_24 = cute.arch.shuffle_sync_bfly(row_sum0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_24)
    _shfl_xor_25 = cute.arch.shuffle_sync_bfly(row_sum0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_25)
    _shfl_xor_26 = cute.arch.shuffle_sync_bfly(row_sum1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_26)
    _shfl_xor_27 = cute.arch.shuffle_sync_bfly(row_sum1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_27)
    qj = cutlass.Int32((lane & 3))
    if (wg == 1):
        prims.store_ext(cutlass.Float32(d_o[0]).ir_value(), (part_lo + (tid_wg)))
        prims.store_ext(cutlass.Float32(d_o[1]).ir_value(), (part_lo + ((128 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[2]).ir_value(), (part_lo + ((256 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[3]).ir_value(), (part_lo + ((384 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[4]).ir_value(), (part_lo + ((512 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[5]).ir_value(), (part_lo + ((640 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[6]).ir_value(), (part_lo + ((768 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[7]).ir_value(), (part_lo + ((896 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[8]).ir_value(), (part_lo + ((1024 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[9]).ir_value(), (part_lo + ((1152 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[10]).ir_value(), (part_lo + ((1280 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[11]).ir_value(), (part_lo + ((1408 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[12]).ir_value(), (part_lo + ((1536 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[13]).ir_value(), (part_lo + ((1664 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[14]).ir_value(), (part_lo + ((1792 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[15]).ir_value(), (part_lo + ((1920 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[16]).ir_value(), (part_lo + ((2048 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[17]).ir_value(), (part_lo + ((2176 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[18]).ir_value(), (part_lo + ((2304 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[19]).ir_value(), (part_lo + ((2432 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[20]).ir_value(), (part_lo + ((2560 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[21]).ir_value(), (part_lo + ((2688 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[22]).ir_value(), (part_lo + ((2816 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[23]).ir_value(), (part_lo + ((2944 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[24]).ir_value(), (part_lo + ((3072 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[25]).ir_value(), (part_lo + ((3200 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[26]).ir_value(), (part_lo + ((3328 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[27]).ir_value(), (part_lo + ((3456 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[28]).ir_value(), (part_lo + ((3584 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[29]).ir_value(), (part_lo + ((3712 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[30]).ir_value(), (part_lo + ((3840 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[31]).ir_value(), (part_lo + ((3968 + tid_wg))))
        if (qj == 0):
            prims.store_ext(cutlass.Float32(row_max0[0]).ir_value(), (mstats + ((quad * 4))))
            prims.store_ext(cutlass.Float32(row_max1[0]).ir_value(), (mstats + (((quad * 4) + 1))))
            prims.store_ext(cutlass.Float32(row_sum0[0]).ir_value(), (mstats + (((quad * 4) + 2))))
            prims.store_ext(cutlass.Float32(row_sum1[0]).ir_value(), (mstats + (((quad * 4) + 3))))
    if (wg == 0):
        prims.store_ext(cutlass.Float32(d_o[32]).ir_value(), (part_hi0 + (tid_wg)))
        prims.store_ext(cutlass.Float32(d_o[33]).ir_value(), (part_hi0 + ((128 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[34]).ir_value(), (part_hi0 + ((256 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[35]).ir_value(), (part_hi0 + ((384 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[36]).ir_value(), (part_hi0 + ((512 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[37]).ir_value(), (part_hi0 + ((640 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[38]).ir_value(), (part_hi0 + ((768 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[39]).ir_value(), (part_hi0 + ((896 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[40]).ir_value(), (part_hi0 + ((1024 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[41]).ir_value(), (part_hi0 + ((1152 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[42]).ir_value(), (part_hi0 + ((1280 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[43]).ir_value(), (part_hi0 + ((1408 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[44]).ir_value(), (part_hi0 + ((1536 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[45]).ir_value(), (part_hi0 + ((1664 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[46]).ir_value(), (part_hi0 + ((1792 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[47]).ir_value(), (part_hi0 + ((1920 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[48]).ir_value(), (part_hi0 + ((2048 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[49]).ir_value(), (part_hi0 + ((2176 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[50]).ir_value(), (part_hi0 + ((2304 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[51]).ir_value(), (part_hi0 + ((2432 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[52]).ir_value(), (part_hi0 + ((2560 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[53]).ir_value(), (part_hi0 + ((2688 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[54]).ir_value(), (part_hi0 + ((2816 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[55]).ir_value(), (part_hi0 + ((2944 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[56]).ir_value(), (part_hi0 + ((3072 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[57]).ir_value(), (part_hi0 + ((3200 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[58]).ir_value(), (part_hi0 + ((3328 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[59]).ir_value(), (part_hi0 + ((3456 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[60]).ir_value(), (part_hi0 + ((3584 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[61]).ir_value(), (part_hi0 + ((3712 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[62]).ir_value(), (part_hi0 + ((3840 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[63]).ir_value(), (part_hi0 + ((3968 + tid_wg))))
        if (qj == 0):
            prims.store_ext(cutlass.Float32(row_max0[0]).ir_value(), (mstats + ((128 + (quad * 4)))))
            prims.store_ext(cutlass.Float32(row_max1[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 1))))
            prims.store_ext(cutlass.Float32(row_sum0[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 2))))
            prims.store_ext(cutlass.Float32(row_sum1[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 3))))
    prims.barrier_cta_sync(8, thread_count=256)
    pbase = cutlass.Int32((0 if (wg == 0) else 128))
    pm0 = cutlass.Float32(_mstats[(pbase + (quad * 4))])
    pm1 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 1)])
    pl0 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 2)])
    pl1 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 3)])
    x0m0 = cutlass.Float32((row_max0[0] if (wg == 0) else pm0))
    x0m1 = cutlass.Float32((row_max1[0] if (wg == 0) else pm1))
    x1m0 = cutlass.Float32((pm0 if (wg == 0) else row_max0[0]))
    x1m1 = cutlass.Float32((pm1 if (wg == 0) else row_max1[0]))
    x0l0 = cutlass.Float32((row_sum0[0] if (wg == 0) else pl0))
    x0l1 = cutlass.Float32((row_sum1[0] if (wg == 0) else pl1))
    x1l0 = cutlass.Float32((pl0 if (wg == 0) else row_sum0[0]))
    x1l1 = cutlass.Float32((pl1 if (wg == 0) else row_sum1[0]))
    _max_228 = cute.arch.fmax(x0m0, x1m0, ftz=False)
    mm0 = cutlass.Float32(_max_228)
    _max_229 = cute.arch.fmax(x0m1, x1m1, ftz=False)
    mm1 = cutlass.Float32(_max_229)
    _exp2_204 = cute.math.exp2((x0m0 - mm0), approx=True, ftz=True)
    f00 = cutlass.Float32((0.0 if (x0m0 == (0 - float("inf"))) else _exp2_204))
    _exp2_205 = cute.math.exp2((x0m1 - mm1), approx=True, ftz=True)
    f01 = cutlass.Float32((0.0 if (x0m1 == (0 - float("inf"))) else _exp2_205))
    _exp2_206 = cute.math.exp2((x1m0 - mm0), approx=True, ftz=True)
    f10 = cutlass.Float32((0.0 if (x1m0 == (0 - float("inf"))) else _exp2_206))
    _exp2_207 = cute.math.exp2((x1m1 - mm1), approx=True, ftz=True)
    f11 = cutlass.Float32((0.0 if (x1m1 == (0 - float("inf"))) else _exp2_207))
    a0 = cutlass.Float32((f00 if (wg == 0) else f10))
    a1 = cutlass.Float32((f01 if (wg == 0) else f11))
    b0 = cutlass.Float32((f10 if (wg == 0) else f00))
    b1 = cutlass.Float32((f11 if (wg == 0) else f01))
    if (wg == 0):
        _fma_0 = cute.math.fma(_part_lo[tid_wg], (b1 if False else b0), (d_o[0] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[0] = cutlass.Float32(_fma_0)
        _fma_1 = cute.math.fma(_part_lo[(128 + tid_wg)], (b1 if False else b0), (d_o[1] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[1] = cutlass.Float32(_fma_1)
        _fma_2 = cute.math.fma(_part_lo[(256 + tid_wg)], (b1 if True else b0), (d_o[2] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[2] = cutlass.Float32(_fma_2)
        _fma_3 = cute.math.fma(_part_lo[(384 + tid_wg)], (b1 if True else b0), (d_o[3] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[3] = cutlass.Float32(_fma_3)
        _fma_4 = cute.math.fma(_part_lo[(512 + tid_wg)], (b1 if False else b0), (d_o[4] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[4] = cutlass.Float32(_fma_4)
        _fma_5 = cute.math.fma(_part_lo[(640 + tid_wg)], (b1 if False else b0), (d_o[5] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[5] = cutlass.Float32(_fma_5)
        _fma_6 = cute.math.fma(_part_lo[(768 + tid_wg)], (b1 if True else b0), (d_o[6] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[6] = cutlass.Float32(_fma_6)
        _fma_7 = cute.math.fma(_part_lo[(896 + tid_wg)], (b1 if True else b0), (d_o[7] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[7] = cutlass.Float32(_fma_7)
        _fma_8 = cute.math.fma(_part_lo[(1024 + tid_wg)], (b1 if False else b0), (d_o[8] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[8] = cutlass.Float32(_fma_8)
        _fma_9 = cute.math.fma(_part_lo[(1152 + tid_wg)], (b1 if False else b0), (d_o[9] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[9] = cutlass.Float32(_fma_9)
        _fma_10 = cute.math.fma(_part_lo[(1280 + tid_wg)], (b1 if True else b0), (d_o[10] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[10] = cutlass.Float32(_fma_10)
        _fma_11 = cute.math.fma(_part_lo[(1408 + tid_wg)], (b1 if True else b0), (d_o[11] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[11] = cutlass.Float32(_fma_11)
        _fma_12 = cute.math.fma(_part_lo[(1536 + tid_wg)], (b1 if False else b0), (d_o[12] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[12] = cutlass.Float32(_fma_12)
        _fma_13 = cute.math.fma(_part_lo[(1664 + tid_wg)], (b1 if False else b0), (d_o[13] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[13] = cutlass.Float32(_fma_13)
        _fma_14 = cute.math.fma(_part_lo[(1792 + tid_wg)], (b1 if True else b0), (d_o[14] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[14] = cutlass.Float32(_fma_14)
        _fma_15 = cute.math.fma(_part_lo[(1920 + tid_wg)], (b1 if True else b0), (d_o[15] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[15] = cutlass.Float32(_fma_15)
        _fma_16 = cute.math.fma(_part_lo[(2048 + tid_wg)], (b1 if False else b0), (d_o[16] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[16] = cutlass.Float32(_fma_16)
        _fma_17 = cute.math.fma(_part_lo[(2176 + tid_wg)], (b1 if False else b0), (d_o[17] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[17] = cutlass.Float32(_fma_17)
        _fma_18 = cute.math.fma(_part_lo[(2304 + tid_wg)], (b1 if True else b0), (d_o[18] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[18] = cutlass.Float32(_fma_18)
        _fma_19 = cute.math.fma(_part_lo[(2432 + tid_wg)], (b1 if True else b0), (d_o[19] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[19] = cutlass.Float32(_fma_19)
        _fma_20 = cute.math.fma(_part_lo[(2560 + tid_wg)], (b1 if False else b0), (d_o[20] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[20] = cutlass.Float32(_fma_20)
        _fma_21 = cute.math.fma(_part_lo[(2688 + tid_wg)], (b1 if False else b0), (d_o[21] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[21] = cutlass.Float32(_fma_21)
        _fma_22 = cute.math.fma(_part_lo[(2816 + tid_wg)], (b1 if True else b0), (d_o[22] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[22] = cutlass.Float32(_fma_22)
        _fma_23 = cute.math.fma(_part_lo[(2944 + tid_wg)], (b1 if True else b0), (d_o[23] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[23] = cutlass.Float32(_fma_23)
        _fma_24 = cute.math.fma(_part_lo[(3072 + tid_wg)], (b1 if False else b0), (d_o[24] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[24] = cutlass.Float32(_fma_24)
        _fma_25 = cute.math.fma(_part_lo[(3200 + tid_wg)], (b1 if False else b0), (d_o[25] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[25] = cutlass.Float32(_fma_25)
        _fma_26 = cute.math.fma(_part_lo[(3328 + tid_wg)], (b1 if True else b0), (d_o[26] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[26] = cutlass.Float32(_fma_26)
        _fma_27 = cute.math.fma(_part_lo[(3456 + tid_wg)], (b1 if True else b0), (d_o[27] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[27] = cutlass.Float32(_fma_27)
        _fma_28 = cute.math.fma(_part_lo[(3584 + tid_wg)], (b1 if False else b0), (d_o[28] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[28] = cutlass.Float32(_fma_28)
        _fma_29 = cute.math.fma(_part_lo[(3712 + tid_wg)], (b1 if False else b0), (d_o[29] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[29] = cutlass.Float32(_fma_29)
        _fma_30 = cute.math.fma(_part_lo[(3840 + tid_wg)], (b1 if True else b0), (d_o[30] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[30] = cutlass.Float32(_fma_30)
        _fma_31 = cute.math.fma(_part_lo[(3968 + tid_wg)], (b1 if True else b0), (d_o[31] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[31] = cutlass.Float32(_fma_31)
    if (wg == 1):
        _fma_32 = cute.math.fma(_part_hi0[tid_wg], (b1 if False else b0), (d_o[32] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[32] = cutlass.Float32(_fma_32)
        _fma_33 = cute.math.fma(_part_hi0[(128 + tid_wg)], (b1 if False else b0), (d_o[33] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[33] = cutlass.Float32(_fma_33)
        _fma_34 = cute.math.fma(_part_hi0[(256 + tid_wg)], (b1 if True else b0), (d_o[34] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[34] = cutlass.Float32(_fma_34)
        _fma_35 = cute.math.fma(_part_hi0[(384 + tid_wg)], (b1 if True else b0), (d_o[35] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[35] = cutlass.Float32(_fma_35)
        _fma_36 = cute.math.fma(_part_hi0[(512 + tid_wg)], (b1 if False else b0), (d_o[36] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[36] = cutlass.Float32(_fma_36)
        _fma_37 = cute.math.fma(_part_hi0[(640 + tid_wg)], (b1 if False else b0), (d_o[37] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[37] = cutlass.Float32(_fma_37)
        _fma_38 = cute.math.fma(_part_hi0[(768 + tid_wg)], (b1 if True else b0), (d_o[38] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[38] = cutlass.Float32(_fma_38)
        _fma_39 = cute.math.fma(_part_hi0[(896 + tid_wg)], (b1 if True else b0), (d_o[39] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[39] = cutlass.Float32(_fma_39)
        _fma_40 = cute.math.fma(_part_hi0[(1024 + tid_wg)], (b1 if False else b0), (d_o[40] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[40] = cutlass.Float32(_fma_40)
        _fma_41 = cute.math.fma(_part_hi0[(1152 + tid_wg)], (b1 if False else b0), (d_o[41] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[41] = cutlass.Float32(_fma_41)
        _fma_42 = cute.math.fma(_part_hi0[(1280 + tid_wg)], (b1 if True else b0), (d_o[42] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[42] = cutlass.Float32(_fma_42)
        _fma_43 = cute.math.fma(_part_hi0[(1408 + tid_wg)], (b1 if True else b0), (d_o[43] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[43] = cutlass.Float32(_fma_43)
        _fma_44 = cute.math.fma(_part_hi0[(1536 + tid_wg)], (b1 if False else b0), (d_o[44] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[44] = cutlass.Float32(_fma_44)
        _fma_45 = cute.math.fma(_part_hi0[(1664 + tid_wg)], (b1 if False else b0), (d_o[45] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[45] = cutlass.Float32(_fma_45)
        _fma_46 = cute.math.fma(_part_hi0[(1792 + tid_wg)], (b1 if True else b0), (d_o[46] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[46] = cutlass.Float32(_fma_46)
        _fma_47 = cute.math.fma(_part_hi0[(1920 + tid_wg)], (b1 if True else b0), (d_o[47] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[47] = cutlass.Float32(_fma_47)
        _fma_48 = cute.math.fma(_part_hi0[(2048 + tid_wg)], (b1 if False else b0), (d_o[48] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[48] = cutlass.Float32(_fma_48)
        _fma_49 = cute.math.fma(_part_hi0[(2176 + tid_wg)], (b1 if False else b0), (d_o[49] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[49] = cutlass.Float32(_fma_49)
        _fma_50 = cute.math.fma(_part_hi0[(2304 + tid_wg)], (b1 if True else b0), (d_o[50] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[50] = cutlass.Float32(_fma_50)
        _fma_51 = cute.math.fma(_part_hi0[(2432 + tid_wg)], (b1 if True else b0), (d_o[51] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[51] = cutlass.Float32(_fma_51)
        _fma_52 = cute.math.fma(_part_hi0[(2560 + tid_wg)], (b1 if False else b0), (d_o[52] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[52] = cutlass.Float32(_fma_52)
        _fma_53 = cute.math.fma(_part_hi0[(2688 + tid_wg)], (b1 if False else b0), (d_o[53] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[53] = cutlass.Float32(_fma_53)
        _fma_54 = cute.math.fma(_part_hi0[(2816 + tid_wg)], (b1 if True else b0), (d_o[54] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[54] = cutlass.Float32(_fma_54)
        _fma_55 = cute.math.fma(_part_hi0[(2944 + tid_wg)], (b1 if True else b0), (d_o[55] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[55] = cutlass.Float32(_fma_55)
        _fma_56 = cute.math.fma(_part_hi0[(3072 + tid_wg)], (b1 if False else b0), (d_o[56] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[56] = cutlass.Float32(_fma_56)
        _fma_57 = cute.math.fma(_part_hi0[(3200 + tid_wg)], (b1 if False else b0), (d_o[57] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[57] = cutlass.Float32(_fma_57)
        _fma_58 = cute.math.fma(_part_hi0[(3328 + tid_wg)], (b1 if True else b0), (d_o[58] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[58] = cutlass.Float32(_fma_58)
        _fma_59 = cute.math.fma(_part_hi0[(3456 + tid_wg)], (b1 if True else b0), (d_o[59] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[59] = cutlass.Float32(_fma_59)
        _fma_60 = cute.math.fma(_part_hi0[(3584 + tid_wg)], (b1 if False else b0), (d_o[60] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[60] = cutlass.Float32(_fma_60)
        _fma_61 = cute.math.fma(_part_hi0[(3712 + tid_wg)], (b1 if False else b0), (d_o[61] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[61] = cutlass.Float32(_fma_61)
        _fma_62 = cute.math.fma(_part_hi0[(3840 + tid_wg)], (b1 if True else b0), (d_o[62] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[62] = cutlass.Float32(_fma_62)
        _fma_63 = cute.math.fma(_part_hi0[(3968 + tid_wg)], (b1 if True else b0), (d_o[63] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[63] = cutlass.Float32(_fma_63)
    _fma_64 = cute.math.fma(x1l0, f10, (x0l0 * f00), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
    row_sum0[0] = cutlass.Float32(_fma_64)
    _fma_65 = cute.math.fma(x1l1, f11, (x0l1 * f01), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
    row_sum1[0] = cutlass.Float32(_fma_65)
    row_max0[0] = cutlass.Float32(mm0)
    row_max1[0] = cutlass.Float32(mm1)
    cute.arch.cluster_wait()
    _phase_recv_full_0[0] = cutlass.Uint32(0)
    if (wg == 0):
        my_rank = cutlass.Int32(cta_rank)
        my_off = cutlass.Int32((tid_wg * 16))
        if (my_rank != 0):
            pslot[0] = cutlass.Int32(my_rank)
            _if_condition_72 = cutlass.Boolean((my_rank > 0))
            pslot[0] = cutlass.Int32(cutlass.select_(_if_condition_72, cutlass.Int32((my_rank - 1)), pslot[0]))
            _cluster_mapa_73 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(0))
            _mapa_0 = cutlass.Uint32(_cluster_mapa_73.toint())
            _cluster_mapa_74 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot[0] * 17408))) + cutlass.Uint32(my_off))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(0))
            _mapa_1 = cutlass.Uint32(_cluster_mapa_74.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_1)).ir_value(), (cutlass.Float32(d_o[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[1]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[2]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[3]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 2048))).ir_value(), (cutlass.Float32(d_o[4]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[5]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[6]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[7]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 4096))).ir_value(), (cutlass.Float32(d_o[8]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[9]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[10]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[11]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 6144))).ir_value(), (cutlass.Float32(d_o[12]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[13]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[14]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[15]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            if (qj == 0):
                _cluster_mapa_75 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot[0] * 17408))) + 16384) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(0))
                _mapa_2 = cutlass.Uint32(_cluster_mapa_75.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_2)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank != 1):
            pslot_1[0] = cutlass.Int32(my_rank)
            _if_condition_76 = cutlass.Boolean((my_rank > 1))
            pslot_1[0] = cutlass.Int32(cutlass.select_(_if_condition_76, cutlass.Int32((my_rank - 1)), pslot_1[0]))
            _cluster_mapa_77 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(1))
            _mapa_3 = cutlass.Uint32(_cluster_mapa_77.toint())
            _cluster_mapa_78 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot_1[0] * 17408))) + cutlass.Uint32(my_off))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(1))
            _mapa_4 = cutlass.Uint32(_cluster_mapa_78.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_4)).ir_value(), (cutlass.Float32(d_o[16]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[17]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[18]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[19]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 2048))).ir_value(), (cutlass.Float32(d_o[20]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[21]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[22]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[23]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 4096))).ir_value(), (cutlass.Float32(d_o[24]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[25]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[26]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[27]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 6144))).ir_value(), (cutlass.Float32(d_o[28]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[29]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[30]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[31]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            if (qj == 0):
                _cluster_mapa_79 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_1[0] * 17408))) + 16384) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(1))
                _mapa_5 = cutlass.Uint32(_cluster_mapa_79.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_5)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank < 2):
            while not prims.mbarrier_try_wait_parity(recv_full_addr, _phase_recv_full_0[0], time_limit=10000000, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.ACQUIRE):
                pass
            _phase_recv_full_0[0] ^= cutlass.Uint32(1)
            merged_max0[0] = cutlass.Float32(row_max0[0])
            merged_max1[0] = cutlass.Float32(row_max1[0])
            _max_230 = cute.arch.fmax(merged_max0[0], _recv_smem[(4096 + (m0_local * 4))], ftz=False)
            merged_max0[0] = cutlass.Float32(_max_230)
            _max_231 = cute.arch.fmax(merged_max1[0], _recv_smem[((4096 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1[0] = cutlass.Float32(_max_231)
            _exp2_208 = cute.math.exp2((row_max0[0] - merged_max0[0]), approx=True, ftz=True)
            _exp2_209 = cute.math.exp2((row_max1[0] - merged_max1[0]), approx=True, ftz=True)
            msum0[0] = cutlass.Float32((row_sum0[0] * _exp2_208))
            msum1[0] = cutlass.Float32((row_sum1[0] * _exp2_209))
            w_peer0 = cute.make_rmem_tensor((1,), cutlass.Float32)
            w_peer1 = cute.make_rmem_tensor((1,), cutlass.Float32)
            sbase = cutlass.Int32((4096 + (m0_local * 4)))
            _exp2_210 = cute.math.exp2((_recv_smem[sbase] - merged_max0[0]), approx=True, ftz=True)
            w_peer0[0] = cutlass.Float32(_exp2_210)
            _exp2_211 = cute.math.exp2((_recv_smem[(sbase + 2)] - merged_max1[0]), approx=True, ftz=True)
            w_peer1[0] = cutlass.Float32(_exp2_211)
            _fma_66 = cute.math.fma(_recv_smem[(sbase + 1)], w_peer0[0], msum0[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0[0] = cutlass.Float32(_fma_66)
            _fma_67 = cute.math.fma(_recv_smem[(sbase + 3)], w_peer1[0], msum1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1[0] = cutlass.Float32(_fma_67)
            my_tid4 = cutlass.Int32((tid_wg * 4))
            acc = cute.make_rmem_tensor((16,), cutlass.Float32)
            sset = cutlass.Int32(my_rank)
            fold[0] = cutlass.Int32((1 if (sset < 4) else 0))
            _if_condition_80 = cutlass.Boolean((sset < 0))
            fold[0] = cutlass.Int32(cutlass.select_(_if_condition_80, cutlass.Int32(0), fold[0]))
            _if_condition_81 = cutlass.Boolean((sset >= 2))
            fold[0] = cutlass.Int32(cutlass.select_(_if_condition_81, cutlass.Int32(0), fold[0]))
            if (fold[0] != 0):
                if (sset == 0):
                    acc[0] = cutlass.Float32((d_o[0] * (_exp2_209 if False else _exp2_208)))
                    acc[1] = cutlass.Float32((d_o[1] * (_exp2_209 if False else _exp2_208)))
                    acc[2] = cutlass.Float32((d_o[2] * (_exp2_209 if True else _exp2_208)))
                    acc[3] = cutlass.Float32((d_o[3] * (_exp2_209 if True else _exp2_208)))
                    acc[4] = cutlass.Float32((d_o[4] * (_exp2_209 if False else _exp2_208)))
                    acc[5] = cutlass.Float32((d_o[5] * (_exp2_209 if False else _exp2_208)))
                    acc[6] = cutlass.Float32((d_o[6] * (_exp2_209 if True else _exp2_208)))
                    acc[7] = cutlass.Float32((d_o[7] * (_exp2_209 if True else _exp2_208)))
                    acc[8] = cutlass.Float32((d_o[8] * (_exp2_209 if False else _exp2_208)))
                    acc[9] = cutlass.Float32((d_o[9] * (_exp2_209 if False else _exp2_208)))
                    acc[10] = cutlass.Float32((d_o[10] * (_exp2_209 if True else _exp2_208)))
                    acc[11] = cutlass.Float32((d_o[11] * (_exp2_209 if True else _exp2_208)))
                    acc[12] = cutlass.Float32((d_o[12] * (_exp2_209 if False else _exp2_208)))
                    acc[13] = cutlass.Float32((d_o[13] * (_exp2_209 if False else _exp2_208)))
                    acc[14] = cutlass.Float32((d_o[14] * (_exp2_209 if True else _exp2_208)))
                    acc[15] = cutlass.Float32((d_o[15] * (_exp2_209 if True else _exp2_208)))
                if (sset == 1):
                    acc[0] = cutlass.Float32((d_o[16] * (_exp2_209 if False else _exp2_208)))
                    acc[1] = cutlass.Float32((d_o[17] * (_exp2_209 if False else _exp2_208)))
                    acc[2] = cutlass.Float32((d_o[18] * (_exp2_209 if True else _exp2_208)))
                    acc[3] = cutlass.Float32((d_o[19] * (_exp2_209 if True else _exp2_208)))
                    acc[4] = cutlass.Float32((d_o[20] * (_exp2_209 if False else _exp2_208)))
                    acc[5] = cutlass.Float32((d_o[21] * (_exp2_209 if False else _exp2_208)))
                    acc[6] = cutlass.Float32((d_o[22] * (_exp2_209 if True else _exp2_208)))
                    acc[7] = cutlass.Float32((d_o[23] * (_exp2_209 if True else _exp2_208)))
                    acc[8] = cutlass.Float32((d_o[24] * (_exp2_209 if False else _exp2_208)))
                    acc[9] = cutlass.Float32((d_o[25] * (_exp2_209 if False else _exp2_208)))
                    acc[10] = cutlass.Float32((d_o[26] * (_exp2_209 if True else _exp2_208)))
                    acc[11] = cutlass.Float32((d_o[27] * (_exp2_209 if True else _exp2_208)))
                    acc[12] = cutlass.Float32((d_o[28] * (_exp2_209 if False else _exp2_208)))
                    acc[13] = cutlass.Float32((d_o[29] * (_exp2_209 if False else _exp2_208)))
                    acc[14] = cutlass.Float32((d_o[30] * (_exp2_209 if True else _exp2_208)))
                    acc[15] = cutlass.Float32((d_o[31] * (_exp2_209 if True else _exp2_208)))
                base = cutlass.Int32(my_tid4)
                _fma_68 = cute.math.fma(_recv_smem[base], (w_peer1[0] if False else w_peer0[0]), acc[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[0] = cutlass.Float32(_fma_68)
                _fma_69 = cute.math.fma(_recv_smem[(base + 1)], (w_peer1[0] if False else w_peer0[0]), acc[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[1] = cutlass.Float32(_fma_69)
                _fma_70 = cute.math.fma(_recv_smem[(base + 2)], (w_peer1[0] if True else w_peer0[0]), acc[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[2] = cutlass.Float32(_fma_70)
                _fma_71 = cute.math.fma(_recv_smem[(base + 3)], (w_peer1[0] if True else w_peer0[0]), acc[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[3] = cutlass.Float32(_fma_71)
                _fma_72 = cute.math.fma(_recv_smem[(base + 512)], (w_peer1[0] if False else w_peer0[0]), acc[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[4] = cutlass.Float32(_fma_72)
                _fma_73 = cute.math.fma(_recv_smem[((base + 512) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[5] = cutlass.Float32(_fma_73)
                _fma_74 = cute.math.fma(_recv_smem[((base + 512) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[6] = cutlass.Float32(_fma_74)
                _fma_75 = cute.math.fma(_recv_smem[((base + 512) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[7] = cutlass.Float32(_fma_75)
                _fma_76 = cute.math.fma(_recv_smem[(base + 1024)], (w_peer1[0] if False else w_peer0[0]), acc[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[8] = cutlass.Float32(_fma_76)
                _fma_77 = cute.math.fma(_recv_smem[((base + 1024) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[9] = cutlass.Float32(_fma_77)
                _fma_78 = cute.math.fma(_recv_smem[((base + 1024) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[10] = cutlass.Float32(_fma_78)
                _fma_79 = cute.math.fma(_recv_smem[((base + 1024) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[11] = cutlass.Float32(_fma_79)
                _fma_80 = cute.math.fma(_recv_smem[(base + 1536)], (w_peer1[0] if False else w_peer0[0]), acc[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[12] = cutlass.Float32(_fma_80)
                _fma_81 = cute.math.fma(_recv_smem[((base + 1536) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[13] = cutlass.Float32(_fma_81)
                _fma_82 = cute.math.fma(_recv_smem[((base + 1536) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[14] = cutlass.Float32(_fma_82)
                _fma_83 = cute.math.fma(_recv_smem[((base + 1536) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[15] = cutlass.Float32(_fma_83)
                _rcp_0 = cute.math.rcp(msum0[0], approx=True, ftz=True)
                _rcp_1 = cute.math.rcp(msum1[0], approx=True, ftz=True)
                acc[0] = cutlass.Float32((acc[0] * (_rcp_1 if False else _rcp_0)))
                acc[1] = cutlass.Float32((acc[1] * (_rcp_1 if False else _rcp_0)))
                acc[2] = cutlass.Float32((acc[2] * (_rcp_1 if True else _rcp_0)))
                acc[3] = cutlass.Float32((acc[3] * (_rcp_1 if True else _rcp_0)))
                acc[4] = cutlass.Float32((acc[4] * (_rcp_1 if False else _rcp_0)))
                acc[5] = cutlass.Float32((acc[5] * (_rcp_1 if False else _rcp_0)))
                acc[6] = cutlass.Float32((acc[6] * (_rcp_1 if True else _rcp_0)))
                acc[7] = cutlass.Float32((acc[7] * (_rcp_1 if True else _rcp_0)))
                acc[8] = cutlass.Float32((acc[8] * (_rcp_1 if False else _rcp_0)))
                acc[9] = cutlass.Float32((acc[9] * (_rcp_1 if False else _rcp_0)))
                acc[10] = cutlass.Float32((acc[10] * (_rcp_1 if True else _rcp_0)))
                acc[11] = cutlass.Float32((acc[11] * (_rcp_1 if True else _rcp_0)))
                acc[12] = cutlass.Float32((acc[12] * (_rcp_1 if False else _rcp_0)))
                acc[13] = cutlass.Float32((acc[13] * (_rcp_1 if False else _rcp_0)))
                acc[14] = cutlass.Float32((acc[14] * (_rcp_1 if True else _rcp_0)))
                acc[15] = cutlass.Float32((acc[15] * (_rcp_1 if True else _rcp_0)))
                qj1 = cutlass.Int32((qj & 1))
                qj2 = cutlass.Int32((qj & 2))
                o_vec = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base = cutlass.Int32((q_row * 128))
                m_local_r = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_96 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[1]), cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_96)
                _bf16x2_97 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[1]), cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_97)
                _bf16x2_98 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[1]), cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_98)
                _bf16x2_99 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[1]), cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_99)
                _shfl_xor_28 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_28)
                _shfl_xor_29 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_29)
                _shfl_xor_30 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_30)
                _shfl_xor_31 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_31)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_32 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_32)
                _shfl_xor_33 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_33)
                _shfl_xor_34 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_34)
                _shfl_xor_35 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_35)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off = cutlass.Int32(((o_row_base + (m_local_r * 128)) + (((4 * sset) + qj) * 8)))
                _gmem_store_raw_82 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_82.ir_value(), O + o_off)
                m_local_r_0 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_100 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[1]), cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_100)
                _bf16x2_101 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[1]), cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_101)
                _bf16x2_102 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[1]), cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_102)
                _bf16x2_103 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[1]), cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_103)
                _shfl_xor_36 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_36)
                _shfl_xor_37 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_37)
                _shfl_xor_38 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_38)
                _shfl_xor_39 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_39)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_40 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_40)
                _shfl_xor_41 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_41)
                _shfl_xor_42 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_42)
                _shfl_xor_43 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_43)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_1 = cutlass.Int32(((o_row_base + (m_local_r_0 * 128)) + (((4 * sset) + qj) * 8)))
                _gmem_store_raw_83 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_83.ir_value(), O + o_off_1)
            sset_0 = cutlass.Int32((my_rank + 2))
            fold_1[0] = cutlass.Int32((1 if (sset_0 < 4) else 0))
            _if_condition_84 = cutlass.Boolean((sset_0 < 0))
            fold_1[0] = cutlass.Int32(cutlass.select_(_if_condition_84, cutlass.Int32(0), fold_1[0]))
            _if_condition_85 = cutlass.Boolean((sset_0 >= 2))
            fold_1[0] = cutlass.Int32(cutlass.select_(_if_condition_85, cutlass.Int32(0), fold_1[0]))
            if (fold_1[0] != 0):
                if (sset_0 == 0):
                    acc[0] = cutlass.Float32((d_o[0] * (_exp2_209 if False else _exp2_208)))
                    acc[1] = cutlass.Float32((d_o[1] * (_exp2_209 if False else _exp2_208)))
                    acc[2] = cutlass.Float32((d_o[2] * (_exp2_209 if True else _exp2_208)))
                    acc[3] = cutlass.Float32((d_o[3] * (_exp2_209 if True else _exp2_208)))
                    acc[4] = cutlass.Float32((d_o[4] * (_exp2_209 if False else _exp2_208)))
                    acc[5] = cutlass.Float32((d_o[5] * (_exp2_209 if False else _exp2_208)))
                    acc[6] = cutlass.Float32((d_o[6] * (_exp2_209 if True else _exp2_208)))
                    acc[7] = cutlass.Float32((d_o[7] * (_exp2_209 if True else _exp2_208)))
                    acc[8] = cutlass.Float32((d_o[8] * (_exp2_209 if False else _exp2_208)))
                    acc[9] = cutlass.Float32((d_o[9] * (_exp2_209 if False else _exp2_208)))
                    acc[10] = cutlass.Float32((d_o[10] * (_exp2_209 if True else _exp2_208)))
                    acc[11] = cutlass.Float32((d_o[11] * (_exp2_209 if True else _exp2_208)))
                    acc[12] = cutlass.Float32((d_o[12] * (_exp2_209 if False else _exp2_208)))
                    acc[13] = cutlass.Float32((d_o[13] * (_exp2_209 if False else _exp2_208)))
                    acc[14] = cutlass.Float32((d_o[14] * (_exp2_209 if True else _exp2_208)))
                    acc[15] = cutlass.Float32((d_o[15] * (_exp2_209 if True else _exp2_208)))
                if (sset_0 == 1):
                    acc[0] = cutlass.Float32((d_o[16] * (_exp2_209 if False else _exp2_208)))
                    acc[1] = cutlass.Float32((d_o[17] * (_exp2_209 if False else _exp2_208)))
                    acc[2] = cutlass.Float32((d_o[18] * (_exp2_209 if True else _exp2_208)))
                    acc[3] = cutlass.Float32((d_o[19] * (_exp2_209 if True else _exp2_208)))
                    acc[4] = cutlass.Float32((d_o[20] * (_exp2_209 if False else _exp2_208)))
                    acc[5] = cutlass.Float32((d_o[21] * (_exp2_209 if False else _exp2_208)))
                    acc[6] = cutlass.Float32((d_o[22] * (_exp2_209 if True else _exp2_208)))
                    acc[7] = cutlass.Float32((d_o[23] * (_exp2_209 if True else _exp2_208)))
                    acc[8] = cutlass.Float32((d_o[24] * (_exp2_209 if False else _exp2_208)))
                    acc[9] = cutlass.Float32((d_o[25] * (_exp2_209 if False else _exp2_208)))
                    acc[10] = cutlass.Float32((d_o[26] * (_exp2_209 if True else _exp2_208)))
                    acc[11] = cutlass.Float32((d_o[27] * (_exp2_209 if True else _exp2_208)))
                    acc[12] = cutlass.Float32((d_o[28] * (_exp2_209 if False else _exp2_208)))
                    acc[13] = cutlass.Float32((d_o[29] * (_exp2_209 if False else _exp2_208)))
                    acc[14] = cutlass.Float32((d_o[30] * (_exp2_209 if True else _exp2_208)))
                    acc[15] = cutlass.Float32((d_o[31] * (_exp2_209 if True else _exp2_208)))
                base_1 = cutlass.Int32((2048 + my_tid4))
                _fma_84 = cute.math.fma(_recv_smem[base_1], (w_peer1[0] if False else w_peer0[0]), acc[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[0] = cutlass.Float32(_fma_84)
                _fma_85 = cute.math.fma(_recv_smem[(base_1 + 1)], (w_peer1[0] if False else w_peer0[0]), acc[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[1] = cutlass.Float32(_fma_85)
                _fma_86 = cute.math.fma(_recv_smem[(base_1 + 2)], (w_peer1[0] if True else w_peer0[0]), acc[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[2] = cutlass.Float32(_fma_86)
                _fma_87 = cute.math.fma(_recv_smem[(base_1 + 3)], (w_peer1[0] if True else w_peer0[0]), acc[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[3] = cutlass.Float32(_fma_87)
                _fma_88 = cute.math.fma(_recv_smem[(base_1 + 512)], (w_peer1[0] if False else w_peer0[0]), acc[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[4] = cutlass.Float32(_fma_88)
                _fma_89 = cute.math.fma(_recv_smem[((base_1 + 512) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[5] = cutlass.Float32(_fma_89)
                _fma_90 = cute.math.fma(_recv_smem[((base_1 + 512) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[6] = cutlass.Float32(_fma_90)
                _fma_91 = cute.math.fma(_recv_smem[((base_1 + 512) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[7] = cutlass.Float32(_fma_91)
                _fma_92 = cute.math.fma(_recv_smem[(base_1 + 1024)], (w_peer1[0] if False else w_peer0[0]), acc[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[8] = cutlass.Float32(_fma_92)
                _fma_93 = cute.math.fma(_recv_smem[((base_1 + 1024) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[9] = cutlass.Float32(_fma_93)
                _fma_94 = cute.math.fma(_recv_smem[((base_1 + 1024) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[10] = cutlass.Float32(_fma_94)
                _fma_95 = cute.math.fma(_recv_smem[((base_1 + 1024) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[11] = cutlass.Float32(_fma_95)
                _fma_96 = cute.math.fma(_recv_smem[(base_1 + 1536)], (w_peer1[0] if False else w_peer0[0]), acc[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[12] = cutlass.Float32(_fma_96)
                _fma_97 = cute.math.fma(_recv_smem[((base_1 + 1536) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[13] = cutlass.Float32(_fma_97)
                _fma_98 = cute.math.fma(_recv_smem[((base_1 + 1536) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[14] = cutlass.Float32(_fma_98)
                _fma_99 = cute.math.fma(_recv_smem[((base_1 + 1536) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[15] = cutlass.Float32(_fma_99)
                _rcp_2 = cute.math.rcp(msum0[0], approx=True, ftz=True)
                _rcp_3 = cute.math.rcp(msum1[0], approx=True, ftz=True)
                acc[0] = cutlass.Float32((acc[0] * (_rcp_3 if False else _rcp_2)))
                acc[1] = cutlass.Float32((acc[1] * (_rcp_3 if False else _rcp_2)))
                acc[2] = cutlass.Float32((acc[2] * (_rcp_3 if True else _rcp_2)))
                acc[3] = cutlass.Float32((acc[3] * (_rcp_3 if True else _rcp_2)))
                acc[4] = cutlass.Float32((acc[4] * (_rcp_3 if False else _rcp_2)))
                acc[5] = cutlass.Float32((acc[5] * (_rcp_3 if False else _rcp_2)))
                acc[6] = cutlass.Float32((acc[6] * (_rcp_3 if True else _rcp_2)))
                acc[7] = cutlass.Float32((acc[7] * (_rcp_3 if True else _rcp_2)))
                acc[8] = cutlass.Float32((acc[8] * (_rcp_3 if False else _rcp_2)))
                acc[9] = cutlass.Float32((acc[9] * (_rcp_3 if False else _rcp_2)))
                acc[10] = cutlass.Float32((acc[10] * (_rcp_3 if True else _rcp_2)))
                acc[11] = cutlass.Float32((acc[11] * (_rcp_3 if True else _rcp_2)))
                acc[12] = cutlass.Float32((acc[12] * (_rcp_3 if False else _rcp_2)))
                acc[13] = cutlass.Float32((acc[13] * (_rcp_3 if False else _rcp_2)))
                acc[14] = cutlass.Float32((acc[14] * (_rcp_3 if True else _rcp_2)))
                acc[15] = cutlass.Float32((acc[15] * (_rcp_3 if True else _rcp_2)))
                qj1_1 = cutlass.Int32((qj & 1))
                qj2_1 = cutlass.Int32((qj & 2))
                o_vec_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base_1 = cutlass.Int32((q_row * 128))
                m_local_r_1 = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_104 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[1]), cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[0] = cutlass.Uint32(_bf16x2_104)
                _bf16x2_105 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[1]), cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[1] = cutlass.Uint32(_bf16x2_105)
                _bf16x2_106 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[1]), cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[2] = cutlass.Uint32(_bf16x2_106)
                _bf16x2_107 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[1]), cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[3] = cutlass.Uint32(_bf16x2_107)
                _shfl_xor_44 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_44)
                _shfl_xor_45 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_45)
                _shfl_xor_46 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_46)
                _shfl_xor_47 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_47)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj1_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj1_1 == 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj1_1 != 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj1_1 == 0) else o_vec_1[3]))
                _shfl_xor_48 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_48)
                _shfl_xor_49 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_49)
                _shfl_xor_50 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_50)
                _shfl_xor_51 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_51)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj2_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj2_1 != 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj2_1 == 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj2_1 == 0) else o_vec_1[3]))
                o_off_2 = cutlass.Int32(((o_row_base_1 + (m_local_r_1 * 128)) + (((4 * sset_0) + qj) * 8)))
                _gmem_store_raw_86 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_1[0]), cutlass.Uint32(o_vec_1[1]), cutlass.Uint32(o_vec_1[2]), cutlass.Uint32(o_vec_1[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_86.ir_value(), O + o_off_2)
                m_local_r_0_1 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_108 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[1]), cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[0] = cutlass.Uint32(_bf16x2_108)
                _bf16x2_109 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[1]), cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[1] = cutlass.Uint32(_bf16x2_109)
                _bf16x2_110 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[1]), cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[2] = cutlass.Uint32(_bf16x2_110)
                _bf16x2_111 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[1]), cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[3] = cutlass.Uint32(_bf16x2_111)
                _shfl_xor_52 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_52)
                _shfl_xor_53 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_53)
                _shfl_xor_54 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_54)
                _shfl_xor_55 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_55)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj1_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj1_1 == 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj1_1 != 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj1_1 == 0) else o_vec_1[3]))
                _shfl_xor_56 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_56)
                _shfl_xor_57 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_57)
                _shfl_xor_58 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_58)
                _shfl_xor_59 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_59)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj2_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj2_1 != 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj2_1 == 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj2_1 == 0) else o_vec_1[3]))
                o_off_1_1 = cutlass.Int32(((o_row_base_1 + (m_local_r_0_1 * 128)) + (((4 * sset_0) + qj) * 8)))
                _gmem_store_raw_87 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_1[0]), cutlass.Uint32(o_vec_1[1]), cutlass.Uint32(o_vec_1[2]), cutlass.Uint32(o_vec_1[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_87.ir_value(), O + o_off_1_1)
    if (wg == 1):
        my_rank_1 = cutlass.Int32(cta_rank)
        my_off_1 = cutlass.Int32((tid_wg * 16))
        if (my_rank_1 != 0):
            pslot_2[0] = cutlass.Int32(my_rank_1)
            _if_condition_88 = cutlass.Boolean((my_rank_1 > 0))
            pslot_2[0] = cutlass.Int32(cutlass.select_(_if_condition_88, cutlass.Int32((my_rank_1 - 1)), pslot_2[0]))
            _cluster_mapa_89 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(0))
            _mapa_6 = cutlass.Uint32(_cluster_mapa_89.toint())
            _cluster_mapa_90 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_2[0] * 17408))) + 8192) + cutlass.Uint32(my_off_1))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(0))
            _mapa_7 = cutlass.Uint32(_cluster_mapa_90.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_7)).ir_value(), (cutlass.Float32(d_o[32]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[33]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[34]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[35]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_6)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_7 + 2048))).ir_value(), (cutlass.Float32(d_o[36]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[37]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[38]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[39]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_6)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_7 + 4096))).ir_value(), (cutlass.Float32(d_o[40]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[41]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[42]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[43]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_6)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_7 + 6144))).ir_value(), (cutlass.Float32(d_o[44]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[45]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[46]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[47]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_6)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
        if (my_rank_1 != 1):
            pslot_3[0] = cutlass.Int32(my_rank_1)
            _if_condition_91 = cutlass.Boolean((my_rank_1 > 1))
            pslot_3[0] = cutlass.Int32(cutlass.select_(_if_condition_91, cutlass.Int32((my_rank_1 - 1)), pslot_3[0]))
            _cluster_mapa_92 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(1))
            _mapa_8 = cutlass.Uint32(_cluster_mapa_92.toint())
            _cluster_mapa_93 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_3[0] * 17408))) + 8192) + cutlass.Uint32(my_off_1))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(1))
            _mapa_9 = cutlass.Uint32(_cluster_mapa_93.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_9)).ir_value(), (cutlass.Float32(d_o[48]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[49]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[50]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[51]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_8)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_9 + 2048))).ir_value(), (cutlass.Float32(d_o[52]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[53]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[54]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[55]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_8)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_9 + 4096))).ir_value(), (cutlass.Float32(d_o[56]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[57]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[58]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[59]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_8)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_9 + 6144))).ir_value(), (cutlass.Float32(d_o[60]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[61]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[62]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[63]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_8)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
        if (my_rank_1 < 2):
            while not prims.mbarrier_try_wait_parity(recv_full_addr, _phase_recv_full_0[0], time_limit=10000000, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.ACQUIRE):
                pass
            _phase_recv_full_0[0] ^= cutlass.Uint32(1)
            merged_max0_1[0] = cutlass.Float32(row_max0[0])
            merged_max1_1[0] = cutlass.Float32(row_max1[0])
            _max_232 = cute.arch.fmax(merged_max0_1[0], _recv_smem[(4096 + (m0_local * 4))], ftz=False)
            merged_max0_1[0] = cutlass.Float32(_max_232)
            _max_233 = cute.arch.fmax(merged_max1_1[0], _recv_smem[((4096 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1_1[0] = cutlass.Float32(_max_233)
            _exp2_212 = cute.math.exp2((row_max0[0] - merged_max0_1[0]), approx=True, ftz=True)
            _exp2_213 = cute.math.exp2((row_max1[0] - merged_max1_1[0]), approx=True, ftz=True)
            msum0_1[0] = cutlass.Float32((row_sum0[0] * _exp2_212))
            msum1_1[0] = cutlass.Float32((row_sum1[0] * _exp2_213))
            w_peer0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
            w_peer1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
            sbase_1 = cutlass.Int32((4096 + (m0_local * 4)))
            _exp2_214 = cute.math.exp2((_recv_smem[sbase_1] - merged_max0_1[0]), approx=True, ftz=True)
            w_peer0_1[0] = cutlass.Float32(_exp2_214)
            _exp2_215 = cute.math.exp2((_recv_smem[(sbase_1 + 2)] - merged_max1_1[0]), approx=True, ftz=True)
            w_peer1_1[0] = cutlass.Float32(_exp2_215)
            _fma_100 = cute.math.fma(_recv_smem[(sbase_1 + 1)], w_peer0_1[0], msum0_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0_1[0] = cutlass.Float32(_fma_100)
            _fma_101 = cute.math.fma(_recv_smem[(sbase_1 + 3)], w_peer1_1[0], msum1_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1_1[0] = cutlass.Float32(_fma_101)
            my_tid4_1 = cutlass.Int32((tid_wg * 4))
            acc_1 = cute.make_rmem_tensor((16,), cutlass.Float32)
            sset_1 = cutlass.Int32(my_rank_1)
            fold_2[0] = cutlass.Int32((1 if (sset_1 < 4) else 0))
            _if_condition_94 = cutlass.Boolean((sset_1 < 2))
            fold_2[0] = cutlass.Int32(cutlass.select_(_if_condition_94, cutlass.Int32(0), fold_2[0]))
            _if_condition_95 = cutlass.Boolean((sset_1 >= 4))
            fold_2[0] = cutlass.Int32(cutlass.select_(_if_condition_95, cutlass.Int32(0), fold_2[0]))
            if (fold_2[0] != 0):
                if (sset_1 == 2):
                    acc_1[0] = cutlass.Float32((d_o[32] * (_exp2_213 if False else _exp2_212)))
                    acc_1[1] = cutlass.Float32((d_o[33] * (_exp2_213 if False else _exp2_212)))
                    acc_1[2] = cutlass.Float32((d_o[34] * (_exp2_213 if True else _exp2_212)))
                    acc_1[3] = cutlass.Float32((d_o[35] * (_exp2_213 if True else _exp2_212)))
                    acc_1[4] = cutlass.Float32((d_o[36] * (_exp2_213 if False else _exp2_212)))
                    acc_1[5] = cutlass.Float32((d_o[37] * (_exp2_213 if False else _exp2_212)))
                    acc_1[6] = cutlass.Float32((d_o[38] * (_exp2_213 if True else _exp2_212)))
                    acc_1[7] = cutlass.Float32((d_o[39] * (_exp2_213 if True else _exp2_212)))
                    acc_1[8] = cutlass.Float32((d_o[40] * (_exp2_213 if False else _exp2_212)))
                    acc_1[9] = cutlass.Float32((d_o[41] * (_exp2_213 if False else _exp2_212)))
                    acc_1[10] = cutlass.Float32((d_o[42] * (_exp2_213 if True else _exp2_212)))
                    acc_1[11] = cutlass.Float32((d_o[43] * (_exp2_213 if True else _exp2_212)))
                    acc_1[12] = cutlass.Float32((d_o[44] * (_exp2_213 if False else _exp2_212)))
                    acc_1[13] = cutlass.Float32((d_o[45] * (_exp2_213 if False else _exp2_212)))
                    acc_1[14] = cutlass.Float32((d_o[46] * (_exp2_213 if True else _exp2_212)))
                    acc_1[15] = cutlass.Float32((d_o[47] * (_exp2_213 if True else _exp2_212)))
                if (sset_1 == 3):
                    acc_1[0] = cutlass.Float32((d_o[48] * (_exp2_213 if False else _exp2_212)))
                    acc_1[1] = cutlass.Float32((d_o[49] * (_exp2_213 if False else _exp2_212)))
                    acc_1[2] = cutlass.Float32((d_o[50] * (_exp2_213 if True else _exp2_212)))
                    acc_1[3] = cutlass.Float32((d_o[51] * (_exp2_213 if True else _exp2_212)))
                    acc_1[4] = cutlass.Float32((d_o[52] * (_exp2_213 if False else _exp2_212)))
                    acc_1[5] = cutlass.Float32((d_o[53] * (_exp2_213 if False else _exp2_212)))
                    acc_1[6] = cutlass.Float32((d_o[54] * (_exp2_213 if True else _exp2_212)))
                    acc_1[7] = cutlass.Float32((d_o[55] * (_exp2_213 if True else _exp2_212)))
                    acc_1[8] = cutlass.Float32((d_o[56] * (_exp2_213 if False else _exp2_212)))
                    acc_1[9] = cutlass.Float32((d_o[57] * (_exp2_213 if False else _exp2_212)))
                    acc_1[10] = cutlass.Float32((d_o[58] * (_exp2_213 if True else _exp2_212)))
                    acc_1[11] = cutlass.Float32((d_o[59] * (_exp2_213 if True else _exp2_212)))
                    acc_1[12] = cutlass.Float32((d_o[60] * (_exp2_213 if False else _exp2_212)))
                    acc_1[13] = cutlass.Float32((d_o[61] * (_exp2_213 if False else _exp2_212)))
                    acc_1[14] = cutlass.Float32((d_o[62] * (_exp2_213 if True else _exp2_212)))
                    acc_1[15] = cutlass.Float32((d_o[63] * (_exp2_213 if True else _exp2_212)))
                base_2 = cutlass.Int32(my_tid4_1)
                _fma_102 = cute.math.fma(_recv_smem[base_2], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[0] = cutlass.Float32(_fma_102)
                _fma_103 = cute.math.fma(_recv_smem[(base_2 + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[1] = cutlass.Float32(_fma_103)
                _fma_104 = cute.math.fma(_recv_smem[(base_2 + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[2] = cutlass.Float32(_fma_104)
                _fma_105 = cute.math.fma(_recv_smem[(base_2 + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[3] = cutlass.Float32(_fma_105)
                _fma_106 = cute.math.fma(_recv_smem[(base_2 + 512)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[4] = cutlass.Float32(_fma_106)
                _fma_107 = cute.math.fma(_recv_smem[((base_2 + 512) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[5] = cutlass.Float32(_fma_107)
                _fma_108 = cute.math.fma(_recv_smem[((base_2 + 512) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[6] = cutlass.Float32(_fma_108)
                _fma_109 = cute.math.fma(_recv_smem[((base_2 + 512) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[7] = cutlass.Float32(_fma_109)
                _fma_110 = cute.math.fma(_recv_smem[(base_2 + 1024)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[8] = cutlass.Float32(_fma_110)
                _fma_111 = cute.math.fma(_recv_smem[((base_2 + 1024) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[9] = cutlass.Float32(_fma_111)
                _fma_112 = cute.math.fma(_recv_smem[((base_2 + 1024) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[10] = cutlass.Float32(_fma_112)
                _fma_113 = cute.math.fma(_recv_smem[((base_2 + 1024) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[11] = cutlass.Float32(_fma_113)
                _fma_114 = cute.math.fma(_recv_smem[(base_2 + 1536)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[12] = cutlass.Float32(_fma_114)
                _fma_115 = cute.math.fma(_recv_smem[((base_2 + 1536) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[13] = cutlass.Float32(_fma_115)
                _fma_116 = cute.math.fma(_recv_smem[((base_2 + 1536) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[14] = cutlass.Float32(_fma_116)
                _fma_117 = cute.math.fma(_recv_smem[((base_2 + 1536) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[15] = cutlass.Float32(_fma_117)
                _rcp_4 = cute.math.rcp(msum0_1[0], approx=True, ftz=True)
                _rcp_5 = cute.math.rcp(msum1_1[0], approx=True, ftz=True)
                acc_1[0] = cutlass.Float32((acc_1[0] * (_rcp_5 if False else _rcp_4)))
                acc_1[1] = cutlass.Float32((acc_1[1] * (_rcp_5 if False else _rcp_4)))
                acc_1[2] = cutlass.Float32((acc_1[2] * (_rcp_5 if True else _rcp_4)))
                acc_1[3] = cutlass.Float32((acc_1[3] * (_rcp_5 if True else _rcp_4)))
                acc_1[4] = cutlass.Float32((acc_1[4] * (_rcp_5 if False else _rcp_4)))
                acc_1[5] = cutlass.Float32((acc_1[5] * (_rcp_5 if False else _rcp_4)))
                acc_1[6] = cutlass.Float32((acc_1[6] * (_rcp_5 if True else _rcp_4)))
                acc_1[7] = cutlass.Float32((acc_1[7] * (_rcp_5 if True else _rcp_4)))
                acc_1[8] = cutlass.Float32((acc_1[8] * (_rcp_5 if False else _rcp_4)))
                acc_1[9] = cutlass.Float32((acc_1[9] * (_rcp_5 if False else _rcp_4)))
                acc_1[10] = cutlass.Float32((acc_1[10] * (_rcp_5 if True else _rcp_4)))
                acc_1[11] = cutlass.Float32((acc_1[11] * (_rcp_5 if True else _rcp_4)))
                acc_1[12] = cutlass.Float32((acc_1[12] * (_rcp_5 if False else _rcp_4)))
                acc_1[13] = cutlass.Float32((acc_1[13] * (_rcp_5 if False else _rcp_4)))
                acc_1[14] = cutlass.Float32((acc_1[14] * (_rcp_5 if True else _rcp_4)))
                acc_1[15] = cutlass.Float32((acc_1[15] * (_rcp_5 if True else _rcp_4)))
                qj1_2 = cutlass.Int32((qj & 1))
                qj2_2 = cutlass.Int32((qj & 2))
                o_vec_2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp_2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base_2 = cutlass.Int32((q_row * 128))
                m_local_r_2 = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_112 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[0] = cutlass.Uint32(_bf16x2_112)
                _bf16x2_113 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[1] = cutlass.Uint32(_bf16x2_113)
                _bf16x2_114 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[2] = cutlass.Uint32(_bf16x2_114)
                _bf16x2_115 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[3] = cutlass.Uint32(_bf16x2_115)
                _shfl_xor_60 = cute.arch.shuffle_sync_bfly(o_vec_2[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[0] = cutlass.Uint32(_shfl_xor_60)
                _shfl_xor_61 = cute.arch.shuffle_sync_bfly(o_vec_2[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[1] = cutlass.Uint32(_shfl_xor_61)
                _shfl_xor_62 = cute.arch.shuffle_sync_bfly(o_vec_2[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[2] = cutlass.Uint32(_shfl_xor_62)
                _shfl_xor_63 = cute.arch.shuffle_sync_bfly(o_vec_2[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[3] = cutlass.Uint32(_shfl_xor_63)
                o_vec_2[0] = cutlass.Uint32((o_tmp_2[0] if (qj1_2 != 0) else o_vec_2[0]))
                o_vec_2[1] = cutlass.Uint32((o_tmp_2[1] if (qj1_2 == 0) else o_vec_2[1]))
                o_vec_2[2] = cutlass.Uint32((o_tmp_2[2] if (qj1_2 != 0) else o_vec_2[2]))
                o_vec_2[3] = cutlass.Uint32((o_tmp_2[3] if (qj1_2 == 0) else o_vec_2[3]))
                _shfl_xor_64 = cute.arch.shuffle_sync_bfly(o_vec_2[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[0] = cutlass.Uint32(_shfl_xor_64)
                _shfl_xor_65 = cute.arch.shuffle_sync_bfly(o_vec_2[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[1] = cutlass.Uint32(_shfl_xor_65)
                _shfl_xor_66 = cute.arch.shuffle_sync_bfly(o_vec_2[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[2] = cutlass.Uint32(_shfl_xor_66)
                _shfl_xor_67 = cute.arch.shuffle_sync_bfly(o_vec_2[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[3] = cutlass.Uint32(_shfl_xor_67)
                o_vec_2[0] = cutlass.Uint32((o_tmp_2[0] if (qj2_2 != 0) else o_vec_2[0]))
                o_vec_2[1] = cutlass.Uint32((o_tmp_2[1] if (qj2_2 != 0) else o_vec_2[1]))
                o_vec_2[2] = cutlass.Uint32((o_tmp_2[2] if (qj2_2 == 0) else o_vec_2[2]))
                o_vec_2[3] = cutlass.Uint32((o_tmp_2[3] if (qj2_2 == 0) else o_vec_2[3]))
                o_off_3 = cutlass.Int32(((o_row_base_2 + (m_local_r_2 * 128)) + (((4 * sset_1) + qj) * 8)))
                _gmem_store_raw_96 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_2[0]), cutlass.Uint32(o_vec_2[1]), cutlass.Uint32(o_vec_2[2]), cutlass.Uint32(o_vec_2[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_96.ir_value(), O + o_off_3)
                m_local_r_0_2 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_116 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[0] = cutlass.Uint32(_bf16x2_116)
                _bf16x2_117 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[1] = cutlass.Uint32(_bf16x2_117)
                _bf16x2_118 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[2] = cutlass.Uint32(_bf16x2_118)
                _bf16x2_119 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_2[3] = cutlass.Uint32(_bf16x2_119)
                _shfl_xor_68 = cute.arch.shuffle_sync_bfly(o_vec_2[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[0] = cutlass.Uint32(_shfl_xor_68)
                _shfl_xor_69 = cute.arch.shuffle_sync_bfly(o_vec_2[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[1] = cutlass.Uint32(_shfl_xor_69)
                _shfl_xor_70 = cute.arch.shuffle_sync_bfly(o_vec_2[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[2] = cutlass.Uint32(_shfl_xor_70)
                _shfl_xor_71 = cute.arch.shuffle_sync_bfly(o_vec_2[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[3] = cutlass.Uint32(_shfl_xor_71)
                o_vec_2[0] = cutlass.Uint32((o_tmp_2[0] if (qj1_2 != 0) else o_vec_2[0]))
                o_vec_2[1] = cutlass.Uint32((o_tmp_2[1] if (qj1_2 == 0) else o_vec_2[1]))
                o_vec_2[2] = cutlass.Uint32((o_tmp_2[2] if (qj1_2 != 0) else o_vec_2[2]))
                o_vec_2[3] = cutlass.Uint32((o_tmp_2[3] if (qj1_2 == 0) else o_vec_2[3]))
                _shfl_xor_72 = cute.arch.shuffle_sync_bfly(o_vec_2[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[0] = cutlass.Uint32(_shfl_xor_72)
                _shfl_xor_73 = cute.arch.shuffle_sync_bfly(o_vec_2[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[1] = cutlass.Uint32(_shfl_xor_73)
                _shfl_xor_74 = cute.arch.shuffle_sync_bfly(o_vec_2[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[2] = cutlass.Uint32(_shfl_xor_74)
                _shfl_xor_75 = cute.arch.shuffle_sync_bfly(o_vec_2[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_2[3] = cutlass.Uint32(_shfl_xor_75)
                o_vec_2[0] = cutlass.Uint32((o_tmp_2[0] if (qj2_2 != 0) else o_vec_2[0]))
                o_vec_2[1] = cutlass.Uint32((o_tmp_2[1] if (qj2_2 != 0) else o_vec_2[1]))
                o_vec_2[2] = cutlass.Uint32((o_tmp_2[2] if (qj2_2 == 0) else o_vec_2[2]))
                o_vec_2[3] = cutlass.Uint32((o_tmp_2[3] if (qj2_2 == 0) else o_vec_2[3]))
                o_off_1_2 = cutlass.Int32(((o_row_base_2 + (m_local_r_0_2 * 128)) + (((4 * sset_1) + qj) * 8)))
                _gmem_store_raw_97 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_2[0]), cutlass.Uint32(o_vec_2[1]), cutlass.Uint32(o_vec_2[2]), cutlass.Uint32(o_vec_2[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_97.ir_value(), O + o_off_1_2)
            sset_0_1 = cutlass.Int32((my_rank_1 + 2))
            fold_1_1[0] = cutlass.Int32((1 if (sset_0_1 < 4) else 0))
            _if_condition_98 = cutlass.Boolean((sset_0_1 < 2))
            fold_1_1[0] = cutlass.Int32(cutlass.select_(_if_condition_98, cutlass.Int32(0), fold_1_1[0]))
            _if_condition_99 = cutlass.Boolean((sset_0_1 >= 4))
            fold_1_1[0] = cutlass.Int32(cutlass.select_(_if_condition_99, cutlass.Int32(0), fold_1_1[0]))
            if (fold_1_1[0] != 0):
                if (sset_0_1 == 2):
                    acc_1[0] = cutlass.Float32((d_o[32] * (_exp2_213 if False else _exp2_212)))
                    acc_1[1] = cutlass.Float32((d_o[33] * (_exp2_213 if False else _exp2_212)))
                    acc_1[2] = cutlass.Float32((d_o[34] * (_exp2_213 if True else _exp2_212)))
                    acc_1[3] = cutlass.Float32((d_o[35] * (_exp2_213 if True else _exp2_212)))
                    acc_1[4] = cutlass.Float32((d_o[36] * (_exp2_213 if False else _exp2_212)))
                    acc_1[5] = cutlass.Float32((d_o[37] * (_exp2_213 if False else _exp2_212)))
                    acc_1[6] = cutlass.Float32((d_o[38] * (_exp2_213 if True else _exp2_212)))
                    acc_1[7] = cutlass.Float32((d_o[39] * (_exp2_213 if True else _exp2_212)))
                    acc_1[8] = cutlass.Float32((d_o[40] * (_exp2_213 if False else _exp2_212)))
                    acc_1[9] = cutlass.Float32((d_o[41] * (_exp2_213 if False else _exp2_212)))
                    acc_1[10] = cutlass.Float32((d_o[42] * (_exp2_213 if True else _exp2_212)))
                    acc_1[11] = cutlass.Float32((d_o[43] * (_exp2_213 if True else _exp2_212)))
                    acc_1[12] = cutlass.Float32((d_o[44] * (_exp2_213 if False else _exp2_212)))
                    acc_1[13] = cutlass.Float32((d_o[45] * (_exp2_213 if False else _exp2_212)))
                    acc_1[14] = cutlass.Float32((d_o[46] * (_exp2_213 if True else _exp2_212)))
                    acc_1[15] = cutlass.Float32((d_o[47] * (_exp2_213 if True else _exp2_212)))
                if (sset_0_1 == 3):
                    acc_1[0] = cutlass.Float32((d_o[48] * (_exp2_213 if False else _exp2_212)))
                    acc_1[1] = cutlass.Float32((d_o[49] * (_exp2_213 if False else _exp2_212)))
                    acc_1[2] = cutlass.Float32((d_o[50] * (_exp2_213 if True else _exp2_212)))
                    acc_1[3] = cutlass.Float32((d_o[51] * (_exp2_213 if True else _exp2_212)))
                    acc_1[4] = cutlass.Float32((d_o[52] * (_exp2_213 if False else _exp2_212)))
                    acc_1[5] = cutlass.Float32((d_o[53] * (_exp2_213 if False else _exp2_212)))
                    acc_1[6] = cutlass.Float32((d_o[54] * (_exp2_213 if True else _exp2_212)))
                    acc_1[7] = cutlass.Float32((d_o[55] * (_exp2_213 if True else _exp2_212)))
                    acc_1[8] = cutlass.Float32((d_o[56] * (_exp2_213 if False else _exp2_212)))
                    acc_1[9] = cutlass.Float32((d_o[57] * (_exp2_213 if False else _exp2_212)))
                    acc_1[10] = cutlass.Float32((d_o[58] * (_exp2_213 if True else _exp2_212)))
                    acc_1[11] = cutlass.Float32((d_o[59] * (_exp2_213 if True else _exp2_212)))
                    acc_1[12] = cutlass.Float32((d_o[60] * (_exp2_213 if False else _exp2_212)))
                    acc_1[13] = cutlass.Float32((d_o[61] * (_exp2_213 if False else _exp2_212)))
                    acc_1[14] = cutlass.Float32((d_o[62] * (_exp2_213 if True else _exp2_212)))
                    acc_1[15] = cutlass.Float32((d_o[63] * (_exp2_213 if True else _exp2_212)))
                base_3 = cutlass.Int32((2048 + my_tid4_1))
                _fma_118 = cute.math.fma(_recv_smem[base_3], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[0] = cutlass.Float32(_fma_118)
                _fma_119 = cute.math.fma(_recv_smem[(base_3 + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[1] = cutlass.Float32(_fma_119)
                _fma_120 = cute.math.fma(_recv_smem[(base_3 + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[2] = cutlass.Float32(_fma_120)
                _fma_121 = cute.math.fma(_recv_smem[(base_3 + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[3] = cutlass.Float32(_fma_121)
                _fma_122 = cute.math.fma(_recv_smem[(base_3 + 512)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[4] = cutlass.Float32(_fma_122)
                _fma_123 = cute.math.fma(_recv_smem[((base_3 + 512) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[5] = cutlass.Float32(_fma_123)
                _fma_124 = cute.math.fma(_recv_smem[((base_3 + 512) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[6] = cutlass.Float32(_fma_124)
                _fma_125 = cute.math.fma(_recv_smem[((base_3 + 512) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[7] = cutlass.Float32(_fma_125)
                _fma_126 = cute.math.fma(_recv_smem[(base_3 + 1024)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[8] = cutlass.Float32(_fma_126)
                _fma_127 = cute.math.fma(_recv_smem[((base_3 + 1024) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[9] = cutlass.Float32(_fma_127)
                _fma_128 = cute.math.fma(_recv_smem[((base_3 + 1024) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[10] = cutlass.Float32(_fma_128)
                _fma_129 = cute.math.fma(_recv_smem[((base_3 + 1024) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[11] = cutlass.Float32(_fma_129)
                _fma_130 = cute.math.fma(_recv_smem[(base_3 + 1536)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[12] = cutlass.Float32(_fma_130)
                _fma_131 = cute.math.fma(_recv_smem[((base_3 + 1536) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[13] = cutlass.Float32(_fma_131)
                _fma_132 = cute.math.fma(_recv_smem[((base_3 + 1536) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[14] = cutlass.Float32(_fma_132)
                _fma_133 = cute.math.fma(_recv_smem[((base_3 + 1536) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[15] = cutlass.Float32(_fma_133)
                _rcp_6 = cute.math.rcp(msum0_1[0], approx=True, ftz=True)
                _rcp_7 = cute.math.rcp(msum1_1[0], approx=True, ftz=True)
                acc_1[0] = cutlass.Float32((acc_1[0] * (_rcp_7 if False else _rcp_6)))
                acc_1[1] = cutlass.Float32((acc_1[1] * (_rcp_7 if False else _rcp_6)))
                acc_1[2] = cutlass.Float32((acc_1[2] * (_rcp_7 if True else _rcp_6)))
                acc_1[3] = cutlass.Float32((acc_1[3] * (_rcp_7 if True else _rcp_6)))
                acc_1[4] = cutlass.Float32((acc_1[4] * (_rcp_7 if False else _rcp_6)))
                acc_1[5] = cutlass.Float32((acc_1[5] * (_rcp_7 if False else _rcp_6)))
                acc_1[6] = cutlass.Float32((acc_1[6] * (_rcp_7 if True else _rcp_6)))
                acc_1[7] = cutlass.Float32((acc_1[7] * (_rcp_7 if True else _rcp_6)))
                acc_1[8] = cutlass.Float32((acc_1[8] * (_rcp_7 if False else _rcp_6)))
                acc_1[9] = cutlass.Float32((acc_1[9] * (_rcp_7 if False else _rcp_6)))
                acc_1[10] = cutlass.Float32((acc_1[10] * (_rcp_7 if True else _rcp_6)))
                acc_1[11] = cutlass.Float32((acc_1[11] * (_rcp_7 if True else _rcp_6)))
                acc_1[12] = cutlass.Float32((acc_1[12] * (_rcp_7 if False else _rcp_6)))
                acc_1[13] = cutlass.Float32((acc_1[13] * (_rcp_7 if False else _rcp_6)))
                acc_1[14] = cutlass.Float32((acc_1[14] * (_rcp_7 if True else _rcp_6)))
                acc_1[15] = cutlass.Float32((acc_1[15] * (_rcp_7 if True else _rcp_6)))
                qj1_3 = cutlass.Int32((qj & 1))
                qj2_3 = cutlass.Int32((qj & 2))
                o_vec_3 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp_3 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base_3 = cutlass.Int32((q_row * 128))
                m_local_r_3 = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_120 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[0] = cutlass.Uint32(_bf16x2_120)
                _bf16x2_121 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[1] = cutlass.Uint32(_bf16x2_121)
                _bf16x2_122 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[2] = cutlass.Uint32(_bf16x2_122)
                _bf16x2_123 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[3] = cutlass.Uint32(_bf16x2_123)
                _shfl_xor_76 = cute.arch.shuffle_sync_bfly(o_vec_3[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[0] = cutlass.Uint32(_shfl_xor_76)
                _shfl_xor_77 = cute.arch.shuffle_sync_bfly(o_vec_3[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[1] = cutlass.Uint32(_shfl_xor_77)
                _shfl_xor_78 = cute.arch.shuffle_sync_bfly(o_vec_3[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[2] = cutlass.Uint32(_shfl_xor_78)
                _shfl_xor_79 = cute.arch.shuffle_sync_bfly(o_vec_3[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[3] = cutlass.Uint32(_shfl_xor_79)
                o_vec_3[0] = cutlass.Uint32((o_tmp_3[0] if (qj1_3 != 0) else o_vec_3[0]))
                o_vec_3[1] = cutlass.Uint32((o_tmp_3[1] if (qj1_3 == 0) else o_vec_3[1]))
                o_vec_3[2] = cutlass.Uint32((o_tmp_3[2] if (qj1_3 != 0) else o_vec_3[2]))
                o_vec_3[3] = cutlass.Uint32((o_tmp_3[3] if (qj1_3 == 0) else o_vec_3[3]))
                _shfl_xor_80 = cute.arch.shuffle_sync_bfly(o_vec_3[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[0] = cutlass.Uint32(_shfl_xor_80)
                _shfl_xor_81 = cute.arch.shuffle_sync_bfly(o_vec_3[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[1] = cutlass.Uint32(_shfl_xor_81)
                _shfl_xor_82 = cute.arch.shuffle_sync_bfly(o_vec_3[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[2] = cutlass.Uint32(_shfl_xor_82)
                _shfl_xor_83 = cute.arch.shuffle_sync_bfly(o_vec_3[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[3] = cutlass.Uint32(_shfl_xor_83)
                o_vec_3[0] = cutlass.Uint32((o_tmp_3[0] if (qj2_3 != 0) else o_vec_3[0]))
                o_vec_3[1] = cutlass.Uint32((o_tmp_3[1] if (qj2_3 != 0) else o_vec_3[1]))
                o_vec_3[2] = cutlass.Uint32((o_tmp_3[2] if (qj2_3 == 0) else o_vec_3[2]))
                o_vec_3[3] = cutlass.Uint32((o_tmp_3[3] if (qj2_3 == 0) else o_vec_3[3]))
                o_off_4 = cutlass.Int32(((o_row_base_3 + (m_local_r_3 * 128)) + (((4 * sset_0_1) + qj) * 8)))
                _gmem_store_raw_100 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_3[0]), cutlass.Uint32(o_vec_3[1]), cutlass.Uint32(o_vec_3[2]), cutlass.Uint32(o_vec_3[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_100.ir_value(), O + o_off_4)
                m_local_r_0_3 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_124 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[0] = cutlass.Uint32(_bf16x2_124)
                _bf16x2_125 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[1] = cutlass.Uint32(_bf16x2_125)
                _bf16x2_126 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[2] = cutlass.Uint32(_bf16x2_126)
                _bf16x2_127 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_3[3] = cutlass.Uint32(_bf16x2_127)
                _shfl_xor_84 = cute.arch.shuffle_sync_bfly(o_vec_3[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[0] = cutlass.Uint32(_shfl_xor_84)
                _shfl_xor_85 = cute.arch.shuffle_sync_bfly(o_vec_3[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[1] = cutlass.Uint32(_shfl_xor_85)
                _shfl_xor_86 = cute.arch.shuffle_sync_bfly(o_vec_3[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[2] = cutlass.Uint32(_shfl_xor_86)
                _shfl_xor_87 = cute.arch.shuffle_sync_bfly(o_vec_3[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[3] = cutlass.Uint32(_shfl_xor_87)
                o_vec_3[0] = cutlass.Uint32((o_tmp_3[0] if (qj1_3 != 0) else o_vec_3[0]))
                o_vec_3[1] = cutlass.Uint32((o_tmp_3[1] if (qj1_3 == 0) else o_vec_3[1]))
                o_vec_3[2] = cutlass.Uint32((o_tmp_3[2] if (qj1_3 != 0) else o_vec_3[2]))
                o_vec_3[3] = cutlass.Uint32((o_tmp_3[3] if (qj1_3 == 0) else o_vec_3[3]))
                _shfl_xor_88 = cute.arch.shuffle_sync_bfly(o_vec_3[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[0] = cutlass.Uint32(_shfl_xor_88)
                _shfl_xor_89 = cute.arch.shuffle_sync_bfly(o_vec_3[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[1] = cutlass.Uint32(_shfl_xor_89)
                _shfl_xor_90 = cute.arch.shuffle_sync_bfly(o_vec_3[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[2] = cutlass.Uint32(_shfl_xor_90)
                _shfl_xor_91 = cute.arch.shuffle_sync_bfly(o_vec_3[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_3[3] = cutlass.Uint32(_shfl_xor_91)
                o_vec_3[0] = cutlass.Uint32((o_tmp_3[0] if (qj2_3 != 0) else o_vec_3[0]))
                o_vec_3[1] = cutlass.Uint32((o_tmp_3[1] if (qj2_3 != 0) else o_vec_3[1]))
                o_vec_3[2] = cutlass.Uint32((o_tmp_3[2] if (qj2_3 == 0) else o_vec_3[2]))
                o_vec_3[3] = cutlass.Uint32((o_tmp_3[3] if (qj2_3 == 0) else o_vec_3[3]))
                o_off_1_3 = cutlass.Int32(((o_row_base_3 + (m_local_r_0_3 * 128)) + (((4 * sset_0_1) + qj) * 8)))
                _gmem_store_raw_101 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_3[0]), cutlass.Uint32(o_vec_3[1]), cutlass.Uint32(o_vec_3[2]), cutlass.Uint32(o_vec_3[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_101.ir_value(), O + o_off_1_3)
    if _cake_ldparam_b64(_plan__base + 0, 'u64') != cutlass.Uint64(plan__slot_0):
        cutlass_llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string='trap;',
            constraints='~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
    if _cake_ldparam_b64(_plan__base + 3496, 'u64') != cutlass.Uint64(plan__slot_437):
        cutlass_llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string='trap;',
            constraints='~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )

@cute.jit
def launch_vsa_sm90_bf16_small_k6c2(Q: cute.Tensor, _cake_tma_Q_dim_0: cutlass.Int64, _cake_tma_Q_dim_1: cutlass.Int64, _cake_tma_Q_dim_2: cutlass.Int64, _cake_tma_Q_stride16_0: cutlass.Int64, _cake_tma_Q_stride16_1: cutlass.Int64, K: cute.Tensor, _cake_tma_K_dim_0: cutlass.Int64, _cake_tma_K_dim_1: cutlass.Int64, _cake_tma_K_dim_2: cutlass.Int64, _cake_tma_K_stride16_0: cutlass.Int64, _cake_tma_K_stride16_1: cutlass.Int64, Vt: cute.Tensor, _cake_tma_Vt_dim_0: cutlass.Int64, _cake_tma_Vt_dim_1: cutlass.Int64, _cake_tma_Vt_dim_2: cutlass.Int64, _cake_tma_Vt_dim_3: cutlass.Int64, _cake_tma_Vt_stride16_0: cutlass.Int64, _cake_tma_Vt_stride16_1: cutlass.Int64, _cake_tma_Vt_stride16_2: cutlass.Int64, O: cute.Tensor, plan__slot_0: cutlass.Uint64, plan__slot_1: cutlass.Uint64, plan__slot_2: cutlass.Uint64, plan__slot_3: cutlass.Uint64, plan__slot_4: cutlass.Uint64, plan__slot_5: cutlass.Uint64, plan__slot_6: cutlass.Uint64, plan__slot_7: cutlass.Uint64, plan__slot_8: cutlass.Uint64, plan__slot_9: cutlass.Uint64, plan__slot_10: cutlass.Uint64, plan__slot_11: cutlass.Uint64, plan__slot_12: cutlass.Uint64, plan__slot_13: cutlass.Uint64, plan__slot_14: cutlass.Uint64, plan__slot_15: cutlass.Uint64, plan__slot_16: cutlass.Uint64, plan__slot_17: cutlass.Uint64, plan__slot_18: cutlass.Uint64, plan__slot_19: cutlass.Uint64, plan__slot_20: cutlass.Uint64, plan__slot_21: cutlass.Uint64, plan__slot_22: cutlass.Uint64, plan__slot_23: cutlass.Uint64, plan__slot_24: cutlass.Uint64, plan__slot_25: cutlass.Uint64, plan__slot_26: cutlass.Uint64, plan__slot_27: cutlass.Uint64, plan__slot_28: cutlass.Uint64, plan__slot_29: cutlass.Uint64, plan__slot_30: cutlass.Uint64, plan__slot_31: cutlass.Uint64, plan__slot_32: cutlass.Uint64, plan__slot_33: cutlass.Uint64, plan__slot_34: cutlass.Uint64, plan__slot_35: cutlass.Uint64, plan__slot_36: cutlass.Uint64, plan__slot_37: cutlass.Uint64, plan__slot_38: cutlass.Uint64, plan__slot_39: cutlass.Uint64, plan__slot_40: cutlass.Uint64, plan__slot_41: cutlass.Uint64, plan__slot_42: cutlass.Uint64, plan__slot_43: cutlass.Uint64, plan__slot_44: cutlass.Uint64, plan__slot_45: cutlass.Uint64, plan__slot_46: cutlass.Uint64, plan__slot_47: cutlass.Uint64, plan__slot_48: cutlass.Uint64, plan__slot_49: cutlass.Uint64, plan__slot_50: cutlass.Uint64, plan__slot_51: cutlass.Uint64, plan__slot_52: cutlass.Uint64, plan__slot_53: cutlass.Uint64, plan__slot_54: cutlass.Uint64, plan__slot_55: cutlass.Uint64, plan__slot_56: cutlass.Uint64, plan__slot_57: cutlass.Uint64, plan__slot_58: cutlass.Uint64, plan__slot_59: cutlass.Uint64, plan__slot_60: cutlass.Uint64, plan__slot_61: cutlass.Uint64, plan__slot_62: cutlass.Uint64, plan__slot_63: cutlass.Uint64, plan__slot_64: cutlass.Uint64, plan__slot_65: cutlass.Uint64, plan__slot_66: cutlass.Uint64, plan__slot_67: cutlass.Uint64, plan__slot_68: cutlass.Uint64, plan__slot_69: cutlass.Uint64, plan__slot_70: cutlass.Uint64, plan__slot_71: cutlass.Uint64, plan__slot_72: cutlass.Uint64, plan__slot_73: cutlass.Uint64, plan__slot_74: cutlass.Uint64, plan__slot_75: cutlass.Uint64, plan__slot_76: cutlass.Uint64, plan__slot_77: cutlass.Uint64, plan__slot_78: cutlass.Uint64, plan__slot_79: cutlass.Uint64, plan__slot_80: cutlass.Uint64, plan__slot_81: cutlass.Uint64, plan__slot_82: cutlass.Uint64, plan__slot_83: cutlass.Uint64, plan__slot_84: cutlass.Uint64, plan__slot_85: cutlass.Uint64, plan__slot_86: cutlass.Uint64, plan__slot_87: cutlass.Uint64, plan__slot_88: cutlass.Uint64, plan__slot_89: cutlass.Uint64, plan__slot_90: cutlass.Uint64, plan__slot_91: cutlass.Uint64, plan__slot_92: cutlass.Uint64, plan__slot_93: cutlass.Uint64, plan__slot_94: cutlass.Uint64, plan__slot_95: cutlass.Uint64, plan__slot_96: cutlass.Uint64, plan__slot_97: cutlass.Uint64, plan__slot_98: cutlass.Uint64, plan__slot_99: cutlass.Uint64, plan__slot_100: cutlass.Uint64, plan__slot_101: cutlass.Uint64, plan__slot_102: cutlass.Uint64, plan__slot_103: cutlass.Uint64, plan__slot_104: cutlass.Uint64, plan__slot_105: cutlass.Uint64, plan__slot_106: cutlass.Uint64, plan__slot_107: cutlass.Uint64, plan__slot_108: cutlass.Uint64, plan__slot_109: cutlass.Uint64, plan__slot_110: cutlass.Uint64, plan__slot_111: cutlass.Uint64, plan__slot_112: cutlass.Uint64, plan__slot_113: cutlass.Uint64, plan__slot_114: cutlass.Uint64, plan__slot_115: cutlass.Uint64, plan__slot_116: cutlass.Uint64, plan__slot_117: cutlass.Uint64, plan__slot_118: cutlass.Uint64, plan__slot_119: cutlass.Uint64, plan__slot_120: cutlass.Uint64, plan__slot_121: cutlass.Uint64, plan__slot_122: cutlass.Uint64, plan__slot_123: cutlass.Uint64, plan__slot_124: cutlass.Uint64, plan__slot_125: cutlass.Uint64, plan__slot_126: cutlass.Uint64, plan__slot_127: cutlass.Uint64, plan__slot_128: cutlass.Uint64, plan__slot_129: cutlass.Uint64, plan__slot_130: cutlass.Uint64, plan__slot_131: cutlass.Uint64, plan__slot_132: cutlass.Uint64, plan__slot_133: cutlass.Uint64, plan__slot_134: cutlass.Uint64, plan__slot_135: cutlass.Uint64, plan__slot_136: cutlass.Uint64, plan__slot_137: cutlass.Uint64, plan__slot_138: cutlass.Uint64, plan__slot_139: cutlass.Uint64, plan__slot_140: cutlass.Uint64, plan__slot_141: cutlass.Uint64, plan__slot_142: cutlass.Uint64, plan__slot_143: cutlass.Uint64, plan__slot_144: cutlass.Uint64, plan__slot_145: cutlass.Uint64, plan__slot_146: cutlass.Uint64, plan__slot_147: cutlass.Uint64, plan__slot_148: cutlass.Uint64, plan__slot_149: cutlass.Uint64, plan__slot_150: cutlass.Uint64, plan__slot_151: cutlass.Uint64, plan__slot_152: cutlass.Uint64, plan__slot_153: cutlass.Uint64, plan__slot_154: cutlass.Uint64, plan__slot_155: cutlass.Uint64, plan__slot_156: cutlass.Uint64, plan__slot_157: cutlass.Uint64, plan__slot_158: cutlass.Uint64, plan__slot_159: cutlass.Uint64, plan__slot_160: cutlass.Uint64, plan__slot_161: cutlass.Uint64, plan__slot_162: cutlass.Uint64, plan__slot_163: cutlass.Uint64, plan__slot_164: cutlass.Uint64, plan__slot_165: cutlass.Uint64, plan__slot_166: cutlass.Uint64, plan__slot_167: cutlass.Uint64, plan__slot_168: cutlass.Uint64, plan__slot_169: cutlass.Uint64, plan__slot_170: cutlass.Uint64, plan__slot_171: cutlass.Uint64, plan__slot_172: cutlass.Uint64, plan__slot_173: cutlass.Uint64, plan__slot_174: cutlass.Uint64, plan__slot_175: cutlass.Uint64, plan__slot_176: cutlass.Uint64, plan__slot_177: cutlass.Uint64, plan__slot_178: cutlass.Uint64, plan__slot_179: cutlass.Uint64, plan__slot_180: cutlass.Uint64, plan__slot_181: cutlass.Uint64, plan__slot_182: cutlass.Uint64, plan__slot_183: cutlass.Uint64, plan__slot_184: cutlass.Uint64, plan__slot_185: cutlass.Uint64, plan__slot_186: cutlass.Uint64, plan__slot_187: cutlass.Uint64, plan__slot_188: cutlass.Uint64, plan__slot_189: cutlass.Uint64, plan__slot_190: cutlass.Uint64, plan__slot_191: cutlass.Uint64, plan__slot_192: cutlass.Uint64, plan__slot_193: cutlass.Uint64, plan__slot_194: cutlass.Uint64, plan__slot_195: cutlass.Uint64, plan__slot_196: cutlass.Uint64, plan__slot_197: cutlass.Uint64, plan__slot_198: cutlass.Uint64, plan__slot_199: cutlass.Uint64, plan__slot_200: cutlass.Uint64, plan__slot_201: cutlass.Uint64, plan__slot_202: cutlass.Uint64, plan__slot_203: cutlass.Uint64, plan__slot_204: cutlass.Uint64, plan__slot_205: cutlass.Uint64, plan__slot_206: cutlass.Uint64, plan__slot_207: cutlass.Uint64, plan__slot_208: cutlass.Uint64, plan__slot_209: cutlass.Uint64, plan__slot_210: cutlass.Uint64, plan__slot_211: cutlass.Uint64, plan__slot_212: cutlass.Uint64, plan__slot_213: cutlass.Uint64, plan__slot_214: cutlass.Uint64, plan__slot_215: cutlass.Uint64, plan__slot_216: cutlass.Uint64, plan__slot_217: cutlass.Uint64, plan__slot_218: cutlass.Uint64, plan__slot_219: cutlass.Uint64, plan__slot_220: cutlass.Uint64, plan__slot_221: cutlass.Uint64, plan__slot_222: cutlass.Uint64, plan__slot_223: cutlass.Uint64, plan__slot_224: cutlass.Uint64, plan__slot_225: cutlass.Uint64, plan__slot_226: cutlass.Uint64, plan__slot_227: cutlass.Uint64, plan__slot_228: cutlass.Uint64, plan__slot_229: cutlass.Uint64, plan__slot_230: cutlass.Uint64, plan__slot_231: cutlass.Uint64, plan__slot_232: cutlass.Uint64, plan__slot_233: cutlass.Uint64, plan__slot_234: cutlass.Uint64, plan__slot_235: cutlass.Uint64, plan__slot_236: cutlass.Uint64, plan__slot_237: cutlass.Uint64, plan__slot_238: cutlass.Uint64, plan__slot_239: cutlass.Uint64, plan__slot_240: cutlass.Uint64, plan__slot_241: cutlass.Uint64, plan__slot_242: cutlass.Uint64, plan__slot_243: cutlass.Uint64, plan__slot_244: cutlass.Uint64, plan__slot_245: cutlass.Uint64, plan__slot_246: cutlass.Uint64, plan__slot_247: cutlass.Uint64, plan__slot_248: cutlass.Uint64, plan__slot_249: cutlass.Uint64, plan__slot_250: cutlass.Uint64, plan__slot_251: cutlass.Uint64, plan__slot_252: cutlass.Uint64, plan__slot_253: cutlass.Uint64, plan__slot_254: cutlass.Uint64, plan__slot_255: cutlass.Uint64, plan__slot_256: cutlass.Uint64, plan__slot_257: cutlass.Uint64, plan__slot_258: cutlass.Uint64, plan__slot_259: cutlass.Uint64, plan__slot_260: cutlass.Uint64, plan__slot_261: cutlass.Uint64, plan__slot_262: cutlass.Uint64, plan__slot_263: cutlass.Uint64, plan__slot_264: cutlass.Uint64, plan__slot_265: cutlass.Uint64, plan__slot_266: cutlass.Uint64, plan__slot_267: cutlass.Uint64, plan__slot_268: cutlass.Uint64, plan__slot_269: cutlass.Uint64, plan__slot_270: cutlass.Uint64, plan__slot_271: cutlass.Uint64, plan__slot_272: cutlass.Uint64, plan__slot_273: cutlass.Uint64, plan__slot_274: cutlass.Uint64, plan__slot_275: cutlass.Uint64, plan__slot_276: cutlass.Uint64, plan__slot_277: cutlass.Uint64, plan__slot_278: cutlass.Uint64, plan__slot_279: cutlass.Uint64, plan__slot_280: cutlass.Uint64, plan__slot_281: cutlass.Uint64, plan__slot_282: cutlass.Uint64, plan__slot_283: cutlass.Uint64, plan__slot_284: cutlass.Uint64, plan__slot_285: cutlass.Uint64, plan__slot_286: cutlass.Uint64, plan__slot_287: cutlass.Uint64, plan__slot_288: cutlass.Uint64, plan__slot_289: cutlass.Uint64, plan__slot_290: cutlass.Uint64, plan__slot_291: cutlass.Uint64, plan__slot_292: cutlass.Uint64, plan__slot_293: cutlass.Uint64, plan__slot_294: cutlass.Uint64, plan__slot_295: cutlass.Uint64, plan__slot_296: cutlass.Uint64, plan__slot_297: cutlass.Uint64, plan__slot_298: cutlass.Uint64, plan__slot_299: cutlass.Uint64, plan__slot_300: cutlass.Uint64, plan__slot_301: cutlass.Uint64, plan__slot_302: cutlass.Uint64, plan__slot_303: cutlass.Uint64, plan__slot_304: cutlass.Uint64, plan__slot_305: cutlass.Uint64, plan__slot_306: cutlass.Uint64, plan__slot_307: cutlass.Uint64, plan__slot_308: cutlass.Uint64, plan__slot_309: cutlass.Uint64, plan__slot_310: cutlass.Uint64, plan__slot_311: cutlass.Uint64, plan__slot_312: cutlass.Uint64, plan__slot_313: cutlass.Uint64, plan__slot_314: cutlass.Uint64, plan__slot_315: cutlass.Uint64, plan__slot_316: cutlass.Uint64, plan__slot_317: cutlass.Uint64, plan__slot_318: cutlass.Uint64, plan__slot_319: cutlass.Uint64, plan__slot_320: cutlass.Uint64, plan__slot_321: cutlass.Uint64, plan__slot_322: cutlass.Uint64, plan__slot_323: cutlass.Uint64, plan__slot_324: cutlass.Uint64, plan__slot_325: cutlass.Uint64, plan__slot_326: cutlass.Uint64, plan__slot_327: cutlass.Uint64, plan__slot_328: cutlass.Uint64, plan__slot_329: cutlass.Uint64, plan__slot_330: cutlass.Uint64, plan__slot_331: cutlass.Uint64, plan__slot_332: cutlass.Uint64, plan__slot_333: cutlass.Uint64, plan__slot_334: cutlass.Uint64, plan__slot_335: cutlass.Uint64, plan__slot_336: cutlass.Uint64, plan__slot_337: cutlass.Uint64, plan__slot_338: cutlass.Uint64, plan__slot_339: cutlass.Uint64, plan__slot_340: cutlass.Uint64, plan__slot_341: cutlass.Uint64, plan__slot_342: cutlass.Uint64, plan__slot_343: cutlass.Uint64, plan__slot_344: cutlass.Uint64, plan__slot_345: cutlass.Uint64, plan__slot_346: cutlass.Uint64, plan__slot_347: cutlass.Uint64, plan__slot_348: cutlass.Uint64, plan__slot_349: cutlass.Uint64, plan__slot_350: cutlass.Uint64, plan__slot_351: cutlass.Uint64, plan__slot_352: cutlass.Uint64, plan__slot_353: cutlass.Uint64, plan__slot_354: cutlass.Uint64, plan__slot_355: cutlass.Uint64, plan__slot_356: cutlass.Uint64, plan__slot_357: cutlass.Uint64, plan__slot_358: cutlass.Uint64, plan__slot_359: cutlass.Uint64, plan__slot_360: cutlass.Uint64, plan__slot_361: cutlass.Uint64, plan__slot_362: cutlass.Uint64, plan__slot_363: cutlass.Uint64, plan__slot_364: cutlass.Uint64, plan__slot_365: cutlass.Uint64, plan__slot_366: cutlass.Uint64, plan__slot_367: cutlass.Uint64, plan__slot_368: cutlass.Uint64, plan__slot_369: cutlass.Uint64, plan__slot_370: cutlass.Uint64, plan__slot_371: cutlass.Uint64, plan__slot_372: cutlass.Uint64, plan__slot_373: cutlass.Uint64, plan__slot_374: cutlass.Uint64, plan__slot_375: cutlass.Uint64, plan__slot_376: cutlass.Uint64, plan__slot_377: cutlass.Uint64, plan__slot_378: cutlass.Uint64, plan__slot_379: cutlass.Uint64, plan__slot_380: cutlass.Uint64, plan__slot_381: cutlass.Uint64, plan__slot_382: cutlass.Uint64, plan__slot_383: cutlass.Uint64, plan__slot_384: cutlass.Uint64, plan__slot_385: cutlass.Uint64, plan__slot_386: cutlass.Uint64, plan__slot_387: cutlass.Uint64, plan__slot_388: cutlass.Uint64, plan__slot_389: cutlass.Uint64, plan__slot_390: cutlass.Uint64, plan__slot_391: cutlass.Uint64, plan__slot_392: cutlass.Uint64, plan__slot_393: cutlass.Uint64, plan__slot_394: cutlass.Uint64, plan__slot_395: cutlass.Uint64, plan__slot_396: cutlass.Uint64, plan__slot_397: cutlass.Uint64, plan__slot_398: cutlass.Uint64, plan__slot_399: cutlass.Uint64, plan__slot_400: cutlass.Uint64, plan__slot_401: cutlass.Uint64, plan__slot_402: cutlass.Uint64, plan__slot_403: cutlass.Uint64, plan__slot_404: cutlass.Uint64, plan__slot_405: cutlass.Uint64, plan__slot_406: cutlass.Uint64, plan__slot_407: cutlass.Uint64, plan__slot_408: cutlass.Uint64, plan__slot_409: cutlass.Uint64, plan__slot_410: cutlass.Uint64, plan__slot_411: cutlass.Uint64, plan__slot_412: cutlass.Uint64, plan__slot_413: cutlass.Uint64, plan__slot_414: cutlass.Uint64, plan__slot_415: cutlass.Uint64, plan__slot_416: cutlass.Uint64, plan__slot_417: cutlass.Uint64, plan__slot_418: cutlass.Uint64, plan__slot_419: cutlass.Uint64, plan__slot_420: cutlass.Uint64, plan__slot_421: cutlass.Uint64, plan__slot_422: cutlass.Uint64, plan__slot_423: cutlass.Uint64, plan__slot_424: cutlass.Uint64, plan__slot_425: cutlass.Uint64, plan__slot_426: cutlass.Uint64, plan__slot_427: cutlass.Uint64, plan__slot_428: cutlass.Uint64, plan__slot_429: cutlass.Uint64, plan__slot_430: cutlass.Uint64, plan__slot_431: cutlass.Uint64, plan__slot_432: cutlass.Uint64, plan__slot_433: cutlass.Uint64, plan__slot_434: cutlass.Uint64, plan__slot_435: cutlass.Uint64, plan__slot_436: cutlass.Uint64, plan__slot_437: cutlass.Uint64, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32, grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32, stream: cuda.CUstream):
    _tma_Q = create_tensor_map_tiled(
        Q.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_Q_dim_0, _cake_tma_Q_dim_1, _cake_tma_Q_dim_2],
        [_cake_tma_Q_stride16_0, _cake_tma_Q_stride16_1],
        [64, 64, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_K = create_tensor_map_tiled(
        K.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_K_dim_0, _cake_tma_K_dim_1, _cake_tma_K_dim_2],
        [_cake_tma_K_stride16_0, _cake_tma_K_stride16_1],
        [64, 64, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_Vt = create_tensor_map_tiled(
        Vt.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_Vt_dim_0, _cake_tma_Vt_dim_1, _cake_tma_Vt_dim_2, _cake_tma_Vt_dim_3],
        [_cake_tma_Vt_stride16_0, _cake_tma_Vt_stride16_1, _cake_tma_Vt_stride16_2],
        [64, 8, 8, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _cake_launch_cluster_spread(kernel_vsa_sm90_bf16_small_k6c2(_tma_Q, _tma_K, _tma_Vt, plan__slot_0, plan__slot_1, plan__slot_2, plan__slot_3, plan__slot_4, plan__slot_5, plan__slot_6, plan__slot_7, plan__slot_8, plan__slot_9, plan__slot_10, plan__slot_11, plan__slot_12, plan__slot_13, plan__slot_14, plan__slot_15, plan__slot_16, plan__slot_17, plan__slot_18, plan__slot_19, plan__slot_20, plan__slot_21, plan__slot_22, plan__slot_23, plan__slot_24, plan__slot_25, plan__slot_26, plan__slot_27, plan__slot_28, plan__slot_29, plan__slot_30, plan__slot_31, plan__slot_32, plan__slot_33, plan__slot_34, plan__slot_35, plan__slot_36, plan__slot_37, plan__slot_38, plan__slot_39, plan__slot_40, plan__slot_41, plan__slot_42, plan__slot_43, plan__slot_44, plan__slot_45, plan__slot_46, plan__slot_47, plan__slot_48, plan__slot_49, plan__slot_50, plan__slot_51, plan__slot_52, plan__slot_53, plan__slot_54, plan__slot_55, plan__slot_56, plan__slot_57, plan__slot_58, plan__slot_59, plan__slot_60, plan__slot_61, plan__slot_62, plan__slot_63, plan__slot_64, plan__slot_65, plan__slot_66, plan__slot_67, plan__slot_68, plan__slot_69, plan__slot_70, plan__slot_71, plan__slot_72, plan__slot_73, plan__slot_74, plan__slot_75, plan__slot_76, plan__slot_77, plan__slot_78, plan__slot_79, plan__slot_80, plan__slot_81, plan__slot_82, plan__slot_83, plan__slot_84, plan__slot_85, plan__slot_86, plan__slot_87, plan__slot_88, plan__slot_89, plan__slot_90, plan__slot_91, plan__slot_92, plan__slot_93, plan__slot_94, plan__slot_95, plan__slot_96, plan__slot_97, plan__slot_98, plan__slot_99, plan__slot_100, plan__slot_101, plan__slot_102, plan__slot_103, plan__slot_104, plan__slot_105, plan__slot_106, plan__slot_107, plan__slot_108, plan__slot_109, plan__slot_110, plan__slot_111, plan__slot_112, plan__slot_113, plan__slot_114, plan__slot_115, plan__slot_116, plan__slot_117, plan__slot_118, plan__slot_119, plan__slot_120, plan__slot_121, plan__slot_122, plan__slot_123, plan__slot_124, plan__slot_125, plan__slot_126, plan__slot_127, plan__slot_128, plan__slot_129, plan__slot_130, plan__slot_131, plan__slot_132, plan__slot_133, plan__slot_134, plan__slot_135, plan__slot_136, plan__slot_137, plan__slot_138, plan__slot_139, plan__slot_140, plan__slot_141, plan__slot_142, plan__slot_143, plan__slot_144, plan__slot_145, plan__slot_146, plan__slot_147, plan__slot_148, plan__slot_149, plan__slot_150, plan__slot_151, plan__slot_152, plan__slot_153, plan__slot_154, plan__slot_155, plan__slot_156, plan__slot_157, plan__slot_158, plan__slot_159, plan__slot_160, plan__slot_161, plan__slot_162, plan__slot_163, plan__slot_164, plan__slot_165, plan__slot_166, plan__slot_167, plan__slot_168, plan__slot_169, plan__slot_170, plan__slot_171, plan__slot_172, plan__slot_173, plan__slot_174, plan__slot_175, plan__slot_176, plan__slot_177, plan__slot_178, plan__slot_179, plan__slot_180, plan__slot_181, plan__slot_182, plan__slot_183, plan__slot_184, plan__slot_185, plan__slot_186, plan__slot_187, plan__slot_188, plan__slot_189, plan__slot_190, plan__slot_191, plan__slot_192, plan__slot_193, plan__slot_194, plan__slot_195, plan__slot_196, plan__slot_197, plan__slot_198, plan__slot_199, plan__slot_200, plan__slot_201, plan__slot_202, plan__slot_203, plan__slot_204, plan__slot_205, plan__slot_206, plan__slot_207, plan__slot_208, plan__slot_209, plan__slot_210, plan__slot_211, plan__slot_212, plan__slot_213, plan__slot_214, plan__slot_215, plan__slot_216, plan__slot_217, plan__slot_218, plan__slot_219, plan__slot_220, plan__slot_221, plan__slot_222, plan__slot_223, plan__slot_224, plan__slot_225, plan__slot_226, plan__slot_227, plan__slot_228, plan__slot_229, plan__slot_230, plan__slot_231, plan__slot_232, plan__slot_233, plan__slot_234, plan__slot_235, plan__slot_236, plan__slot_237, plan__slot_238, plan__slot_239, plan__slot_240, plan__slot_241, plan__slot_242, plan__slot_243, plan__slot_244, plan__slot_245, plan__slot_246, plan__slot_247, plan__slot_248, plan__slot_249, plan__slot_250, plan__slot_251, plan__slot_252, plan__slot_253, plan__slot_254, plan__slot_255, plan__slot_256, plan__slot_257, plan__slot_258, plan__slot_259, plan__slot_260, plan__slot_261, plan__slot_262, plan__slot_263, plan__slot_264, plan__slot_265, plan__slot_266, plan__slot_267, plan__slot_268, plan__slot_269, plan__slot_270, plan__slot_271, plan__slot_272, plan__slot_273, plan__slot_274, plan__slot_275, plan__slot_276, plan__slot_277, plan__slot_278, plan__slot_279, plan__slot_280, plan__slot_281, plan__slot_282, plan__slot_283, plan__slot_284, plan__slot_285, plan__slot_286, plan__slot_287, plan__slot_288, plan__slot_289, plan__slot_290, plan__slot_291, plan__slot_292, plan__slot_293, plan__slot_294, plan__slot_295, plan__slot_296, plan__slot_297, plan__slot_298, plan__slot_299, plan__slot_300, plan__slot_301, plan__slot_302, plan__slot_303, plan__slot_304, plan__slot_305, plan__slot_306, plan__slot_307, plan__slot_308, plan__slot_309, plan__slot_310, plan__slot_311, plan__slot_312, plan__slot_313, plan__slot_314, plan__slot_315, plan__slot_316, plan__slot_317, plan__slot_318, plan__slot_319, plan__slot_320, plan__slot_321, plan__slot_322, plan__slot_323, plan__slot_324, plan__slot_325, plan__slot_326, plan__slot_327, plan__slot_328, plan__slot_329, plan__slot_330, plan__slot_331, plan__slot_332, plan__slot_333, plan__slot_334, plan__slot_335, plan__slot_336, plan__slot_337, plan__slot_338, plan__slot_339, plan__slot_340, plan__slot_341, plan__slot_342, plan__slot_343, plan__slot_344, plan__slot_345, plan__slot_346, plan__slot_347, plan__slot_348, plan__slot_349, plan__slot_350, plan__slot_351, plan__slot_352, plan__slot_353, plan__slot_354, plan__slot_355, plan__slot_356, plan__slot_357, plan__slot_358, plan__slot_359, plan__slot_360, plan__slot_361, plan__slot_362, plan__slot_363, plan__slot_364, plan__slot_365, plan__slot_366, plan__slot_367, plan__slot_368, plan__slot_369, plan__slot_370, plan__slot_371, plan__slot_372, plan__slot_373, plan__slot_374, plan__slot_375, plan__slot_376, plan__slot_377, plan__slot_378, plan__slot_379, plan__slot_380, plan__slot_381, plan__slot_382, plan__slot_383, plan__slot_384, plan__slot_385, plan__slot_386, plan__slot_387, plan__slot_388, plan__slot_389, plan__slot_390, plan__slot_391, plan__slot_392, plan__slot_393, plan__slot_394, plan__slot_395, plan__slot_396, plan__slot_397, plan__slot_398, plan__slot_399, plan__slot_400, plan__slot_401, plan__slot_402, plan__slot_403, plan__slot_404, plan__slot_405, plan__slot_406, plan__slot_407, plan__slot_408, plan__slot_409, plan__slot_410, plan__slot_411, plan__slot_412, plan__slot_413, plan__slot_414, plan__slot_415, plan__slot_416, plan__slot_417, plan__slot_418, plan__slot_419, plan__slot_420, plan__slot_421, plan__slot_422, plan__slot_423, plan__slot_424, plan__slot_425, plan__slot_426, plan__slot_427, plan__slot_428, plan__slot_429, plan__slot_430, plan__slot_431, plan__slot_432, plan__slot_433, plan__slot_434, plan__slot_435, plan__slot_436, plan__slot_437, O.iterator, seqlen_q, seqlen_k, scale_log2),
        grid=(grid_x, grid_y, grid_z),
        block=(256, 1, 1),
        cluster=(2, 1, 1),
        smem=232448,
        min_blocks_per_mp=1,
        stream=stream,
    )

def compile_program():
    return cute.compile(launch_vsa_sm90_bf16_small_k6c2,
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='Q'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='K'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='Vt'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='O'),), (1,), assumed_align=16),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Float32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_90a',
    )
