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

NUM_K_PIPE_STAGES = 3
NUM_V_PIPE_STAGES = 3
NUM_Q_PIPE_STAGES = 2
NUM_META_PIPE_STAGES = 2
SMEM_Q_SMEM_OFF = 1024
SMEM_Q_SMEM_STAGE_BYTES = 16384
SMEM_Q_SMEM_STRIDE = 16384
SMEM_K_SMEM_OFF = 33792
SMEM_K_SMEM_STAGE_BYTES = 32768
SMEM_K_SMEM_STRIDE = 32768
SMEM_VT_SMEM_A_OFF = 132096
SMEM_VT_SMEM_A_STAGE_BYTES = 16384
SMEM_VT_SMEM_A_STRIDE = 32768
SMEM_VT_SMEM_B_OFF = 148480
SMEM_VT_SMEM_B_STAGE_BYTES = 16384
SMEM_VT_SMEM_B_STRIDE = 32768
SMEM_META_SMEM_OFF = 230400
SMEM_META_SMEM_STAGE_BYTES = 1152
SMEM_META_SMEM_STRIDE = 1152
SMEM_MERGE_O_OFF = 17408
SMEM_MERGE_O_STAGE_BYTES = 16384
SMEM_MERGE_O_STRIDE = 16384
SMEM_MERGE_ML_OFF = 231552
SMEM_MERGE_ML_STAGE_BYTES = 512
SMEM_MERGE_ML_STRIDE = 512
SMEM_SMEM_V7_OFF = 148480
SMEM_SMEM_V7_STAGE_BYTES = 16384
SMEM_SMEM_V7_STRIDE = 16384
SMEM_SMEM_V8_OFF = 181248
SMEM_SMEM_V8_STAGE_BYTES = 16384
SMEM_SMEM_V8_STRIDE = 16384
SMEM_SMEM_V9_OFF = 214016
SMEM_SMEM_V9_STAGE_BYTES = 16384
SMEM_SMEM_V9_STRIDE = 16384
SMEM_TOTAL = 232064
THREADS = 384
CAKE_TARGET_ARCH = 'sm_90a'
CAKE_SMEM_BYTES = 232064

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

@cute.kernel
def kernel_vsa_sm90_bf16_fwd(Q: cutlass.GridConstant[TensorMap], K: cutlass.GridConstant[TensorMap], Vt: cutlass.GridConstant[TensorMap], hdr__slot_0: cutlass.Uint64, hdr__slot_1: cutlass.Uint64, hdr__slot_2: cutlass.Uint64, hdr__slot_3: cutlass.Uint64, hdr__slot_4: cutlass.Uint64, hdr__slot_5: cutlass.Uint64, hdr__slot_6: cutlass.Uint64, hdr__slot_7: cutlass.Uint64, hdr__slot_8: cutlass.Uint64, hdr__slot_9: cutlass.Uint64, hdr__slot_10: cutlass.Uint64, hdr__slot_11: cutlass.Uint64, hdr__slot_12: cutlass.Uint64, hdr__slot_13: cutlass.Uint64, hdr__slot_14: cutlass.Uint64, hdr__slot_15: cutlass.Uint64, hdr__slot_16: cutlass.Uint64, hdr__slot_17: cutlass.Uint64, hdr__slot_18: cutlass.Uint64, hdr__slot_19: cutlass.Uint64, hdr__slot_20: cutlass.Uint64, hdr__slot_21: cutlass.Uint64, hdr__slot_22: cutlass.Uint64, hdr__slot_23: cutlass.Uint64, hdr__slot_24: cutlass.Uint64, hdr__slot_25: cutlass.Uint64, hdr__slot_26: cutlass.Uint64, hdr__slot_27: cutlass.Uint64, hdr__slot_28: cutlass.Uint64, hdr__slot_29: cutlass.Uint64, hdr__slot_30: cutlass.Uint64, hdr__slot_31: cutlass.Uint64, hdr__slot_32: cutlass.Uint64, hdr__slot_33: cutlass.Uint64, hdr__slot_34: cutlass.Uint64, hdr__slot_35: cutlass.Uint64, hdr__slot_36: cutlass.Uint64, hdr__slot_37: cutlass.Uint64, hdr__slot_38: cutlass.Uint64, hdr__slot_39: cutlass.Uint64, hdr__slot_40: cutlass.Uint64, hdr__slot_41: cutlass.Uint64, hdr__slot_42: cutlass.Uint64, hdr__slot_43: cutlass.Uint64, hdr__slot_44: cutlass.Uint64, hdr__slot_45: cutlass.Uint64, hdr__slot_46: cutlass.Uint64, hdr__slot_47: cutlass.Uint64, hdr__slot_48: cutlass.Uint64, hdr__slot_49: cutlass.Uint64, hdr__slot_50: cutlass.Uint64, hdr__slot_51: cutlass.Uint64, hdr__slot_52: cutlass.Uint64, hdr__slot_53: cutlass.Uint64, hdr__slot_54: cutlass.Uint64, hdr__slot_55: cutlass.Uint64, hdr__slot_56: cutlass.Uint64, hdr__slot_57: cutlass.Uint64, hdr__slot_58: cutlass.Uint64, hdr__slot_59: cutlass.Uint64, hdr__slot_60: cutlass.Uint64, hdr__slot_61: cutlass.Uint64, hdr__slot_62: cutlass.Uint64, hdr__slot_63: cutlass.Uint64, hdr__slot_64: cutlass.Uint64, hdr__slot_65: cutlass.Uint64, hdr__slot_66: cutlass.Uint64, hdr__slot_67: cutlass.Uint64, hdr__slot_68: cutlass.Uint64, hdr__slot_69: cutlass.Uint64, hdr__slot_70: cutlass.Uint64, hdr__slot_71: cutlass.Uint64, hdr__slot_72: cutlass.Uint64, hdr__slot_73: cutlass.Uint64, hdr__slot_74: cutlass.Uint64, hdr__slot_75: cutlass.Uint64, hdr__slot_76: cutlass.Uint64, hdr__slot_77: cutlass.Uint64, hdr__slot_78: cutlass.Uint64, hdr__slot_79: cutlass.Uint64, hdr__slot_80: cutlass.Uint64, hdr__slot_81: cutlass.Uint64, hdr__slot_82: cutlass.Uint64, hdr__slot_83: cutlass.Uint64, hdr__slot_84: cutlass.Uint64, hdr__slot_85: cutlass.Uint64, hdr__slot_86: cutlass.Uint64, hdr__slot_87: cutlass.Uint64, hdr__slot_88: cutlass.Uint64, hdr__slot_89: cutlass.Uint64, hdr__slot_90: cutlass.Uint64, hdr__slot_91: cutlass.Uint64, hdr__slot_92: cutlass.Uint64, hdr__slot_93: cutlass.Uint64, hdr__slot_94: cutlass.Uint64, hdr__slot_95: cutlass.Uint64, hdr__slot_96: cutlass.Uint64, hdr__slot_97: cutlass.Uint64, hdr__slot_98: cutlass.Uint64, hdr__slot_99: cutlass.Uint64, hdr__slot_100: cutlass.Uint64, hdr__slot_101: cutlass.Uint64, hdr__slot_102: cutlass.Uint64, hdr__slot_103: cutlass.Uint64, hdr__slot_104: cutlass.Uint64, hdr__slot_105: cutlass.Uint64, hdr__slot_106: cutlass.Uint64, hdr__slot_107: cutlass.Uint64, hdr__slot_108: cutlass.Uint64, hdr__slot_109: cutlass.Uint64, hdr__slot_110: cutlass.Uint64, hdr__slot_111: cutlass.Uint64, hdr__slot_112: cutlass.Uint64, hdr__slot_113: cutlass.Uint64, hdr__slot_114: cutlass.Uint64, hdr__slot_115: cutlass.Uint64, hdr__slot_116: cutlass.Uint64, hdr__slot_117: cutlass.Uint64, hdr__slot_118: cutlass.Uint64, hdr__slot_119: cutlass.Uint64, hdr__slot_120: cutlass.Uint64, hdr__slot_121: cutlass.Uint64, hdr__slot_122: cutlass.Uint64, hdr__slot_123: cutlass.Uint64, hdr__slot_124: cutlass.Uint64, hdr__slot_125: cutlass.Uint64, hdr__slot_126: cutlass.Uint64, hdr__slot_127: cutlass.Uint64, hdr__slot_128: cutlass.Uint64, hdr__slot_129: cutlass.Uint64, hdr__slot_130: cutlass.Uint64, hdr__slot_131: cutlass.Uint64, hdr__slot_132: cutlass.Uint64, hdr__slot_133: cutlass.Uint64, hdr__slot_134: cutlass.Uint64, hdr__slot_135: cutlass.Uint64, hdr__slot_136: cutlass.Uint64, hdr__slot_137: cutlass.Uint64, hdr__slot_138: cutlass.Uint64, hdr__slot_139: cutlass.Uint64, hdr__slot_140: cutlass.Uint64, hdr__slot_141: cutlass.Uint64, hdr__slot_142: cutlass.Uint64, hdr__slot_143: cutlass.Uint64, hdr__slot_144: cutlass.Uint64, hdr__slot_145: cutlass.Uint64, hdr__slot_146: cutlass.Uint64, hdr__slot_147: cutlass.Uint64, hdr__slot_148: cutlass.Uint64, hdr__slot_149: cutlass.Uint64, hdr__slot_150: cutlass.Uint64, hdr__slot_151: cutlass.Uint64, hdr__slot_152: cutlass.Uint64, hdr__slot_153: cutlass.Uint64, hdr__slot_154: cutlass.Uint64, hdr__slot_155: cutlass.Uint64, hdr__slot_156: cutlass.Uint64, hdr__slot_157: cutlass.Uint64, hdr__slot_158: cutlass.Uint64, hdr__slot_159: cutlass.Uint64, hdr__slot_160: cutlass.Uint64, hdr__slot_161: cutlass.Uint64, hdr__slot_162: cutlass.Uint64, hdr__slot_163: cutlass.Uint64, hdr__slot_164: cutlass.Uint64, hdr__slot_165: cutlass.Uint64, hdr__slot_166: cutlass.Uint64, hdr__slot_167: cutlass.Uint64, hdr__slot_168: cutlass.Uint64, hdr__slot_169: cutlass.Uint64, hdr__slot_170: cutlass.Uint64, hdr__slot_171: cutlass.Uint64, hdr__slot_172: cutlass.Uint64, hdr__slot_173: cutlass.Uint64, hdr__slot_174: cutlass.Uint64, hdr__slot_175: cutlass.Uint64, hdr__slot_176: cutlass.Uint64, hdr__slot_177: cutlass.Uint64, hdr__slot_178: cutlass.Uint64, hdr__slot_179: cutlass.Uint64, hdr__slot_180: cutlass.Uint64, hdr__slot_181: cutlass.Uint64, hdr__slot_182: cutlass.Uint64, hdr__slot_183: cutlass.Uint64, hdr__slot_184: cutlass.Uint64, hdr__slot_185: cutlass.Uint64, hdr__slot_186: cutlass.Uint64, hdr__slot_187: cutlass.Uint64, hdr__slot_188: cutlass.Uint64, hdr__slot_189: cutlass.Uint64, hdr__slot_190: cutlass.Uint64, hdr__slot_191: cutlass.Uint64, hdr__slot_192: cutlass.Uint64, hdr__slot_193: cutlass.Uint64, hdr__slot_194: cutlass.Uint64, hdr__slot_195: cutlass.Uint64, hdr__slot_196: cutlass.Uint64, hdr__slot_197: cutlass.Uint64, hdr__slot_198: cutlass.Uint64, hdr__slot_199: cutlass.Uint64, hdr__slot_200: cutlass.Uint64, hdr__slot_201: cutlass.Uint64, hdr__slot_202: cutlass.Uint64, hdr__slot_203: cutlass.Uint64, hdr__slot_204: cutlass.Uint64, hdr__slot_205: cutlass.Uint64, hdr__slot_206: cutlass.Uint64, hdr__slot_207: cutlass.Uint64, hdr__slot_208: cutlass.Uint64, hdr__slot_209: cutlass.Uint64, hdr__slot_210: cutlass.Uint64, hdr__slot_211: cutlass.Uint64, hdr__slot_212: cutlass.Uint64, hdr__slot_213: cutlass.Uint64, hdr__slot_214: cutlass.Uint64, hdr__slot_215: cutlass.Uint64, hdr__slot_216: cutlass.Uint64, hdr__slot_217: cutlass.Uint64, hdr__slot_218: cutlass.Uint64, hdr__slot_219: cutlass.Uint64, hdr__slot_220: cutlass.Uint64, hdr__slot_221: cutlass.Uint64, hdr__slot_222: cutlass.Uint64, hdr__slot_223: cutlass.Uint64, hdr__slot_224: cutlass.Uint64, hdr__slot_225: cutlass.Uint64, hdr__slot_226: cutlass.Uint64, hdr__slot_227: cutlass.Uint64, hdr__slot_228: cutlass.Uint64, hdr__slot_229: cutlass.Uint64, hdr__slot_230: cutlass.Uint64, hdr__slot_231: cutlass.Uint64, hdr__slot_232: cutlass.Uint64, hdr__slot_233: cutlass.Uint64, hdr__slot_234: cutlass.Uint64, hdr__slot_235: cutlass.Uint64, hdr__slot_236: cutlass.Uint64, hdr__slot_237: cutlass.Uint64, hdr__slot_238: cutlass.Uint64, hdr__slot_239: cutlass.Uint64, hdr__slot_240: cutlass.Uint64, hdr__slot_241: cutlass.Uint64, hdr__slot_242: cutlass.Uint64, hdr__slot_243: cutlass.Uint64, hdr__slot_244: cutlass.Uint64, hdr__slot_245: cutlass.Uint64, hdr__slot_246: cutlass.Uint64, hdr__slot_247: cutlass.Uint64, hdr__slot_248: cutlass.Uint64, hdr__slot_249: cutlass.Uint64, hdr__slot_250: cutlass.Uint64, hdr__slot_251: cutlass.Uint64, hdr__slot_252: cutlass.Uint64, hdr__slot_253: cutlass.Uint64, hdr__slot_254: cutlass.Uint64, hdr__slot_255: cutlass.Uint64, hdr__slot_256: cutlass.Uint64, hdr__slot_257: cutlass.Uint64, hdr__slot_258: cutlass.Uint64, hdr__slot_259: cutlass.Uint64, hdr__slot_260: cutlass.Uint64, hdr__slot_261: cutlass.Uint64, hdr__slot_262: cutlass.Uint64, hdr__slot_263: cutlass.Uint64, hdr__slot_264: cutlass.Uint64, hdr__slot_265: cutlass.Uint64, hdr__slot_266: cutlass.Uint64, hdr__slot_267: cutlass.Uint64, hdr__slot_268: cutlass.Uint64, hdr__slot_269: cutlass.Uint64, hdr__slot_270: cutlass.Uint64, hdr__slot_271: cutlass.Uint64, hdr__slot_272: cutlass.Uint64, hdr__slot_273: cutlass.Uint64, hdr__slot_274: cutlass.Uint64, hdr__slot_275: cutlass.Uint64, hdr__slot_276: cutlass.Uint64, hdr__slot_277: cutlass.Uint64, hdr__slot_278: cutlass.Uint64, hdr__slot_279: cutlass.Uint64, hdr__slot_280: cutlass.Uint64, hdr__slot_281: cutlass.Uint64, hdr__slot_282: cutlass.Uint64, hdr__slot_283: cutlass.Uint64, hdr__slot_284: cutlass.Uint64, hdr__slot_285: cutlass.Uint64, hdr__slot_286: cutlass.Uint64, hdr__slot_287: cutlass.Uint64, hdr__slot_288: cutlass.Uint64, hdr__slot_289: cutlass.Uint64, hdr__slot_290: cutlass.Uint64, hdr__slot_291: cutlass.Uint64, hdr__slot_292: cutlass.Uint64, hdr__slot_293: cutlass.Uint64, hdr__slot_294: cutlass.Uint64, hdr__slot_295: cutlass.Uint64, hdr__slot_296: cutlass.Uint64, hdr__slot_297: cutlass.Uint64, hdr__slot_298: cutlass.Uint64, hdr__slot_299: cutlass.Uint64, hdr__slot_300: cutlass.Uint64, hdr__slot_301: cutlass.Uint64, hdr__slot_302: cutlass.Uint64, hdr__slot_303: cutlass.Uint64, hdr__slot_304: cutlass.Uint64, hdr__slot_305: cutlass.Uint64, hdr__slot_306: cutlass.Uint64, hdr__slot_307: cutlass.Uint64, hdr__slot_308: cutlass.Uint64, hdr__slot_309: cutlass.Uint64, hdr__slot_310: cutlass.Uint64, hdr__slot_311: cutlass.Uint64, hdr__slot_312: cutlass.Uint64, hdr__slot_313: cutlass.Uint64, hdr__slot_314: cutlass.Uint64, hdr__slot_315: cutlass.Uint64, hdr__slot_316: cutlass.Uint64, hdr__slot_317: cutlass.Uint64, hdr__slot_318: cutlass.Uint64, hdr__slot_319: cutlass.Uint64, hdr__slot_320: cutlass.Uint64, hdr__slot_321: cutlass.Uint64, hdr__slot_322: cutlass.Uint64, hdr__slot_323: cutlass.Uint64, hdr__slot_324: cutlass.Uint64, hdr__slot_325: cutlass.Uint64, hdr__slot_326: cutlass.Uint64, hdr__slot_327: cutlass.Uint64, hdr__slot_328: cutlass.Uint64, hdr__slot_329: cutlass.Uint64, hdr__slot_330: cutlass.Uint64, hdr__slot_331: cutlass.Uint64, hdr__slot_332: cutlass.Uint64, hdr__slot_333: cutlass.Uint64, hdr__slot_334: cutlass.Uint64, hdr__slot_335: cutlass.Uint64, hdr__slot_336: cutlass.Uint64, hdr__slot_337: cutlass.Uint64, hdr__slot_338: cutlass.Uint64, hdr__slot_339: cutlass.Uint64, hdr__slot_340: cutlass.Uint64, hdr__slot_341: cutlass.Uint64, hdr__slot_342: cutlass.Uint64, hdr__slot_343: cutlass.Uint64, hdr__slot_344: cutlass.Uint64, hdr__slot_345: cutlass.Uint64, hdr__slot_346: cutlass.Uint64, hdr__slot_347: cutlass.Uint64, hdr__slot_348: cutlass.Uint64, hdr__slot_349: cutlass.Uint64, hdr__slot_350: cutlass.Uint64, hdr__slot_351: cutlass.Uint64, hdr__slot_352: cutlass.Uint64, hdr__slot_353: cutlass.Uint64, hdr__slot_354: cutlass.Uint64, hdr__slot_355: cutlass.Uint64, hdr__slot_356: cutlass.Uint64, hdr__slot_357: cutlass.Uint64, hdr__slot_358: cutlass.Uint64, hdr__slot_359: cutlass.Uint64, hdr__slot_360: cutlass.Uint64, hdr__slot_361: cutlass.Uint64, hdr__slot_362: cutlass.Uint64, hdr__slot_363: cutlass.Uint64, hdr__slot_364: cutlass.Uint64, hdr__slot_365: cutlass.Uint64, hdr__slot_366: cutlass.Uint64, hdr__slot_367: cutlass.Uint64, hdr__slot_368: cutlass.Uint64, hdr__slot_369: cutlass.Uint64, hdr__slot_370: cutlass.Uint64, hdr__slot_371: cutlass.Uint64, hdr__slot_372: cutlass.Uint64, hdr__slot_373: cutlass.Uint64, hdr__slot_374: cutlass.Uint64, hdr__slot_375: cutlass.Uint64, hdr__slot_376: cutlass.Uint64, hdr__slot_377: cutlass.Uint64, hdr__slot_378: cutlass.Uint64, hdr__slot_379: cutlass.Uint64, hdr__slot_380: cutlass.Uint64, hdr__slot_381: cutlass.Uint64, hdr__slot_382: cutlass.Uint64, hdr__slot_383: cutlass.Uint64, hdr__slot_384: cutlass.Uint64, hdr__slot_385: cutlass.Uint64, hdr__slot_386: cutlass.Uint64, hdr__slot_387: cutlass.Uint64, hdr__slot_388: cutlass.Uint64, hdr__slot_389: cutlass.Uint64, hdr__slot_390: cutlass.Uint64, hdr__slot_391: cutlass.Uint64, hdr__slot_392: cutlass.Uint64, hdr__slot_393: cutlass.Uint64, hdr__slot_394: cutlass.Uint64, hdr__slot_395: cutlass.Uint64, hdr__slot_396: cutlass.Uint64, hdr__slot_397: cutlass.Uint64, hdr__slot_398: cutlass.Uint64, hdr__slot_399: cutlass.Uint64, hdr__slot_400: cutlass.Uint64, hdr__slot_401: cutlass.Uint64, hdr__slot_402: cutlass.Uint64, hdr__slot_403: cutlass.Uint64, hdr__slot_404: cutlass.Uint64, hdr__slot_405: cutlass.Uint64, hdr__slot_406: cutlass.Uint64, hdr__slot_407: cutlass.Uint64, hdr__slot_408: cutlass.Uint64, hdr__slot_409: cutlass.Uint64, hdr__slot_410: cutlass.Uint64, hdr__slot_411: cutlass.Uint64, hdr__slot_412: cutlass.Uint64, hdr__slot_413: cutlass.Uint64, hdr__slot_414: cutlass.Uint64, hdr__slot_415: cutlass.Uint64, hdr__slot_416: cutlass.Uint64, hdr__slot_417: cutlass.Uint64, hdr__slot_418: cutlass.Uint64, hdr__slot_419: cutlass.Uint64, hdr__slot_420: cutlass.Uint64, hdr__slot_421: cutlass.Uint64, hdr__slot_422: cutlass.Uint64, hdr__slot_423: cutlass.Uint64, hdr__slot_424: cutlass.Uint64, hdr__slot_425: cutlass.Uint64, hdr__slot_426: cutlass.Uint64, hdr__slot_427: cutlass.Uint64, hdr__slot_428: cutlass.Uint64, hdr__slot_429: cutlass.Uint64, hdr__slot_430: cutlass.Uint64, hdr__slot_431: cutlass.Uint64, O: cute.Pointer, meta: cute.Pointer, tile_stride: cutlass.Int32, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32, dbg: cute.Pointer, tl: cute.Pointer):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _O = cute.make_tensor(O, _flat_layout)
    _meta = cute.make_tensor(meta, _flat_layout)
    _dbg = cute.make_tensor(dbg, _flat_layout)
    _tl = cute.make_tensor(tl, _flat_layout)
    _cake_param_slots_base = cutlass.Uint64(Vt.get_ptr().toint()) + 128
    _hdr__base = _cake_param_slots_base + 0
    q_smem = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _q_smem = cute.make_tensor(q_smem, _flat_layout)
    q_smem_addr = smem + 1024
    k_smem = cute.recast_ptr(smem_raw + 33792, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _k_smem = cute.make_tensor(k_smem, _flat_layout)
    k_smem_addr = smem + 33792
    vt_smem_a = cute.recast_ptr(smem_raw + 132096, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _vt_smem_a = cute.make_tensor(vt_smem_a, _flat_layout)
    vt_smem_a_addr = smem + 132096
    vt_smem_b = cute.recast_ptr(smem_raw + 148480, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _vt_smem_b = cute.make_tensor(vt_smem_b, _flat_layout)
    vt_smem_b_addr = smem + 148480
    meta_smem = cute.recast_ptr(smem_raw + 230400, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Int32)
    _meta_smem = cute.make_tensor(meta_smem, _flat_layout)
    meta_smem_addr = smem + 230400
    merge_o = cute.recast_ptr(smem_raw + 17408, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _merge_o = cute.make_tensor(merge_o, _flat_layout)
    merge_o_addr = smem + 17408
    merge_ml = cute.recast_ptr(smem_raw + 231552, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _merge_ml = cute.make_tensor(merge_ml, _flat_layout)
    merge_ml_addr = smem + 231552
    smem_v7 = cute.recast_ptr(smem_raw + 148480, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _smem_v7 = cute.make_tensor(smem_v7, _flat_layout)
    smem_v7_addr = smem + 148480
    smem_v8 = cute.recast_ptr(smem_raw + 181248, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _smem_v8 = cute.make_tensor(smem_v8, _flat_layout)
    smem_v8_addr = smem + 181248
    smem_v9 = cute.recast_ptr(smem_raw + 214016, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _smem_v9 = cute.make_tensor(smem_v9, _flat_layout)
    smem_v9_addr = smem + 214016
    if (warp == 0):
        if prims.elect_sync():
            cute.arch.prefetch(Q.get_ptr(), tensormap=True)
            cute.arch.prefetch(K.get_ptr(), tensormap=True)
            cute.arch.prefetch(Vt.get_ptr(), tensormap=True)
    cute.arch.sync_warp()
    q_ready_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    q_empty_addr = cute.recast_ptr(smem_raw + 16, dtype=cutlass.Uint64)
    k_full_addr = cute.recast_ptr(smem_raw + 24, dtype=cutlass.Uint64)
    v_full_addr = cute.recast_ptr(smem_raw + 48, dtype=cutlass.Uint64)
    k_empty_addr = cute.recast_ptr(smem_raw + 72, dtype=cutlass.Uint64)
    v_empty_addr = cute.recast_ptr(smem_raw + 96, dtype=cutlass.Uint64)
    meta_full_addr = cute.recast_ptr(smem_raw + 120, dtype=cutlass.Uint64)
    meta_empty_addr = cute.recast_ptr(smem_raw + 136, dtype=cutlass.Uint64)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_ready_addr + 0, 1)
            cute.arch.mbarrier_init(q_ready_addr + 1, 1)
            cute.arch.mbarrier_init(q_empty_addr + 0, 8)
            cute.arch.mbarrier_init(k_full_addr + 0, 1)
            cute.arch.mbarrier_init(k_full_addr + 1, 1)
            cute.arch.mbarrier_init(k_full_addr + 2, 1)
            cute.arch.mbarrier_init(v_full_addr + 0, 1)
            cute.arch.mbarrier_init(v_full_addr + 1, 1)
            cute.arch.mbarrier_init(v_full_addr + 2, 1)
            cute.arch.mbarrier_init(k_empty_addr + 0, 8)
            cute.arch.mbarrier_init(k_empty_addr + 1, 8)
            cute.arch.mbarrier_init(k_empty_addr + 2, 8)
            cute.arch.mbarrier_init(v_empty_addr + 0, 8)
            cute.arch.mbarrier_init(v_empty_addr + 1, 8)
            cute.arch.mbarrier_init(v_empty_addr + 2, 8)
            cute.arch.mbarrier_init(meta_full_addr + 0, 64)
            cute.arch.mbarrier_init(meta_full_addr + 1, 64)
            cute.arch.mbarrier_init(meta_empty_addr + 0, 10)
            cute.arch.mbarrier_init(meta_empty_addr + 1, 10)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if warp <= 3:
        cute.arch.setmaxregister_decrease(24)
        w2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        w2_4 = cute.make_rmem_tensor((1,), cutlass.Int32)
        gk = cute.make_rmem_tensor((1,), cutlass.Int32)
        has_b = cute.make_rmem_tensor((1,), cutlass.Int32)
        i0_k = cute.make_rmem_tensor((1,), cutlass.Int32)
        has_b_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        gv = cute.make_rmem_tensor((1,), cutlass.Int32)
        v_has_b = cute.make_rmem_tensor((1,), cutlass.Int32)
        i0_v = cute.make_rmem_tensor((1,), cutlass.Int32)
        v_has_b_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        cta = cutlass.Int32(bid)
        row0 = cutlass.Int32((cta * tile_stride))
        hb = cutlass.Int32((cta * 12))
        if (warp == 1):
            prims.barrier_cta_sync(8, thread_count=384)
        else:
            cute.arch.barrier_arrive(barrier_id=8, number_of_threads=384, aligned=False)
        if (warp >= 2):
            st_t = cutlass.Int32((((warp - 2) * 32) + lane))
            if (warp == 2):
                if prims.elect_sync():
                    h_head_q = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (hb) * 2, 's16')))
                    h_qb0 = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + ((hb + 1)) * 2, 's16')))
                    h_qb1 = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + ((hb + 2)) * 2, 's16')))
                    h_mode = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + ((hb + 3)) * 2, 's16')))
                    cute.arch.mbarrier_arrive_and_expect_tx(q_ready_addr, 16384)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32(q_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                        Q.get_ptr(),
                        [cutlass.Int32(0), cutlass.Int32(((h_head_q * seqlen_q) + (h_qb0 * 64))), cutlass.Int32(0)],
                        q_ready_addr,
                        l2_cache_hint=0x12F0000000000000,
                        mode=prims.TMALoadMode.TILE,
                    )
                    if (h_mode != 1):
                        cute.arch.mbarrier_arrive_and_expect_tx(q_ready_addr + 1, 16384)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            Q.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(((h_head_q * seqlen_q) + (h_qb1 * 64))), cutlass.Int32(0)],
                            q_ready_addr + 1,
                            l2_cache_hint=0x12F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
            nt_st_ld = cutlass.Int32(_meta[((row0 * 144) + 7)])
            slot = 0
            mrow = cutlass.Int32((row0 * 144))
            w0 = cutlass.Int32(_meta[(mrow + st_t)])
            w1 = cutlass.Int32(_meta[((mrow + st_t) + 64)])
            w2[0] = cutlass.Int32(0)
            if ((st_t + 128) < 144):
                w2[0] = cutlass.Int32(_meta[((mrow + st_t) + 128)])
            while not prims.mbarrier_wait_parity(meta_empty_addr + slot, 1, prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                pass
            if (warp == 2):
                _shfl_0 = cute.arch.shuffle_sync(w0, 0, mask=4294967295, mask_and_clamp=31)
                head = cutlass.Int32(_shfl_0)
                _shfl_1 = cute.arch.shuffle_sync(w0, 1, mask=4294967295, mask_and_clamp=31)
                qb0 = cutlass.Int32(_shfl_1)
                _shfl_2 = cute.arch.shuffle_sync(w0, 2, mask=4294967295, mask_and_clamp=31)
                qb1 = cutlass.Int32(_shfl_2)
                _shfl_3 = cute.arch.shuffle_sync(w0, 3, mask=4294967295, mask_and_clamp=31)
                mode_w = cutlass.Int32(_shfl_3)
                if prims.elect_sync():
                    pass
            prims.store_ext(cutlass.Int32(w0).ir_value(), (meta_smem + (((slot * 144) + st_t))))
            prims.store_ext(cutlass.Int32(w1).ir_value(), (meta_smem + ((((slot * 144) + st_t) + 64))))
            if ((st_t + 128) < 144):
                prims.store_ext(cutlass.Int32(w2[0]).ir_value(), (meta_smem + ((((slot * 144) + st_t) + 128))))
            cute.arch.mbarrier_arrive(meta_full_addr + slot)
            _shfl_4 = cute.arch.shuffle_sync(nt_st_ld, 0, mask=4294967295, mask_and_clamp=31)
            nt_st = cutlass.Int32(_shfl_4)
            for ti in cutlass.range(cutlass.Int32(1), cutlass.Int32(nt_st), cutlass.Int32(1), unroll=1, unroll_full=False):
                slot_0 = cutlass.Int32((ti & 1))
                mrow_1 = cutlass.Int32(((row0 + ti) * 144))
                w0_2 = cutlass.Int32(_meta[(mrow_1 + st_t)])
                w1_3 = cutlass.Int32(_meta[((mrow_1 + st_t) + 64)])
                w2_4[0] = cutlass.Int32(0)
                if ((st_t + 128) < 144):
                    w2_4[0] = cutlass.Int32(_meta[((mrow_1 + st_t) + 128)])
                while not prims.mbarrier_wait_parity(meta_empty_addr + slot_0, (((ti >> 1) + 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                if (warp == 2):
                    _shfl_5 = cute.arch.shuffle_sync(w0_2, 0, mask=4294967295, mask_and_clamp=31)
                    head_1 = cutlass.Int32(_shfl_5)
                    _shfl_6 = cute.arch.shuffle_sync(w0_2, 1, mask=4294967295, mask_and_clamp=31)
                    qb0_1 = cutlass.Int32(_shfl_6)
                    _shfl_7 = cute.arch.shuffle_sync(w0_2, 2, mask=4294967295, mask_and_clamp=31)
                    qb1_1 = cutlass.Int32(_shfl_7)
                    _shfl_8 = cute.arch.shuffle_sync(w0_2, 3, mask=4294967295, mask_and_clamp=31)
                    mode_w_1 = cutlass.Int32(_shfl_8)
                    if prims.elect_sync():
                        if (ti > 0):
                            while not prims.mbarrier_wait_parity(q_empty_addr, ((ti - 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            cute.arch.mbarrier_arrive_and_expect_tx(q_ready_addr, 16384)
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32(q_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                Q.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(((head_1 * seqlen_q) + (qb0_1 * 64))), cutlass.Int32(0)],
                                q_ready_addr,
                                l2_cache_hint=0x12F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                            if (mode_w_1 != 1):
                                cute.arch.mbarrier_arrive_and_expect_tx(q_ready_addr + 1, 16384)
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    Q.get_ptr(),
                                    [cutlass.Int32(0), cutlass.Int32(((head_1 * seqlen_q) + (qb1_1 * 64))), cutlass.Int32(0)],
                                    q_ready_addr + 1,
                                    l2_cache_hint=0x12F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                )
                prims.store_ext(cutlass.Int32(w0_2).ir_value(), (meta_smem + (((slot_0 * 144) + st_t))))
                prims.store_ext(cutlass.Int32(w1_3).ir_value(), (meta_smem + ((((slot_0 * 144) + st_t) + 64))))
                if ((st_t + 128) < 144):
                    prims.store_ext(cutlass.Int32(w2_4[0]).ir_value(), (meta_smem + ((((slot_0 * 144) + st_t) + 128))))
                cute.arch.mbarrier_arrive(meta_full_addr + slot_0)
        if (warp == 0):
            if prims.elect_sync():
                gk[0] = cutlass.Int32(0)
                h_head_k = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (hb) * 2, 's16')))
                n_hdr_k = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + ((hb + 4)) * 2, 's16')))
                kv_base_h = cutlass.Int32((h_head_k * seqlen_k))
                for hp in cutlass.range(cutlass.Int32(0), cutlass.Int32(3), cutlass.Int32(1)):
                    if (n_hdr_k > hp):
                        while not prims.mbarrier_wait_parity(k_empty_addr + (hp % 3), ((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(hp).ir_value(), cutlass.Int32(3).ir_value())) + 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        hblk_a = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (((hb + 5) + (2 * hp))) * 2, 's16')))
                        hblk_b = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (((hb + 6) + (2 * hp))) * 2, 's16')))
                        kv_row_a = cutlass.Int32((kv_base_h + (hblk_a * 64)))
                        kv_row_b = cutlass.Int32((kv_base_h + (hblk_b * 64)))
                        has_b[0] = cutlass.Int32(1)
                        _if_condition_0 = cutlass.Boolean((hblk_b < 0))
                        has_b[0] = cutlass.Int32(cutlass.select_(_if_condition_0, cutlass.Int32(0), has_b[0]))
                        k_dst = cutlass.Int32((k_smem_addr + cutlass.Uint32(((hp % 3) * 32768))))
                        cute.arch.mbarrier_arrive_and_expect_tx(k_full_addr + (hp % 3), (16384 * (1 + has_b[0])))
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32(k_dst), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            K.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(kv_row_a), cutlass.Int32(0)],
                            k_full_addr + (hp % 3),
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_dst + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            K.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(kv_row_a), cutlass.Int32(1)],
                            k_full_addr + (hp % 3),
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        if (has_b[0] != 0):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_dst + 8192)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                K.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(kv_row_b), cutlass.Int32(0)],
                                k_full_addr + (hp % 3),
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((k_dst + 16384) + 8192)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                K.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(kv_row_b), cutlass.Int32(1)],
                                k_full_addr + (hp % 3),
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                while not prims.mbarrier_wait_parity(meta_full_addr, 0, prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                nt_k = cutlass.Int32(_meta_smem[7])
                for ti_1 in cutlass.range(cutlass.Int32(0), cutlass.Int32(nt_k), cutlass.Int32(1), unroll=1, unroll_full=False):
                    slot_k = cutlass.Int32((ti_1 & 1))
                    mb_k = cutlass.Int32((slot_k * 144))
                    if (ti_1 > 0):
                        while not prims.mbarrier_wait_parity(meta_full_addr + slot_k, ((ti_1 >> 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                    head_k = cutlass.Int32(_meta_smem[mb_k])
                    n_seq_k = cutlass.Int32(_meta_smem[(mb_k + 4)])
                    kv_base = cutlass.Int32((head_k * seqlen_k))
                    i0_k[0] = cutlass.Int32(0)
                    _if_condition_1 = cutlass.Boolean((ti_1 == 0))
                    i0_k[0] = cutlass.Int32(cutlass.select_(_if_condition_1, cutlass.Int32(n_hdr_k), i0_k[0]))
                    for i in cutlass.range(cutlass.Int32(i0_k[0]), cutlass.Int32(n_seq_k), cutlass.Int32(1), unroll=1, unroll_full=False):
                        p = cutlass.Int32((gk[0] + i))
                        stage = cutlass.Int32((p % 3))
                        while not prims.mbarrier_wait_parity(k_empty_addr + stage, ((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p).ir_value(), cutlass.Int32(3).ir_value())) + 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        sw = cutlass.Int32(_meta_smem[((mb_k + 8) + i)])
                        blk_a = cutlass.Int32(((sw << 16) >> 16))
                        blk_b = cutlass.Int32((sw >> 16))
                        kv_row_a_1 = cutlass.Int32((kv_base + (blk_a * 64)))
                        kv_row_b_1 = cutlass.Int32((kv_base + (blk_b * 64)))
                        has_b_1[0] = cutlass.Int32(1)
                        _if_condition_2 = cutlass.Boolean((blk_b < 0))
                        has_b_1[0] = cutlass.Int32(cutlass.select_(_if_condition_2, cutlass.Int32(0), has_b_1[0]))
                        k_dst_1 = cutlass.Int32((k_smem_addr + cutlass.Uint32((stage * 32768))))
                        cute.arch.mbarrier_arrive_and_expect_tx(k_full_addr + stage, (16384 * (1 + has_b_1[0])))
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32(k_dst_1), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            K.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(kv_row_a_1), cutlass.Int32(0)],
                            k_full_addr + stage,
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_dst_1 + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            K.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(kv_row_a_1), cutlass.Int32(1)],
                            k_full_addr + stage,
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        if (has_b_1[0] != 0):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_dst_1 + 8192)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                K.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(kv_row_b_1), cutlass.Int32(0)],
                                k_full_addr + stage,
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((k_dst_1 + 16384) + 8192)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                K.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(kv_row_b_1), cutlass.Int32(1)],
                                k_full_addr + stage,
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                    gk[0] += cutlass.Int32(n_seq_k)
                    cute.arch.mbarrier_arrive(meta_empty_addr + slot_k)
        if (warp == 1):
            if prims.elect_sync():
                gv[0] = cutlass.Int32(0)
                h_head_v = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (hb) * 2, 's16')))
                n_hdr_v = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + ((hb + 4)) * 2, 's16')))
                kv_base_hv = cutlass.Int32((h_head_v * seqlen_k))
                for hp_1 in cutlass.range(cutlass.Int32(0), cutlass.Int32(3), cutlass.Int32(1)):
                    if (n_hdr_v > hp_1):
                        while not prims.mbarrier_wait_parity(v_empty_addr + (hp_1 % 3), ((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(hp_1).ir_value(), cutlass.Int32(3).ir_value())) + 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        hvblk_a = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (((hb + 5) + (2 * hp_1))) * 2, 's16')))
                        hvblk_b = cutlass.Int32(cutlass.Int16(_cake_ldparam_b32(_hdr__base + (((hb + 6) + (2 * hp_1))) * 2, 's16')))
                        v_row_a = cutlass.Int32((kv_base_hv + (hvblk_a * 64)))
                        v_row_b = cutlass.Int32((kv_base_hv + (hvblk_b * 64)))
                        v_has_b[0] = cutlass.Int32(1)
                        _if_condition_3 = cutlass.Boolean((hvblk_b < 0))
                        v_has_b[0] = cutlass.Int32(cutlass.select_(_if_condition_3, cutlass.Int32(0), v_has_b[0]))
                        cute.arch.mbarrier_arrive_and_expect_tx(v_full_addr + (hp_1 % 3), (16384 * (1 + v_has_b[0])))
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_a_addr + cutlass.Uint32(((hp_1 % 3) * 32768)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            Vt.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(v_row_a).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                            v_full_addr + (hp_1 % 3),
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        if (v_has_b[0] != 0):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_b_addr + cutlass.Uint32(((hp_1 % 3) * 32768)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                Vt.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(v_row_b).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                                v_full_addr + (hp_1 % 3),
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                while not prims.mbarrier_wait_parity(meta_full_addr, 0, prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                nt_v = cutlass.Int32(_meta_smem[7])
                for ti_2 in cutlass.range(cutlass.Int32(0), cutlass.Int32(nt_v), cutlass.Int32(1), unroll=1, unroll_full=False):
                    slot_v = cutlass.Int32((ti_2 & 1))
                    mb_v = cutlass.Int32((slot_v * 144))
                    if (ti_2 > 0):
                        while not prims.mbarrier_wait_parity(meta_full_addr + slot_v, ((ti_2 >> 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                    head_v = cutlass.Int32(_meta_smem[mb_v])
                    n_seq_v = cutlass.Int32(_meta_smem[(mb_v + 4)])
                    kv_base_v = cutlass.Int32((head_v * seqlen_k))
                    i0_v[0] = cutlass.Int32(0)
                    _if_condition_4 = cutlass.Boolean((ti_2 == 0))
                    i0_v[0] = cutlass.Int32(cutlass.select_(_if_condition_4, cutlass.Int32(n_hdr_v), i0_v[0]))
                    for i_1 in cutlass.range(cutlass.Int32(i0_v[0]), cutlass.Int32(n_seq_v), cutlass.Int32(1), unroll=1, unroll_full=False):
                        pv = cutlass.Int32((gv[0] + i_1))
                        stage_v = cutlass.Int32((pv % 3))
                        while not prims.mbarrier_wait_parity(v_empty_addr + stage_v, ((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(pv).ir_value(), cutlass.Int32(3).ir_value())) + 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        svw = cutlass.Int32(_meta_smem[((mb_v + 8) + i_1)])
                        vblk_a = cutlass.Int32(((svw << 16) >> 16))
                        vblk_b = cutlass.Int32((svw >> 16))
                        v_row_a_1 = cutlass.Int32((kv_base_v + (vblk_a * 64)))
                        v_row_b_1 = cutlass.Int32((kv_base_v + (vblk_b * 64)))
                        v_has_b_1[0] = cutlass.Int32(1)
                        _if_condition_5 = cutlass.Boolean((vblk_b < 0))
                        v_has_b_1[0] = cutlass.Int32(cutlass.select_(_if_condition_5, cutlass.Int32(0), v_has_b_1[0]))
                        cute.arch.mbarrier_arrive_and_expect_tx(v_full_addr + stage_v, (16384 * (1 + v_has_b_1[0])))
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_a_addr + cutlass.Uint32((stage_v * 32768)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            Vt.get_ptr(),
                            [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(v_row_a_1).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                            v_full_addr + stage_v,
                            l2_cache_hint=0x14F0000000000000,
                            mode=prims.TMALoadMode.TILE,
                        )
                        if (v_has_b_1[0] != 0):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_b_addr + cutlass.Uint32((stage_v * 32768)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                Vt.get_ptr(),
                                [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(v_row_b_1).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                                v_full_addr + stage_v,
                                l2_cache_hint=0x14F0000000000000,
                                mode=prims.TMALoadMode.TILE,
                            )
                    gv[0] += cutlass.Int32(n_seq_v)
                    cute.arch.mbarrier_arrive(meta_empty_addr + slot_v)
    elif warp >= 4 and warp <= 11:
        cute.arch.setmaxregister_increase(240)
        row_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        row_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        row_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        row_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        gbase = cute.make_rmem_tensor((1,), cutlass.Int32)
        prev = cute.make_rmem_tensor((1,), cutlass.Int32)
        cur_pos = cute.make_rmem_tensor((1,), cutlass.Int32)
        cur_has2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        f_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        f_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        m0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        m1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        f_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        f_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sa0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sa1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sb0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sb1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        new_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        new_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        m0_0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        m1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        merged_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        merged_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        new_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        new_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sa0_0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sa1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sb0_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        sb1_3 = cute.make_rmem_tensor((1,), cutlass.Float32)
        store_wg = cute.make_rmem_tensor((1,), cutlass.Int32)
        do_merge = cute.make_rmem_tensor((1,), cutlass.Int32)
        a0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        b0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        b1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        cta_m = cutlass.Int32(bid)
        warp_ld = cutlass.Int32((warp - 4))
        _shfl_9 = cute.arch.shuffle_sync(warp_ld, 0, mask=4294967295, mask_and_clamp=31)
        consumer_warp = cutlass.Int32(_shfl_9)
        cwg = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(consumer_warp).ir_value(), cutlass.Int32(4).ir_value())))
        warp_in_wg = cutlass.Int32((consumer_warp % 4))
        tid_wg = cutlass.Int32(((warp_in_wg * 32) + lane))
        m0_local = cutlass.Int32(((warp_in_wg * 16) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
        m1_local = cutlass.Int32((m0_local + 8))
        quad = cutlass.Int32(((warp_in_wg * 8) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
        tid_c = cutlass.Int32(((cwg * 128) + tid_wg))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + (tid_c)))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((256 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((512 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((768 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((1024 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((1280 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((1536 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((1792 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((2048 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((2304 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((2560 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((2816 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((3072 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((3328 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((3584 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v7 + ((3840 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + (tid_c)))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((256 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((512 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((768 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((1024 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((1280 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((1536 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((1792 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((2048 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((2304 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((2560 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((2816 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((3072 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((3328 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((3584 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v8 + ((3840 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + (tid_c)))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((256 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((512 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((768 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((1024 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((1280 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((1536 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((1792 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((2048 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((2304 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((2560 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((2816 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((3072 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((3328 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((3584 + tid_c))))
        prims.store_ext(cutlass.Float32(0.0).ir_value(), (smem_v9 + ((3840 + tid_c))))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier_arrive(barrier_id=8, number_of_threads=384, aligned=False)
        d_o = cute.make_rmem_tensor((64,), cutlass.Float32)
        d_qk = cute.make_rmem_tensor((64,), cutlass.Float32)
        p_bf16 = cute.make_rmem_tensor((32,), cutlass.Uint32)
        poly_tf = [None] * 2
        poly_ti = [None] * 2
        q_frag = cute.make_rmem_tensor((32,), cutlass.Uint32)
        row_max0[0] = cutlass.Float32((0 - float("inf")))
        row_max1[0] = cutlass.Float32((0 - float("inf")))
        row_sum0[0] = cutlass.Float32(0.0)
        row_sum1[0] = cutlass.Float32(0.0)
        q_ld_row = cutlass.Int32((((warp_in_wg * 16) + (lane % 8)) + (8 * (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(8).ir_value())) % 2))))
        q_ld_col_lane = cutlass.Int32((8 * cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(16).ir_value()))))
        gbase[0] = cutlass.Int32(0)
        prev[0] = cutlass.Int32(-1)
        while not prims.mbarrier_wait_parity(meta_full_addr, 0, prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _shfl_10 = cute.arch.shuffle_sync(_meta_smem[7], 0, mask=4294967295, mask_and_clamp=31)
        nt_m = cutlass.Int32(_shfl_10)
        for ti_3 in cutlass.range(cutlass.Int32(0), cutlass.Int32(nt_m), cutlass.Int32(1), unroll=1, unroll_full=False):
            slot_m = cutlass.Int32((ti_3 & 1))
            mbase = cutlass.Int32((slot_m * 144))
            if (ti_3 > 0):
                while not prims.mbarrier_wait_parity(meta_full_addr + slot_m, ((ti_3 >> 1) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
            _shfl_11 = cute.arch.shuffle_sync(_meta_smem[mbase], 0, mask=4294967295, mask_and_clamp=31)
            head_m = cutlass.Int32(_shfl_11)
            _shfl_12 = cute.arch.shuffle_sync(_meta_smem[(mbase + 1)], 0, mask=4294967295, mask_and_clamp=31)
            qb0_m = cutlass.Int32(_shfl_12)
            _shfl_13 = cute.arch.shuffle_sync(_meta_smem[(mbase + 2)], 0, mask=4294967295, mask_and_clamp=31)
            qb1_m = cutlass.Int32(_shfl_13)
            _shfl_14 = cute.arch.shuffle_sync(_meta_smem[(mbase + 4)], 0, mask=4294967295, mask_and_clamp=31)
            n_seq_m = cutlass.Int32(_shfl_14)
            _shfl_15 = cute.arch.shuffle_sync(_meta_smem[((mbase + 5) + cwg)], 0, mask=4294967295, mask_and_clamp=31)
            n_own = cutlass.Int32(_shfl_15)
            _shfl_16 = cute.arch.shuffle_sync(_meta_smem[(((mbase + 5) + 1) - cwg)], 0, mask=4294967295, mask_and_clamp=31)
            n_other = cutlass.Int32(_shfl_16)
            _shfl_17 = cute.arch.shuffle_sync(_meta_smem[(mbase + 3)], 0, mask=4294967295, mask_and_clamp=31)
            mode_m = cutlass.Int32(_shfl_17)
            is_split = cutlass.Int32((mode_m == 1))
            my_qb = cutlass.Int32((qb0_m + (cwg * (qb1_m - qb0_m))))
            qs = cutlass.Int32((0 if (is_split != 0) else cwg))
            own_base_w = cutlass.Int32(((mbase + 76) + (cwg * 34)))
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
            row_max0[0] = cutlass.Float32((0 - float("inf")))
            row_max1[0] = cutlass.Float32((0 - float("inf")))
            row_sum0[0] = cutlass.Float32(0.0)
            row_sum1[0] = cutlass.Float32(0.0)
            while not prims.mbarrier_wait_parity(q_ready_addr + qs, (ti_3 & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                pass
            q_half_base = cutlass.Int32(((q_smem_addr + cutlass.Uint32((qs * 16384))) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes = cutlass.Int32((q_ld_col_lane * 2))
            _ldmatrix_6 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base + (q_col_bytes ^ (((q_half_base >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[0] = cutlass.Uint32(_ldmatrix_6[0])
            q_frag[1] = cutlass.Uint32(_ldmatrix_6[1])
            q_frag[2] = cutlass.Uint32(_ldmatrix_6[2])
            q_frag[3] = cutlass.Uint32(_ldmatrix_6[3])
            q_half_base_0 = cutlass.Int32(((q_smem_addr + cutlass.Uint32((qs * 16384))) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_1 = cutlass.Int32(((16 + q_ld_col_lane) * 2))
            _ldmatrix_7 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_0 + (q_col_bytes_1 ^ (((q_half_base_0 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[4] = cutlass.Uint32(_ldmatrix_7[0])
            q_frag[5] = cutlass.Uint32(_ldmatrix_7[1])
            q_frag[6] = cutlass.Uint32(_ldmatrix_7[2])
            q_frag[7] = cutlass.Uint32(_ldmatrix_7[3])
            q_half_base_2 = cutlass.Int32(((q_smem_addr + cutlass.Uint32((qs * 16384))) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_3 = cutlass.Int32(((32 + q_ld_col_lane) * 2))
            _ldmatrix_8 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_2 + (q_col_bytes_3 ^ (((q_half_base_2 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[8] = cutlass.Uint32(_ldmatrix_8[0])
            q_frag[9] = cutlass.Uint32(_ldmatrix_8[1])
            q_frag[10] = cutlass.Uint32(_ldmatrix_8[2])
            q_frag[11] = cutlass.Uint32(_ldmatrix_8[3])
            q_half_base_4 = cutlass.Int32(((q_smem_addr + cutlass.Uint32((qs * 16384))) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_5 = cutlass.Int32(((48 + q_ld_col_lane) * 2))
            _ldmatrix_9 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_4 + (q_col_bytes_5 ^ (((q_half_base_4 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[12] = cutlass.Uint32(_ldmatrix_9[0])
            q_frag[13] = cutlass.Uint32(_ldmatrix_9[1])
            q_frag[14] = cutlass.Uint32(_ldmatrix_9[2])
            q_frag[15] = cutlass.Uint32(_ldmatrix_9[3])
            q_half_base_6 = cutlass.Int32((((q_smem_addr + cutlass.Uint32((qs * 16384))) + 8192) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_7 = cutlass.Int32((q_ld_col_lane * 2))
            _ldmatrix_10 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_6 + (q_col_bytes_7 ^ (((q_half_base_6 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[16] = cutlass.Uint32(_ldmatrix_10[0])
            q_frag[17] = cutlass.Uint32(_ldmatrix_10[1])
            q_frag[18] = cutlass.Uint32(_ldmatrix_10[2])
            q_frag[19] = cutlass.Uint32(_ldmatrix_10[3])
            q_half_base_8 = cutlass.Int32((((q_smem_addr + cutlass.Uint32((qs * 16384))) + 8192) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_9 = cutlass.Int32(((16 + q_ld_col_lane) * 2))
            _ldmatrix_11 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_8 + (q_col_bytes_9 ^ (((q_half_base_8 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[20] = cutlass.Uint32(_ldmatrix_11[0])
            q_frag[21] = cutlass.Uint32(_ldmatrix_11[1])
            q_frag[22] = cutlass.Uint32(_ldmatrix_11[2])
            q_frag[23] = cutlass.Uint32(_ldmatrix_11[3])
            q_half_base_10 = cutlass.Int32((((q_smem_addr + cutlass.Uint32((qs * 16384))) + 8192) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_11 = cutlass.Int32(((32 + q_ld_col_lane) * 2))
            _ldmatrix_12 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_10 + (q_col_bytes_11 ^ (((q_half_base_10 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[24] = cutlass.Uint32(_ldmatrix_12[0])
            q_frag[25] = cutlass.Uint32(_ldmatrix_12[1])
            q_frag[26] = cutlass.Uint32(_ldmatrix_12[2])
            q_frag[27] = cutlass.Uint32(_ldmatrix_12[3])
            q_half_base_12 = cutlass.Int32((((q_smem_addr + cutlass.Uint32((qs * 16384))) + 8192) + cutlass.Uint32((q_ld_row * 128))))
            q_col_bytes_13 = cutlass.Int32(((48 + q_ld_col_lane) * 2))
            _ldmatrix_13 = prims.ldmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((q_half_base_12 + (q_col_bytes_13 ^ (((q_half_base_12 >> 7) & 7) << 4)))), mem_space=cute.AddressSpace.smem, assumed_align=16).to_llvm_ptr(), num=4, layout=prims.MMALayout.ROW, shape=prims.LoadShape.M8N8)
            q_frag[28] = cutlass.Uint32(_ldmatrix_13[0])
            q_frag[29] = cutlass.Uint32(_ldmatrix_13[1])
            q_frag[30] = cutlass.Uint32(_ldmatrix_13[2])
            q_frag[31] = cutlass.Uint32(_ldmatrix_13[3])
            if prims.elect_sync():
                cute.arch.mbarrier_arrive(q_empty_addr)
            cur_pos[0] = cutlass.Int32(0)
            cur_has2[0] = cutlass.Int32(1)
            if (n_own > 0):
                w_e = cutlass.Int32(_meta_smem[own_base_w])
                e0 = cutlass.Int32((w_e & 65535))
                pos0 = cutlass.Int32((gbase[0] + (e0 >> 3)))
                has2_0 = cutlass.Int32(((e0 >> 2) & 1))
                for p_1 in cutlass.range(cutlass.Int32((prev[0] + 1)), cutlass.Int32(pos0), cutlass.Int32(1), unroll=1, unroll_full=False):
                    while not prims.mbarrier_wait_parity(k_full_addr + (p_1 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_1).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                        pass
                    if prims.elect_sync():
                        cute.arch.mbarrier_arrive(k_empty_addr + (p_1 % 3))
                    while not prims.mbarrier_wait_parity(v_full_addr + (p_1 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_1).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                        pass
                    if prims.elect_sync():
                        cute.arch.mbarrier_arrive(v_empty_addr + (p_1 % 3))
                cur_pos[0] = cutlass.Int32(pos0)
                cur_has2[0] = cutlass.Int32(has2_0)
                stage_0 = cutlass.Int32((pos0 % 3))
                while not prims.mbarrier_wait_parity(k_full_addr + stage_0, (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(pos0).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                cute.nvgpu.warpgroup.fence()
                _wgmma_b_0_0_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + cutlass.Uint32((stage_0 * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                _wgmma_b_0_0 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_0_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_0_raw)))
                _wgmma_14_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64(_wgmma_b_0_0).ir_value(),
                        cutlass.Uint32(q_frag[(0) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(0) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(0) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(0) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 0, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_14, position=[63]))
                _wgmma_15_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_0 + 2)).ir_value(),
                        cutlass.Uint32(q_frag[(4) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(4) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(4) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(4) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_15, position=[63]))
                _wgmma_16_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_0 + 4)).ir_value(),
                        cutlass.Uint32(q_frag[(8) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(8) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(8) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(8) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_16, position=[63]))
                _wgmma_17_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_0 + 6)).ir_value(),
                        cutlass.Uint32(q_frag[(12) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(12) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(12) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(12) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_17, position=[63]))
                _wgmma_b_0_1_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + cutlass.Uint32((stage_0 * 32768))) + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                _wgmma_b_0_1 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw)))
                _wgmma_18_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64(_wgmma_b_0_1).ir_value(),
                        cutlass.Uint32(q_frag[(16) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(16) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(16) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(16) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_18, position=[63]))
                _wgmma_19_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_1 + 2)).ir_value(),
                        cutlass.Uint32(q_frag[(20) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(20) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(20) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(20) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_19, position=[63]))
                _wgmma_20_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_20 = cutlass_llvm.inline_asm(
                    _wgmma_20_ty,
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_1 + 4)).ir_value(),
                        cutlass.Uint32(q_frag[(24) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(24) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(24) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(24) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[0]))
                d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[1]))
                d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[2]))
                d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[3]))
                d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[4]))
                d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[5]))
                d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[6]))
                d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[7]))
                d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[8]))
                d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[9]))
                d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[10]))
                d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[11]))
                d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[12]))
                d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[13]))
                d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[14]))
                d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[15]))
                d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[16]))
                d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[17]))
                d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[18]))
                d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[19]))
                d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[20]))
                d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[21]))
                d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[22]))
                d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[23]))
                d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[24]))
                d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[25]))
                d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[26]))
                d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[27]))
                d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[28]))
                d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[29]))
                d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[30]))
                d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[31]))
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_20, position=[63]))
                _wgmma_21_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_21 = cutlass_llvm.inline_asm(
                    _wgmma_21_ty,
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
                        cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                        cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                        cutlass.Uint64((_wgmma_b_0_1 + 6)).ir_value(),
                        cutlass.Uint32(q_frag[(28) + 0]).ir_value(),
                        cutlass.Uint32(q_frag[(28) + 1]).ir_value(),
                        cutlass.Uint32(q_frag[(28) + 2]).ir_value(),
                        cutlass.Uint32(q_frag[(28) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[0]))
                d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[1]))
                d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[2]))
                d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[3]))
                d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[4]))
                d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[5]))
                d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[6]))
                d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[7]))
                d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[8]))
                d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[9]))
                d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[10]))
                d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[11]))
                d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[12]))
                d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[13]))
                d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[14]))
                d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[15]))
                d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[16]))
                d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[17]))
                d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[18]))
                d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[19]))
                d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[20]))
                d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[21]))
                d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[22]))
                d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[23]))
                d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[24]))
                d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[25]))
                d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[26]))
                d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[27]))
                d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[28]))
                d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[29]))
                d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[30]))
                d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[31]))
                d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[32]))
                d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[33]))
                d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[34]))
                d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[35]))
                d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[36]))
                d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[37]))
                d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[38]))
                d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[39]))
                d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[40]))
                d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[41]))
                d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[42]))
                d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[43]))
                d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[44]))
                d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[45]))
                d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[46]))
                d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[47]))
                d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[48]))
                d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[49]))
                d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[50]))
                d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[51]))
                d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[52]))
                d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[53]))
                d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[54]))
                d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[55]))
                d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[56]))
                d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[57]))
                d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[58]))
                d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[59]))
                d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[60]))
                d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[61]))
                d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[62]))
                d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_21, position=[63]))
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive(k_empty_addr + (pos0 % 3))
                f_max0[0] = cutlass.Float32((0 - float("inf")))
                f_max1[0] = cutlass.Float32((0 - float("inf")))
                m0[0] = cutlass.Float32((0 - float("inf")))
                m1[0] = cutlass.Float32((0 - float("inf")))
                if (scale_log2 >= 0.0):
                    _max_0 = cute.arch.fmax(d_qk[0], d_qk[1], ftz=False)
                    _max_1 = cute.arch.fmax(d_qk[4], d_qk[5], ftz=False)
                    _max_2 = cute.arch.fmax(d_qk[8], d_qk[9], ftz=False)
                    _max_3 = cute.arch.fmax(d_qk[12], d_qk[13], ftz=False)
                    _max_4 = cute.arch.fmax(d_qk[16], d_qk[17], ftz=False)
                    _max_5 = cute.arch.fmax(d_qk[20], d_qk[21], ftz=False)
                    _max_6 = cute.arch.fmax(d_qk[24], d_qk[25], ftz=False)
                    _max_7 = cute.arch.fmax(d_qk[28], d_qk[29], ftz=False)
                    _max_8 = cute.arch.fmax(_max_0, _max_1, ftz=False)
                    _max_9 = cute.arch.fmax(_max_2, _max_3, ftz=False)
                    _max_10 = cute.arch.fmax(_max_4, _max_5, ftz=False)
                    _max_11 = cute.arch.fmax(_max_6, _max_7, ftz=False)
                    _max_12 = cute.arch.fmax(_max_8, _max_9, ftz=False)
                    _max_13 = cute.arch.fmax(_max_10, _max_11, ftz=False)
                    _max_14 = cute.arch.fmax(_max_12, _max_13, ftz=False)
                    _max_15 = cute.arch.fmax(d_qk[2], d_qk[3], ftz=False)
                    _max_16 = cute.arch.fmax(d_qk[6], d_qk[7], ftz=False)
                    _max_17 = cute.arch.fmax(d_qk[10], d_qk[11], ftz=False)
                    _max_18 = cute.arch.fmax(d_qk[14], d_qk[15], ftz=False)
                    _max_19 = cute.arch.fmax(d_qk[18], d_qk[19], ftz=False)
                    _max_20 = cute.arch.fmax(d_qk[22], d_qk[23], ftz=False)
                    _max_21 = cute.arch.fmax(d_qk[26], d_qk[27], ftz=False)
                    _max_22 = cute.arch.fmax(d_qk[30], d_qk[31], ftz=False)
                    _max_23 = cute.arch.fmax(_max_15, _max_16, ftz=False)
                    _max_24 = cute.arch.fmax(_max_17, _max_18, ftz=False)
                    _max_25 = cute.arch.fmax(_max_19, _max_20, ftz=False)
                    _max_26 = cute.arch.fmax(_max_21, _max_22, ftz=False)
                    _max_27 = cute.arch.fmax(_max_23, _max_24, ftz=False)
                    _max_28 = cute.arch.fmax(_max_25, _max_26, ftz=False)
                    _max_29 = cute.arch.fmax(_max_27, _max_28, ftz=False)
                    _max_30 = cute.arch.fmax(d_qk[32], d_qk[33], ftz=False)
                    _max_31 = cute.arch.fmax(d_qk[36], d_qk[37], ftz=False)
                    _max_32 = cute.arch.fmax(d_qk[40], d_qk[41], ftz=False)
                    _max_33 = cute.arch.fmax(d_qk[44], d_qk[45], ftz=False)
                    _max_34 = cute.arch.fmax(d_qk[48], d_qk[49], ftz=False)
                    _max_35 = cute.arch.fmax(d_qk[52], d_qk[53], ftz=False)
                    _max_36 = cute.arch.fmax(d_qk[56], d_qk[57], ftz=False)
                    _max_37 = cute.arch.fmax(d_qk[60], d_qk[61], ftz=False)
                    _max_38 = cute.arch.fmax(_max_30, _max_31, ftz=False)
                    _max_39 = cute.arch.fmax(_max_32, _max_33, ftz=False)
                    _max_40 = cute.arch.fmax(_max_34, _max_35, ftz=False)
                    _max_41 = cute.arch.fmax(_max_36, _max_37, ftz=False)
                    _max_42 = cute.arch.fmax(_max_38, _max_39, ftz=False)
                    _max_43 = cute.arch.fmax(_max_40, _max_41, ftz=False)
                    _max_44 = cute.arch.fmax(_max_42, _max_43, ftz=False)
                    _max_45 = cute.arch.fmax(d_qk[34], d_qk[35], ftz=False)
                    _max_46 = cute.arch.fmax(d_qk[38], d_qk[39], ftz=False)
                    _max_47 = cute.arch.fmax(d_qk[42], d_qk[43], ftz=False)
                    _max_48 = cute.arch.fmax(d_qk[46], d_qk[47], ftz=False)
                    _max_49 = cute.arch.fmax(d_qk[50], d_qk[51], ftz=False)
                    _max_50 = cute.arch.fmax(d_qk[54], d_qk[55], ftz=False)
                    _max_51 = cute.arch.fmax(d_qk[58], d_qk[59], ftz=False)
                    _max_52 = cute.arch.fmax(d_qk[62], d_qk[63], ftz=False)
                    _max_53 = cute.arch.fmax(_max_45, _max_46, ftz=False)
                    _max_54 = cute.arch.fmax(_max_47, _max_48, ftz=False)
                    _max_55 = cute.arch.fmax(_max_49, _max_50, ftz=False)
                    _max_56 = cute.arch.fmax(_max_51, _max_52, ftz=False)
                    _max_57 = cute.arch.fmax(_max_53, _max_54, ftz=False)
                    _max_58 = cute.arch.fmax(_max_55, _max_56, ftz=False)
                    _max_59 = cute.arch.fmax(_max_57, _max_58, ftz=False)
                    _max_60 = cute.arch.fmax(_max_14, _max_44, ftz=False)
                    m0[0] = cutlass.Float32((_max_60 if (has2_0 != 0) else _max_14))
                    _max_61 = cute.arch.fmax(_max_29, _max_59, ftz=False)
                    m1[0] = cutlass.Float32((_max_61 if (has2_0 != 0) else _max_29))
                    _shfl_xor_0 = cute.arch.shuffle_sync_bfly(m0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _max_62 = cute.arch.fmax(m0[0], _shfl_xor_0, ftz=False)
                    m0[0] = cutlass.Float32(_max_62)
                    _shfl_xor_1 = cute.arch.shuffle_sync_bfly(m0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _max_63 = cute.arch.fmax(m0[0], _shfl_xor_1, ftz=False)
                    m0[0] = cutlass.Float32(_max_63)
                    _shfl_xor_2 = cute.arch.shuffle_sync_bfly(m1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _max_64 = cute.arch.fmax(m1[0], _shfl_xor_2, ftz=False)
                    m1[0] = cutlass.Float32(_max_64)
                    _shfl_xor_3 = cute.arch.shuffle_sync_bfly(m1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _max_65 = cute.arch.fmax(m1[0], _shfl_xor_3, ftz=False)
                    m1[0] = cutlass.Float32(_max_65)
                else:
                    _min_0 = cute.arch.fmin(d_qk[0], d_qk[1], ftz=False)
                    _min_1 = cute.arch.fmin(d_qk[4], d_qk[5], ftz=False)
                    _min_2 = cute.arch.fmin(d_qk[8], d_qk[9], ftz=False)
                    _min_3 = cute.arch.fmin(d_qk[12], d_qk[13], ftz=False)
                    _min_4 = cute.arch.fmin(d_qk[16], d_qk[17], ftz=False)
                    _min_5 = cute.arch.fmin(d_qk[20], d_qk[21], ftz=False)
                    _min_6 = cute.arch.fmin(d_qk[24], d_qk[25], ftz=False)
                    _min_7 = cute.arch.fmin(d_qk[28], d_qk[29], ftz=False)
                    _min_8 = cute.arch.fmin(_min_0, _min_1, ftz=False)
                    _min_9 = cute.arch.fmin(_min_2, _min_3, ftz=False)
                    _min_10 = cute.arch.fmin(_min_4, _min_5, ftz=False)
                    _min_11 = cute.arch.fmin(_min_6, _min_7, ftz=False)
                    _min_12 = cute.arch.fmin(_min_8, _min_9, ftz=False)
                    _min_13 = cute.arch.fmin(_min_10, _min_11, ftz=False)
                    _min_14 = cute.arch.fmin(_min_12, _min_13, ftz=False)
                    _min_15 = cute.arch.fmin(d_qk[2], d_qk[3], ftz=False)
                    _min_16 = cute.arch.fmin(d_qk[6], d_qk[7], ftz=False)
                    _min_17 = cute.arch.fmin(d_qk[10], d_qk[11], ftz=False)
                    _min_18 = cute.arch.fmin(d_qk[14], d_qk[15], ftz=False)
                    _min_19 = cute.arch.fmin(d_qk[18], d_qk[19], ftz=False)
                    _min_20 = cute.arch.fmin(d_qk[22], d_qk[23], ftz=False)
                    _min_21 = cute.arch.fmin(d_qk[26], d_qk[27], ftz=False)
                    _min_22 = cute.arch.fmin(d_qk[30], d_qk[31], ftz=False)
                    _min_23 = cute.arch.fmin(_min_15, _min_16, ftz=False)
                    _min_24 = cute.arch.fmin(_min_17, _min_18, ftz=False)
                    _min_25 = cute.arch.fmin(_min_19, _min_20, ftz=False)
                    _min_26 = cute.arch.fmin(_min_21, _min_22, ftz=False)
                    _min_27 = cute.arch.fmin(_min_23, _min_24, ftz=False)
                    _min_28 = cute.arch.fmin(_min_25, _min_26, ftz=False)
                    _min_29 = cute.arch.fmin(_min_27, _min_28, ftz=False)
                    _min_30 = cute.arch.fmin(d_qk[32], d_qk[33], ftz=False)
                    _min_31 = cute.arch.fmin(d_qk[36], d_qk[37], ftz=False)
                    _min_32 = cute.arch.fmin(d_qk[40], d_qk[41], ftz=False)
                    _min_33 = cute.arch.fmin(d_qk[44], d_qk[45], ftz=False)
                    _min_34 = cute.arch.fmin(d_qk[48], d_qk[49], ftz=False)
                    _min_35 = cute.arch.fmin(d_qk[52], d_qk[53], ftz=False)
                    _min_36 = cute.arch.fmin(d_qk[56], d_qk[57], ftz=False)
                    _min_37 = cute.arch.fmin(d_qk[60], d_qk[61], ftz=False)
                    _min_38 = cute.arch.fmin(_min_30, _min_31, ftz=False)
                    _min_39 = cute.arch.fmin(_min_32, _min_33, ftz=False)
                    _min_40 = cute.arch.fmin(_min_34, _min_35, ftz=False)
                    _min_41 = cute.arch.fmin(_min_36, _min_37, ftz=False)
                    _min_42 = cute.arch.fmin(_min_38, _min_39, ftz=False)
                    _min_43 = cute.arch.fmin(_min_40, _min_41, ftz=False)
                    _min_44 = cute.arch.fmin(_min_42, _min_43, ftz=False)
                    _min_45 = cute.arch.fmin(d_qk[34], d_qk[35], ftz=False)
                    _min_46 = cute.arch.fmin(d_qk[38], d_qk[39], ftz=False)
                    _min_47 = cute.arch.fmin(d_qk[42], d_qk[43], ftz=False)
                    _min_48 = cute.arch.fmin(d_qk[46], d_qk[47], ftz=False)
                    _min_49 = cute.arch.fmin(d_qk[50], d_qk[51], ftz=False)
                    _min_50 = cute.arch.fmin(d_qk[54], d_qk[55], ftz=False)
                    _min_51 = cute.arch.fmin(d_qk[58], d_qk[59], ftz=False)
                    _min_52 = cute.arch.fmin(d_qk[62], d_qk[63], ftz=False)
                    _min_53 = cute.arch.fmin(_min_45, _min_46, ftz=False)
                    _min_54 = cute.arch.fmin(_min_47, _min_48, ftz=False)
                    _min_55 = cute.arch.fmin(_min_49, _min_50, ftz=False)
                    _min_56 = cute.arch.fmin(_min_51, _min_52, ftz=False)
                    _min_57 = cute.arch.fmin(_min_53, _min_54, ftz=False)
                    _min_58 = cute.arch.fmin(_min_55, _min_56, ftz=False)
                    _min_59 = cute.arch.fmin(_min_57, _min_58, ftz=False)
                    _min_60 = cute.arch.fmin(_min_14, _min_44, ftz=False)
                    m0[0] = cutlass.Float32((_min_60 if (has2_0 != 0) else _min_14))
                    _min_61 = cute.arch.fmin(_min_29, _min_59, ftz=False)
                    m1[0] = cutlass.Float32((_min_61 if (has2_0 != 0) else _min_29))
                    _shfl_xor_4 = cute.arch.shuffle_sync_bfly(m0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _min_62 = cute.arch.fmin(m0[0], _shfl_xor_4, ftz=False)
                    m0[0] = cutlass.Float32(_min_62)
                    _shfl_xor_5 = cute.arch.shuffle_sync_bfly(m0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _min_63 = cute.arch.fmin(m0[0], _shfl_xor_5, ftz=False)
                    m0[0] = cutlass.Float32(_min_63)
                    _shfl_xor_6 = cute.arch.shuffle_sync_bfly(m1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _min_64 = cute.arch.fmin(m1[0], _shfl_xor_6, ftz=False)
                    m1[0] = cutlass.Float32(_min_64)
                    _shfl_xor_7 = cute.arch.shuffle_sync_bfly(m1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    _min_65 = cute.arch.fmin(m1[0], _shfl_xor_7, ftz=False)
                    m1[0] = cutlass.Float32(_min_65)
                f_max0[0] = cutlass.Float32(m0[0])
                f_max1[0] = cutlass.Float32(m1[0])
                row_max0[0] = cutlass.Float32((f_max0[0] * scale_log2))
                row_max1[0] = cutlass.Float32((f_max1[0] * scale_log2))
                f_sum0[0] = cutlass.Float32(0.0)
                f_sum1[0] = cutlass.Float32(0.0)
                sa0[0] = cutlass.Float32(0.0)
                sa1[0] = cutlass.Float32(0.0)
                sb0[0] = cutlass.Float32(0.0)
                sb1[0] = cutlass.Float32(0.0)
                _exp2_0 = cute.math.exp2(((d_qk[0] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_1 = cute.math.exp2(((d_qk[1] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[0] = cutlass.Float32(_exp2_0)
                d_qk[1] = cutlass.Float32(_exp2_1)
                sa0[0] += cutlass.Float32((_exp2_0 + _exp2_1))
                _exp2_2 = cute.math.exp2(((d_qk[2] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_3 = cute.math.exp2(((d_qk[3] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[2] = cutlass.Float32(_exp2_2)
                d_qk[3] = cutlass.Float32(_exp2_3)
                sa1[0] += cutlass.Float32((_exp2_2 + _exp2_3))
                _exp2_4 = cute.math.exp2(((d_qk[4] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_5 = cute.math.exp2(((d_qk[5] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[4] = cutlass.Float32(_exp2_4)
                d_qk[5] = cutlass.Float32(_exp2_5)
                sa0[0] += cutlass.Float32((_exp2_4 + _exp2_5))
                _exp2_6 = cute.math.exp2(((d_qk[6] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_7 = cute.math.exp2(((d_qk[7] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[6] = cutlass.Float32(_exp2_6)
                d_qk[7] = cutlass.Float32(_exp2_7)
                sa1[0] += cutlass.Float32((_exp2_6 + _exp2_7))
                _exp2_8 = cute.math.exp2(((d_qk[8] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_9 = cute.math.exp2(((d_qk[9] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[8] = cutlass.Float32(_exp2_8)
                d_qk[9] = cutlass.Float32(_exp2_9)
                sa0[0] += cutlass.Float32((_exp2_8 + _exp2_9))
                _exp2_10 = cute.math.exp2(((d_qk[10] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_11 = cute.math.exp2(((d_qk[11] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[10] = cutlass.Float32(_exp2_10)
                d_qk[11] = cutlass.Float32(_exp2_11)
                sa1[0] += cutlass.Float32((_exp2_10 + _exp2_11))
                _exp2_12 = cute.math.exp2(((d_qk[12] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_13 = cute.math.exp2(((d_qk[13] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[12] = cutlass.Float32(_exp2_12)
                d_qk[13] = cutlass.Float32(_exp2_13)
                sa0[0] += cutlass.Float32((_exp2_12 + _exp2_13))
                _exp2_14 = cute.math.exp2(((d_qk[14] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_15 = cute.math.exp2(((d_qk[15] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[14] = cutlass.Float32(_exp2_14)
                d_qk[15] = cutlass.Float32(_exp2_15)
                sa1[0] += cutlass.Float32((_exp2_14 + _exp2_15))
                _exp2_16 = cute.math.exp2(((d_qk[16] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_17 = cute.math.exp2(((d_qk[17] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[16] = cutlass.Float32(_exp2_16)
                d_qk[17] = cutlass.Float32(_exp2_17)
                sa0[0] += cutlass.Float32((_exp2_16 + _exp2_17))
                _exp2_18 = cute.math.exp2(((d_qk[18] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_19 = cute.math.exp2(((d_qk[19] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[18] = cutlass.Float32(_exp2_18)
                d_qk[19] = cutlass.Float32(_exp2_19)
                sa1[0] += cutlass.Float32((_exp2_18 + _exp2_19))
                _exp2_20 = cute.math.exp2(((d_qk[20] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_21 = cute.math.exp2(((d_qk[21] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[20] = cutlass.Float32(_exp2_20)
                d_qk[21] = cutlass.Float32(_exp2_21)
                sa0[0] += cutlass.Float32((_exp2_20 + _exp2_21))
                _exp2_22 = cute.math.exp2(((d_qk[22] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_23 = cute.math.exp2(((d_qk[23] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[22] = cutlass.Float32(_exp2_22)
                d_qk[23] = cutlass.Float32(_exp2_23)
                sa1[0] += cutlass.Float32((_exp2_22 + _exp2_23))
                _exp2_24 = cute.math.exp2(((d_qk[24] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_25 = cute.math.exp2(((d_qk[25] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[24] = cutlass.Float32(_exp2_24)
                d_qk[25] = cutlass.Float32(_exp2_25)
                sa0[0] += cutlass.Float32((_exp2_24 + _exp2_25))
                _exp2_26 = cute.math.exp2(((d_qk[26] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_27 = cute.math.exp2(((d_qk[27] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[26] = cutlass.Float32(_exp2_26)
                d_qk[27] = cutlass.Float32(_exp2_27)
                sa1[0] += cutlass.Float32((_exp2_26 + _exp2_27))
                _exp2_28 = cute.math.exp2(((d_qk[28] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_29 = cute.math.exp2(((d_qk[29] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[28] = cutlass.Float32(_exp2_28)
                d_qk[29] = cutlass.Float32(_exp2_29)
                sa0[0] += cutlass.Float32((_exp2_28 + _exp2_29))
                _exp2_30 = cute.math.exp2(((d_qk[30] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_31 = cute.math.exp2(((d_qk[31] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[30] = cutlass.Float32(_exp2_30)
                d_qk[31] = cutlass.Float32(_exp2_31)
                sa1[0] += cutlass.Float32((_exp2_30 + _exp2_31))
                _exp2_32 = cute.math.exp2(((d_qk[32] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_33 = cute.math.exp2(((d_qk[33] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[32] = cutlass.Float32(_exp2_32)
                d_qk[33] = cutlass.Float32(_exp2_33)
                sb0[0] += cutlass.Float32((_exp2_32 + _exp2_33))
                _exp2_34 = cute.math.exp2(((d_qk[34] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_35 = cute.math.exp2(((d_qk[35] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[34] = cutlass.Float32(_exp2_34)
                d_qk[35] = cutlass.Float32(_exp2_35)
                sb1[0] += cutlass.Float32((_exp2_34 + _exp2_35))
                _exp2_36 = cute.math.exp2(((d_qk[36] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_37 = cute.math.exp2(((d_qk[37] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[36] = cutlass.Float32(_exp2_36)
                d_qk[37] = cutlass.Float32(_exp2_37)
                sb0[0] += cutlass.Float32((_exp2_36 + _exp2_37))
                _exp2_38 = cute.math.exp2(((d_qk[38] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_39 = cute.math.exp2(((d_qk[39] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[38] = cutlass.Float32(_exp2_38)
                d_qk[39] = cutlass.Float32(_exp2_39)
                sb1[0] += cutlass.Float32((_exp2_38 + _exp2_39))
                _exp2_40 = cute.math.exp2(((d_qk[40] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_41 = cute.math.exp2(((d_qk[41] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[40] = cutlass.Float32(_exp2_40)
                d_qk[41] = cutlass.Float32(_exp2_41)
                sb0[0] += cutlass.Float32((_exp2_40 + _exp2_41))
                _exp2_42 = cute.math.exp2(((d_qk[42] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_43 = cute.math.exp2(((d_qk[43] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[42] = cutlass.Float32(_exp2_42)
                d_qk[43] = cutlass.Float32(_exp2_43)
                sb1[0] += cutlass.Float32((_exp2_42 + _exp2_43))
                _exp2_44 = cute.math.exp2(((d_qk[44] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_45 = cute.math.exp2(((d_qk[45] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[44] = cutlass.Float32(_exp2_44)
                d_qk[45] = cutlass.Float32(_exp2_45)
                sb0[0] += cutlass.Float32((_exp2_44 + _exp2_45))
                _exp2_46 = cute.math.exp2(((d_qk[46] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_47 = cute.math.exp2(((d_qk[47] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[46] = cutlass.Float32(_exp2_46)
                d_qk[47] = cutlass.Float32(_exp2_47)
                sb1[0] += cutlass.Float32((_exp2_46 + _exp2_47))
                _exp2_48 = cute.math.exp2(((d_qk[48] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_49 = cute.math.exp2(((d_qk[49] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[48] = cutlass.Float32(_exp2_48)
                d_qk[49] = cutlass.Float32(_exp2_49)
                sb0[0] += cutlass.Float32((_exp2_48 + _exp2_49))
                _exp2_50 = cute.math.exp2(((d_qk[50] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_51 = cute.math.exp2(((d_qk[51] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[50] = cutlass.Float32(_exp2_50)
                d_qk[51] = cutlass.Float32(_exp2_51)
                sb1[0] += cutlass.Float32((_exp2_50 + _exp2_51))
                _exp2_52 = cute.math.exp2(((d_qk[52] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_53 = cute.math.exp2(((d_qk[53] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[52] = cutlass.Float32(_exp2_52)
                d_qk[53] = cutlass.Float32(_exp2_53)
                sb0[0] += cutlass.Float32((_exp2_52 + _exp2_53))
                _exp2_54 = cute.math.exp2(((d_qk[54] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_55 = cute.math.exp2(((d_qk[55] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[54] = cutlass.Float32(_exp2_54)
                d_qk[55] = cutlass.Float32(_exp2_55)
                sb1[0] += cutlass.Float32((_exp2_54 + _exp2_55))
                _exp2_56 = cute.math.exp2(((d_qk[56] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_57 = cute.math.exp2(((d_qk[57] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[56] = cutlass.Float32(_exp2_56)
                d_qk[57] = cutlass.Float32(_exp2_57)
                sb0[0] += cutlass.Float32((_exp2_56 + _exp2_57))
                _exp2_58 = cute.math.exp2(((d_qk[58] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_59 = cute.math.exp2(((d_qk[59] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[58] = cutlass.Float32(_exp2_58)
                d_qk[59] = cutlass.Float32(_exp2_59)
                sb1[0] += cutlass.Float32((_exp2_58 + _exp2_59))
                _exp2_60 = cute.math.exp2(((d_qk[60] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                _exp2_61 = cute.math.exp2(((d_qk[61] * scale_log2) - row_max0[0]), approx=True, ftz=True)
                d_qk[60] = cutlass.Float32(_exp2_60)
                d_qk[61] = cutlass.Float32(_exp2_61)
                sb0[0] += cutlass.Float32((_exp2_60 + _exp2_61))
                _exp2_62 = cute.math.exp2(((d_qk[62] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                _exp2_63 = cute.math.exp2(((d_qk[63] * scale_log2) - row_max1[0]), approx=True, ftz=True)
                d_qk[62] = cutlass.Float32(_exp2_62)
                d_qk[63] = cutlass.Float32(_exp2_63)
                sb1[0] += cutlass.Float32((_exp2_62 + _exp2_63))
                f_sum0[0] += cutlass.Float32((sa0[0] + (sb0[0] if (has2_0 != 0) else 0.0)))
                f_sum1[0] += cutlass.Float32((sa1[0] + (sb1[0] if (has2_0 != 0) else 0.0)))
                _shfl_xor_8 = cute.arch.shuffle_sync_bfly(f_sum0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                f_sum0[0] += cutlass.Float32(_shfl_xor_8)
                _shfl_xor_9 = cute.arch.shuffle_sync_bfly(f_sum0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                f_sum0[0] += cutlass.Float32(_shfl_xor_9)
                _shfl_xor_10 = cute.arch.shuffle_sync_bfly(f_sum1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                f_sum1[0] += cutlass.Float32(_shfl_xor_10)
                _shfl_xor_11 = cute.arch.shuffle_sync_bfly(f_sum1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                f_sum1[0] += cutlass.Float32(_shfl_xor_11)
                row_sum0[0] = cutlass.Float32(f_sum0[0])
                row_sum1[0] = cutlass.Float32(f_sum1[0])
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
                _bf16x2_16 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[32]), cutlass.Float32(d_qk[33])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[32]), cutlass.Float32(d_qk[33])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[16] = cutlass.Uint32(_bf16x2_16)
                _bf16x2_17 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[34]), cutlass.Float32(d_qk[35])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[34]), cutlass.Float32(d_qk[35])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[17] = cutlass.Uint32(_bf16x2_17)
                _bf16x2_18 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[36]), cutlass.Float32(d_qk[37])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[36]), cutlass.Float32(d_qk[37])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[18] = cutlass.Uint32(_bf16x2_18)
                _bf16x2_19 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[38]), cutlass.Float32(d_qk[39])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[38]), cutlass.Float32(d_qk[39])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[19] = cutlass.Uint32(_bf16x2_19)
                _bf16x2_20 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[40]), cutlass.Float32(d_qk[41])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[40]), cutlass.Float32(d_qk[41])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[20] = cutlass.Uint32(_bf16x2_20)
                _bf16x2_21 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[42]), cutlass.Float32(d_qk[43])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[42]), cutlass.Float32(d_qk[43])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[21] = cutlass.Uint32(_bf16x2_21)
                _bf16x2_22 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[44]), cutlass.Float32(d_qk[45])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[44]), cutlass.Float32(d_qk[45])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[22] = cutlass.Uint32(_bf16x2_22)
                _bf16x2_23 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[46]), cutlass.Float32(d_qk[47])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[46]), cutlass.Float32(d_qk[47])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[23] = cutlass.Uint32(_bf16x2_23)
                _bf16x2_24 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[48]), cutlass.Float32(d_qk[49])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[48]), cutlass.Float32(d_qk[49])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[24] = cutlass.Uint32(_bf16x2_24)
                _bf16x2_25 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[50]), cutlass.Float32(d_qk[51])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[50]), cutlass.Float32(d_qk[51])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[25] = cutlass.Uint32(_bf16x2_25)
                _bf16x2_26 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[52]), cutlass.Float32(d_qk[53])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[52]), cutlass.Float32(d_qk[53])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[26] = cutlass.Uint32(_bf16x2_26)
                _bf16x2_27 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[54]), cutlass.Float32(d_qk[55])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[54]), cutlass.Float32(d_qk[55])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[27] = cutlass.Uint32(_bf16x2_27)
                _bf16x2_28 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[56]), cutlass.Float32(d_qk[57])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[56]), cutlass.Float32(d_qk[57])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[28] = cutlass.Uint32(_bf16x2_28)
                _bf16x2_29 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[58]), cutlass.Float32(d_qk[59])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[58]), cutlass.Float32(d_qk[59])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[29] = cutlass.Uint32(_bf16x2_29)
                _bf16x2_30 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[60]), cutlass.Float32(d_qk[61])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[60]), cutlass.Float32(d_qk[61])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[30] = cutlass.Uint32(_bf16x2_30)
                _bf16x2_31 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[62]), cutlass.Float32(d_qk[63])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[62]), cutlass.Float32(d_qk[63])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                p_bf16[31] = cutlass.Uint32(_bf16x2_31)
                if (has2_0 == 0):
                    p_bf16[16] = cutlass.Uint32(0)
                    p_bf16[17] = cutlass.Uint32(0)
                    p_bf16[18] = cutlass.Uint32(0)
                    p_bf16[19] = cutlass.Uint32(0)
                    p_bf16[20] = cutlass.Uint32(0)
                    p_bf16[21] = cutlass.Uint32(0)
                    p_bf16[22] = cutlass.Uint32(0)
                    p_bf16[23] = cutlass.Uint32(0)
                    p_bf16[24] = cutlass.Uint32(0)
                    p_bf16[25] = cutlass.Uint32(0)
                    p_bf16[26] = cutlass.Uint32(0)
                    p_bf16[27] = cutlass.Uint32(0)
                    p_bf16[28] = cutlass.Uint32(0)
                    p_bf16[29] = cutlass.Uint32(0)
                    p_bf16[30] = cutlass.Uint32(0)
                    p_bf16[31] = cutlass.Uint32(0)
                n_loop = cutlass.Int32((n_own - 1))
                for n in cutlass.range(cutlass.Int32(0), cutlass.Int32(n_loop), cutlass.Int32(1), unroll=1, unroll_full=False):
                    cur_stage = cutlass.Int32((cur_pos[0] % 3))
                    cur_ph = cutlass.Int32((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(cur_pos[0]).ir_value(), cutlass.Int32(3).ir_value())) & 1))
                    w_e_0 = cutlass.Int32(_meta_smem[(own_base_w + ((n + 1) >> 1))])
                    en = cutlass.Int32(((w_e_0 >> (((n + 1) & 1) * 16)) & 65535))
                    nxt_pos = cutlass.Int32((gbase[0] + (en >> 3)))
                    nxt_has2 = cutlass.Int32(((en >> 2) & 1))
                    for p_2 in cutlass.range(cutlass.Int32((cur_pos[0] + 1)), cutlass.Int32(nxt_pos), cutlass.Int32(1), unroll=1, unroll_full=False):
                        while not prims.mbarrier_wait_parity(k_full_addr + (p_2 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_2).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        if prims.elect_sync():
                            cute.arch.mbarrier_arrive(k_empty_addr + (p_2 % 3))
                        while not prims.mbarrier_wait_parity(v_full_addr + (p_2 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_2).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        if prims.elect_sync():
                            cute.arch.mbarrier_arrive(v_empty_addr + (p_2 % 3))
                    nxt_stage = cutlass.Int32((nxt_pos % 3))
                    while not prims.mbarrier_wait_parity(k_full_addr + nxt_stage, (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(nxt_pos).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                        pass
                    while not prims.mbarrier_wait_parity(v_full_addr + cur_stage, cur_ph, prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                        pass
                    cute.nvgpu.warpgroup.fence()
                    _wgmma_b_0_2_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + cutlass.Uint32((nxt_stage * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                    _wgmma_b_0_2 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_2_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_2_raw)))
                    _wgmma_22_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_22 = cutlass_llvm.inline_asm(
                        _wgmma_22_ty,
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64(_wgmma_b_0_2).ir_value(),
                            cutlass.Uint32(q_frag[(0) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(0) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(0) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(0) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 0, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[0]))
                    d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[1]))
                    d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[2]))
                    d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[3]))
                    d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[4]))
                    d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[5]))
                    d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[6]))
                    d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[7]))
                    d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[8]))
                    d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[9]))
                    d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[10]))
                    d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[11]))
                    d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[12]))
                    d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[13]))
                    d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[14]))
                    d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[15]))
                    d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[16]))
                    d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[17]))
                    d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[18]))
                    d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[19]))
                    d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[20]))
                    d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[21]))
                    d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[22]))
                    d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[23]))
                    d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[24]))
                    d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[25]))
                    d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[26]))
                    d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[27]))
                    d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[28]))
                    d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[29]))
                    d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[30]))
                    d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[31]))
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_22, position=[63]))
                    _wgmma_23_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_23 = cutlass_llvm.inline_asm(
                        _wgmma_23_ty,
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_2 + 2)).ir_value(),
                            cutlass.Uint32(q_frag[(4) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(4) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(4) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(4) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_qk[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[0]))
                    d_qk[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[1]))
                    d_qk[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[2]))
                    d_qk[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[3]))
                    d_qk[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[4]))
                    d_qk[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[5]))
                    d_qk[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[6]))
                    d_qk[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[7]))
                    d_qk[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[8]))
                    d_qk[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[9]))
                    d_qk[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[10]))
                    d_qk[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[11]))
                    d_qk[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[12]))
                    d_qk[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[13]))
                    d_qk[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[14]))
                    d_qk[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[15]))
                    d_qk[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[16]))
                    d_qk[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[17]))
                    d_qk[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[18]))
                    d_qk[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[19]))
                    d_qk[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[20]))
                    d_qk[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[21]))
                    d_qk[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[22]))
                    d_qk[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[23]))
                    d_qk[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[24]))
                    d_qk[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[25]))
                    d_qk[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[26]))
                    d_qk[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[27]))
                    d_qk[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[28]))
                    d_qk[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[29]))
                    d_qk[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[30]))
                    d_qk[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[31]))
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_23, position=[63]))
                    _wgmma_24_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_2 + 4)).ir_value(),
                            cutlass.Uint32(q_frag[(8) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(8) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(8) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(8) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_24, position=[63]))
                    _wgmma_25_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_2 + 6)).ir_value(),
                            cutlass.Uint32(q_frag[(12) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(12) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(12) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(12) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_25, position=[63]))
                    _wgmma_b_0_3_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + cutlass.Uint32((nxt_stage * 32768))) + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                    _wgmma_b_0_3 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw)))
                    _wgmma_26_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64(_wgmma_b_0_3).ir_value(),
                            cutlass.Uint32(q_frag[(16) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(16) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(16) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(16) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_26, position=[63]))
                    _wgmma_27_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_3 + 2)).ir_value(),
                            cutlass.Uint32(q_frag[(20) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(20) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(20) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(20) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_27, position=[63]))
                    _wgmma_28_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_3 + 4)).ir_value(),
                            cutlass.Uint32(q_frag[(24) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(24) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(24) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(24) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_28, position=[63]))
                    _wgmma_29_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
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
                            cutlass.Float32(d_qk[(0) + 32]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 33]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 34]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 35]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 36]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 37]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 38]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 39]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 40]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 41]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 42]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 43]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 44]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 45]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 46]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 47]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 48]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 49]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 50]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 51]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 52]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 53]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 54]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 55]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 56]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 57]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 58]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 59]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 60]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 61]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 62]).ir_value(),
                            cutlass.Float32(d_qk[(0) + 63]).ir_value(),
                            cutlass.Uint64((_wgmma_b_0_3 + 6)).ir_value(),
                            cutlass.Uint32(q_frag[(28) + 0]).ir_value(),
                            cutlass.Uint32(q_frag[(28) + 1]).ir_value(),
                            cutlass.Uint32(q_frag[(28) + 2]).ir_value(),
                            cutlass.Uint32(q_frag[(28) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 0;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    d_qk[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[32]))
                    d_qk[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[33]))
                    d_qk[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[34]))
                    d_qk[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[35]))
                    d_qk[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[36]))
                    d_qk[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[37]))
                    d_qk[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[38]))
                    d_qk[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[39]))
                    d_qk[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[40]))
                    d_qk[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[41]))
                    d_qk[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[42]))
                    d_qk[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[43]))
                    d_qk[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[44]))
                    d_qk[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[45]))
                    d_qk[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[46]))
                    d_qk[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[47]))
                    d_qk[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[48]))
                    d_qk[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[49]))
                    d_qk[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[50]))
                    d_qk[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[51]))
                    d_qk[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[52]))
                    d_qk[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[53]))
                    d_qk[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[54]))
                    d_qk[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[55]))
                    d_qk[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[56]))
                    d_qk[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[57]))
                    d_qk[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[58]))
                    d_qk[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[59]))
                    d_qk[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[60]))
                    d_qk[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[61]))
                    d_qk[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[62]))
                    d_qk[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_29, position=[63]))
                    cute.nvgpu.warpgroup.commit_group()
                    _wgmma_b_0_4_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_a_addr + cutlass.Uint32((cur_stage * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                    _wgmma_b_0_4 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw)))
                    _wgmma_30_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_30 = cutlass_llvm.inline_asm(
                        _wgmma_30_ty,
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
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[0]))
                    d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[1]))
                    d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[2]))
                    d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[3]))
                    d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[4]))
                    d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[5]))
                    d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[6]))
                    d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[7]))
                    d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[8]))
                    d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[9]))
                    d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[10]))
                    d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[11]))
                    d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[12]))
                    d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[13]))
                    d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[14]))
                    d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[15]))
                    d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[16]))
                    d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[17]))
                    d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[18]))
                    d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[19]))
                    d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[20]))
                    d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[21]))
                    d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[22]))
                    d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[23]))
                    d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[24]))
                    d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[25]))
                    d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[26]))
                    d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[27]))
                    d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[28]))
                    d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[29]))
                    d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[30]))
                    d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[31]))
                    d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[32]))
                    d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[33]))
                    d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[34]))
                    d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[35]))
                    d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[36]))
                    d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[37]))
                    d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[38]))
                    d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[39]))
                    d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[40]))
                    d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[41]))
                    d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[42]))
                    d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[43]))
                    d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[44]))
                    d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[45]))
                    d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[46]))
                    d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[47]))
                    d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[48]))
                    d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[49]))
                    d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[50]))
                    d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[51]))
                    d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[52]))
                    d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[53]))
                    d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[54]))
                    d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[55]))
                    d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[56]))
                    d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[57]))
                    d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[58]))
                    d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[59]))
                    d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[60]))
                    d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[61]))
                    d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[62]))
                    d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_30, position=[63]))
                    _wgmma_31_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_31 = cutlass_llvm.inline_asm(
                        _wgmma_31_ty,
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
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[0]))
                    d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[1]))
                    d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[2]))
                    d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[3]))
                    d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[4]))
                    d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[5]))
                    d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[6]))
                    d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[7]))
                    d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[8]))
                    d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[9]))
                    d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[10]))
                    d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[11]))
                    d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[12]))
                    d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[13]))
                    d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[14]))
                    d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[15]))
                    d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[16]))
                    d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[17]))
                    d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[18]))
                    d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[19]))
                    d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[20]))
                    d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[21]))
                    d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[22]))
                    d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[23]))
                    d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[24]))
                    d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[25]))
                    d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[26]))
                    d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[27]))
                    d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[28]))
                    d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[29]))
                    d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[30]))
                    d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[31]))
                    d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[32]))
                    d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[33]))
                    d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[34]))
                    d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[35]))
                    d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[36]))
                    d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[37]))
                    d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[38]))
                    d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[39]))
                    d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[40]))
                    d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[41]))
                    d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[42]))
                    d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[43]))
                    d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[44]))
                    d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[45]))
                    d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[46]))
                    d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[47]))
                    d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[48]))
                    d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[49]))
                    d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[50]))
                    d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[51]))
                    d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[52]))
                    d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[53]))
                    d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[54]))
                    d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[55]))
                    d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[56]))
                    d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[57]))
                    d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[58]))
                    d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[59]))
                    d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[60]))
                    d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[61]))
                    d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[62]))
                    d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_31, position=[63]))
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
                            cutlass.Uint64((_wgmma_b_0_4 + 256)).ir_value(),
                            cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                            cutlass.Uint64((_wgmma_b_0_4 + 384)).ir_value(),
                            cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    _wgmma_b_0_5_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_b_addr + cutlass.Uint32((cur_stage * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                    _wgmma_b_0_5 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw)))
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
                            cutlass.Uint64(_wgmma_b_0_5).ir_value(),
                            cutlass.Uint32(p_bf16[(16) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(16) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(16) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(16) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                            cutlass.Uint64((_wgmma_b_0_5 + 128)).ir_value(),
                            cutlass.Uint32(p_bf16[(20) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(20) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(20) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(20) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                    _wgmma_36_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_36 = cutlass_llvm.inline_asm(
                        _wgmma_36_ty,
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
                            cutlass.Uint64((_wgmma_b_0_5 + 256)).ir_value(),
                            cutlass.Uint32(p_bf16[(24) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(24) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(24) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(24) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[0]))
                    d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[1]))
                    d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[2]))
                    d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[3]))
                    d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[4]))
                    d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[5]))
                    d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[6]))
                    d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[7]))
                    d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[8]))
                    d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[9]))
                    d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[10]))
                    d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[11]))
                    d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[12]))
                    d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[13]))
                    d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[14]))
                    d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[15]))
                    d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[16]))
                    d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[17]))
                    d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[18]))
                    d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[19]))
                    d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[20]))
                    d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[21]))
                    d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[22]))
                    d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[23]))
                    d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[24]))
                    d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[25]))
                    d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[26]))
                    d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[27]))
                    d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[28]))
                    d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[29]))
                    d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[30]))
                    d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[31]))
                    d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[32]))
                    d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[33]))
                    d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[34]))
                    d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[35]))
                    d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[36]))
                    d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[37]))
                    d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[38]))
                    d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[39]))
                    d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[40]))
                    d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[41]))
                    d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[42]))
                    d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[43]))
                    d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[44]))
                    d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[45]))
                    d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[46]))
                    d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[47]))
                    d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[48]))
                    d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[49]))
                    d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[50]))
                    d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[51]))
                    d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[52]))
                    d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[53]))
                    d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[54]))
                    d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[55]))
                    d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[56]))
                    d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[57]))
                    d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[58]))
                    d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[59]))
                    d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[60]))
                    d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[61]))
                    d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[62]))
                    d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_36, position=[63]))
                    _wgmma_37_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                    _wgmma_37 = cutlass_llvm.inline_asm(
                        _wgmma_37_ty,
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
                            cutlass.Uint64((_wgmma_b_0_5 + 384)).ir_value(),
                            cutlass.Uint32(p_bf16[(28) + 0]).ir_value(),
                            cutlass.Uint32(p_bf16[(28) + 1]).ir_value(),
                            cutlass.Uint32(p_bf16[(28) + 2]).ir_value(),
                            cutlass.Uint32(p_bf16[(28) + 3]).ir_value(),
                        ],
                        asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                        constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[0]))
                    d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[1]))
                    d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[2]))
                    d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[3]))
                    d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[4]))
                    d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[5]))
                    d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[6]))
                    d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[7]))
                    d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[8]))
                    d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[9]))
                    d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[10]))
                    d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[11]))
                    d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[12]))
                    d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[13]))
                    d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[14]))
                    d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[15]))
                    d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[16]))
                    d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[17]))
                    d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[18]))
                    d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[19]))
                    d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[20]))
                    d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[21]))
                    d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[22]))
                    d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[23]))
                    d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[24]))
                    d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[25]))
                    d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[26]))
                    d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[27]))
                    d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[28]))
                    d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[29]))
                    d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[30]))
                    d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[31]))
                    d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[32]))
                    d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[33]))
                    d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[34]))
                    d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[35]))
                    d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[36]))
                    d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[37]))
                    d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[38]))
                    d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[39]))
                    d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[40]))
                    d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[41]))
                    d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[42]))
                    d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[43]))
                    d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[44]))
                    d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[45]))
                    d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[46]))
                    d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[47]))
                    d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[48]))
                    d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[49]))
                    d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[50]))
                    d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[51]))
                    d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[52]))
                    d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[53]))
                    d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[54]))
                    d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[55]))
                    d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[56]))
                    d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[57]))
                    d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[58]))
                    d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[59]))
                    d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[60]))
                    d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[61]))
                    d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[62]))
                    d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_37, position=[63]))
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                    if prims.elect_sync():
                        cute.arch.mbarrier_arrive(k_empty_addr + (nxt_pos % 3))
                    new_max0[0] = cutlass.Float32((0 - float("inf")))
                    new_max1[0] = cutlass.Float32((0 - float("inf")))
                    m0_0[0] = cutlass.Float32((0 - float("inf")))
                    m1_1[0] = cutlass.Float32((0 - float("inf")))
                    if (scale_log2 >= 0.0):
                        _max_66 = cute.arch.fmax(d_qk[0], d_qk[1], ftz=False)
                        _max_67 = cute.arch.fmax(d_qk[4], d_qk[5], ftz=False)
                        _max_68 = cute.arch.fmax(d_qk[8], d_qk[9], ftz=False)
                        _max_69 = cute.arch.fmax(d_qk[12], d_qk[13], ftz=False)
                        _max_70 = cute.arch.fmax(d_qk[16], d_qk[17], ftz=False)
                        _max_71 = cute.arch.fmax(d_qk[20], d_qk[21], ftz=False)
                        _max_72 = cute.arch.fmax(d_qk[24], d_qk[25], ftz=False)
                        _max_73 = cute.arch.fmax(d_qk[28], d_qk[29], ftz=False)
                        _max_74 = cute.arch.fmax(_max_66, _max_67, ftz=False)
                        _max_75 = cute.arch.fmax(_max_68, _max_69, ftz=False)
                        _max_76 = cute.arch.fmax(_max_70, _max_71, ftz=False)
                        _max_77 = cute.arch.fmax(_max_72, _max_73, ftz=False)
                        _max_78 = cute.arch.fmax(_max_74, _max_75, ftz=False)
                        _max_79 = cute.arch.fmax(_max_76, _max_77, ftz=False)
                        _max_80 = cute.arch.fmax(_max_78, _max_79, ftz=False)
                        _max_81 = cute.arch.fmax(d_qk[2], d_qk[3], ftz=False)
                        _max_82 = cute.arch.fmax(d_qk[6], d_qk[7], ftz=False)
                        _max_83 = cute.arch.fmax(d_qk[10], d_qk[11], ftz=False)
                        _max_84 = cute.arch.fmax(d_qk[14], d_qk[15], ftz=False)
                        _max_85 = cute.arch.fmax(d_qk[18], d_qk[19], ftz=False)
                        _max_86 = cute.arch.fmax(d_qk[22], d_qk[23], ftz=False)
                        _max_87 = cute.arch.fmax(d_qk[26], d_qk[27], ftz=False)
                        _max_88 = cute.arch.fmax(d_qk[30], d_qk[31], ftz=False)
                        _max_89 = cute.arch.fmax(_max_81, _max_82, ftz=False)
                        _max_90 = cute.arch.fmax(_max_83, _max_84, ftz=False)
                        _max_91 = cute.arch.fmax(_max_85, _max_86, ftz=False)
                        _max_92 = cute.arch.fmax(_max_87, _max_88, ftz=False)
                        _max_93 = cute.arch.fmax(_max_89, _max_90, ftz=False)
                        _max_94 = cute.arch.fmax(_max_91, _max_92, ftz=False)
                        _max_95 = cute.arch.fmax(_max_93, _max_94, ftz=False)
                        _max_96 = cute.arch.fmax(d_qk[32], d_qk[33], ftz=False)
                        _max_97 = cute.arch.fmax(d_qk[36], d_qk[37], ftz=False)
                        _max_98 = cute.arch.fmax(d_qk[40], d_qk[41], ftz=False)
                        _max_99 = cute.arch.fmax(d_qk[44], d_qk[45], ftz=False)
                        _max_100 = cute.arch.fmax(d_qk[48], d_qk[49], ftz=False)
                        _max_101 = cute.arch.fmax(d_qk[52], d_qk[53], ftz=False)
                        _max_102 = cute.arch.fmax(d_qk[56], d_qk[57], ftz=False)
                        _max_103 = cute.arch.fmax(d_qk[60], d_qk[61], ftz=False)
                        _max_104 = cute.arch.fmax(_max_96, _max_97, ftz=False)
                        _max_105 = cute.arch.fmax(_max_98, _max_99, ftz=False)
                        _max_106 = cute.arch.fmax(_max_100, _max_101, ftz=False)
                        _max_107 = cute.arch.fmax(_max_102, _max_103, ftz=False)
                        _max_108 = cute.arch.fmax(_max_104, _max_105, ftz=False)
                        _max_109 = cute.arch.fmax(_max_106, _max_107, ftz=False)
                        _max_110 = cute.arch.fmax(_max_108, _max_109, ftz=False)
                        _max_111 = cute.arch.fmax(d_qk[34], d_qk[35], ftz=False)
                        _max_112 = cute.arch.fmax(d_qk[38], d_qk[39], ftz=False)
                        _max_113 = cute.arch.fmax(d_qk[42], d_qk[43], ftz=False)
                        _max_114 = cute.arch.fmax(d_qk[46], d_qk[47], ftz=False)
                        _max_115 = cute.arch.fmax(d_qk[50], d_qk[51], ftz=False)
                        _max_116 = cute.arch.fmax(d_qk[54], d_qk[55], ftz=False)
                        _max_117 = cute.arch.fmax(d_qk[58], d_qk[59], ftz=False)
                        _max_118 = cute.arch.fmax(d_qk[62], d_qk[63], ftz=False)
                        _max_119 = cute.arch.fmax(_max_111, _max_112, ftz=False)
                        _max_120 = cute.arch.fmax(_max_113, _max_114, ftz=False)
                        _max_121 = cute.arch.fmax(_max_115, _max_116, ftz=False)
                        _max_122 = cute.arch.fmax(_max_117, _max_118, ftz=False)
                        _max_123 = cute.arch.fmax(_max_119, _max_120, ftz=False)
                        _max_124 = cute.arch.fmax(_max_121, _max_122, ftz=False)
                        _max_125 = cute.arch.fmax(_max_123, _max_124, ftz=False)
                        _max_126 = cute.arch.fmax(_max_80, _max_110, ftz=False)
                        m0_0[0] = cutlass.Float32((_max_126 if (nxt_has2 != 0) else _max_80))
                        _max_127 = cute.arch.fmax(_max_95, _max_125, ftz=False)
                        m1_1[0] = cutlass.Float32((_max_127 if (nxt_has2 != 0) else _max_95))
                        _shfl_xor_12 = cute.arch.shuffle_sync_bfly(m0_0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _max_128 = cute.arch.fmax(m0_0[0], _shfl_xor_12, ftz=False)
                        m0_0[0] = cutlass.Float32(_max_128)
                        _shfl_xor_13 = cute.arch.shuffle_sync_bfly(m0_0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _max_129 = cute.arch.fmax(m0_0[0], _shfl_xor_13, ftz=False)
                        m0_0[0] = cutlass.Float32(_max_129)
                        _shfl_xor_14 = cute.arch.shuffle_sync_bfly(m1_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _max_130 = cute.arch.fmax(m1_1[0], _shfl_xor_14, ftz=False)
                        m1_1[0] = cutlass.Float32(_max_130)
                        _shfl_xor_15 = cute.arch.shuffle_sync_bfly(m1_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _max_131 = cute.arch.fmax(m1_1[0], _shfl_xor_15, ftz=False)
                        m1_1[0] = cutlass.Float32(_max_131)
                    else:
                        _min_66 = cute.arch.fmin(d_qk[0], d_qk[1], ftz=False)
                        _min_67 = cute.arch.fmin(d_qk[4], d_qk[5], ftz=False)
                        _min_68 = cute.arch.fmin(d_qk[8], d_qk[9], ftz=False)
                        _min_69 = cute.arch.fmin(d_qk[12], d_qk[13], ftz=False)
                        _min_70 = cute.arch.fmin(d_qk[16], d_qk[17], ftz=False)
                        _min_71 = cute.arch.fmin(d_qk[20], d_qk[21], ftz=False)
                        _min_72 = cute.arch.fmin(d_qk[24], d_qk[25], ftz=False)
                        _min_73 = cute.arch.fmin(d_qk[28], d_qk[29], ftz=False)
                        _min_74 = cute.arch.fmin(_min_66, _min_67, ftz=False)
                        _min_75 = cute.arch.fmin(_min_68, _min_69, ftz=False)
                        _min_76 = cute.arch.fmin(_min_70, _min_71, ftz=False)
                        _min_77 = cute.arch.fmin(_min_72, _min_73, ftz=False)
                        _min_78 = cute.arch.fmin(_min_74, _min_75, ftz=False)
                        _min_79 = cute.arch.fmin(_min_76, _min_77, ftz=False)
                        _min_80 = cute.arch.fmin(_min_78, _min_79, ftz=False)
                        _min_81 = cute.arch.fmin(d_qk[2], d_qk[3], ftz=False)
                        _min_82 = cute.arch.fmin(d_qk[6], d_qk[7], ftz=False)
                        _min_83 = cute.arch.fmin(d_qk[10], d_qk[11], ftz=False)
                        _min_84 = cute.arch.fmin(d_qk[14], d_qk[15], ftz=False)
                        _min_85 = cute.arch.fmin(d_qk[18], d_qk[19], ftz=False)
                        _min_86 = cute.arch.fmin(d_qk[22], d_qk[23], ftz=False)
                        _min_87 = cute.arch.fmin(d_qk[26], d_qk[27], ftz=False)
                        _min_88 = cute.arch.fmin(d_qk[30], d_qk[31], ftz=False)
                        _min_89 = cute.arch.fmin(_min_81, _min_82, ftz=False)
                        _min_90 = cute.arch.fmin(_min_83, _min_84, ftz=False)
                        _min_91 = cute.arch.fmin(_min_85, _min_86, ftz=False)
                        _min_92 = cute.arch.fmin(_min_87, _min_88, ftz=False)
                        _min_93 = cute.arch.fmin(_min_89, _min_90, ftz=False)
                        _min_94 = cute.arch.fmin(_min_91, _min_92, ftz=False)
                        _min_95 = cute.arch.fmin(_min_93, _min_94, ftz=False)
                        _min_96 = cute.arch.fmin(d_qk[32], d_qk[33], ftz=False)
                        _min_97 = cute.arch.fmin(d_qk[36], d_qk[37], ftz=False)
                        _min_98 = cute.arch.fmin(d_qk[40], d_qk[41], ftz=False)
                        _min_99 = cute.arch.fmin(d_qk[44], d_qk[45], ftz=False)
                        _min_100 = cute.arch.fmin(d_qk[48], d_qk[49], ftz=False)
                        _min_101 = cute.arch.fmin(d_qk[52], d_qk[53], ftz=False)
                        _min_102 = cute.arch.fmin(d_qk[56], d_qk[57], ftz=False)
                        _min_103 = cute.arch.fmin(d_qk[60], d_qk[61], ftz=False)
                        _min_104 = cute.arch.fmin(_min_96, _min_97, ftz=False)
                        _min_105 = cute.arch.fmin(_min_98, _min_99, ftz=False)
                        _min_106 = cute.arch.fmin(_min_100, _min_101, ftz=False)
                        _min_107 = cute.arch.fmin(_min_102, _min_103, ftz=False)
                        _min_108 = cute.arch.fmin(_min_104, _min_105, ftz=False)
                        _min_109 = cute.arch.fmin(_min_106, _min_107, ftz=False)
                        _min_110 = cute.arch.fmin(_min_108, _min_109, ftz=False)
                        _min_111 = cute.arch.fmin(d_qk[34], d_qk[35], ftz=False)
                        _min_112 = cute.arch.fmin(d_qk[38], d_qk[39], ftz=False)
                        _min_113 = cute.arch.fmin(d_qk[42], d_qk[43], ftz=False)
                        _min_114 = cute.arch.fmin(d_qk[46], d_qk[47], ftz=False)
                        _min_115 = cute.arch.fmin(d_qk[50], d_qk[51], ftz=False)
                        _min_116 = cute.arch.fmin(d_qk[54], d_qk[55], ftz=False)
                        _min_117 = cute.arch.fmin(d_qk[58], d_qk[59], ftz=False)
                        _min_118 = cute.arch.fmin(d_qk[62], d_qk[63], ftz=False)
                        _min_119 = cute.arch.fmin(_min_111, _min_112, ftz=False)
                        _min_120 = cute.arch.fmin(_min_113, _min_114, ftz=False)
                        _min_121 = cute.arch.fmin(_min_115, _min_116, ftz=False)
                        _min_122 = cute.arch.fmin(_min_117, _min_118, ftz=False)
                        _min_123 = cute.arch.fmin(_min_119, _min_120, ftz=False)
                        _min_124 = cute.arch.fmin(_min_121, _min_122, ftz=False)
                        _min_125 = cute.arch.fmin(_min_123, _min_124, ftz=False)
                        _min_126 = cute.arch.fmin(_min_80, _min_110, ftz=False)
                        m0_0[0] = cutlass.Float32((_min_126 if (nxt_has2 != 0) else _min_80))
                        _min_127 = cute.arch.fmin(_min_95, _min_125, ftz=False)
                        m1_1[0] = cutlass.Float32((_min_127 if (nxt_has2 != 0) else _min_95))
                        _shfl_xor_16 = cute.arch.shuffle_sync_bfly(m0_0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _min_128 = cute.arch.fmin(m0_0[0], _shfl_xor_16, ftz=False)
                        m0_0[0] = cutlass.Float32(_min_128)
                        _shfl_xor_17 = cute.arch.shuffle_sync_bfly(m0_0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _min_129 = cute.arch.fmin(m0_0[0], _shfl_xor_17, ftz=False)
                        m0_0[0] = cutlass.Float32(_min_129)
                        _shfl_xor_18 = cute.arch.shuffle_sync_bfly(m1_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _min_130 = cute.arch.fmin(m1_1[0], _shfl_xor_18, ftz=False)
                        m1_1[0] = cutlass.Float32(_min_130)
                        _shfl_xor_19 = cute.arch.shuffle_sync_bfly(m1_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _min_131 = cute.arch.fmin(m1_1[0], _shfl_xor_19, ftz=False)
                        m1_1[0] = cutlass.Float32(_min_131)
                    new_max0[0] = cutlass.Float32(m0_0[0])
                    new_max1[0] = cutlass.Float32(m1_1[0])
                    merged_max0[0] = cutlass.Float32((new_max0[0] * scale_log2))
                    merged_max1[0] = cutlass.Float32((new_max1[0] * scale_log2))
                    _max_132 = cute.arch.fmax(merged_max0[0], row_max0[0], ftz=False)
                    merged_max0[0] = cutlass.Float32(_max_132)
                    _max_133 = cute.arch.fmax(merged_max1[0], row_max1[0], ftz=False)
                    merged_max1[0] = cutlass.Float32(_max_133)
                    _exp2_64 = cute.math.exp2((row_max0[0] - merged_max0[0]), approx=True, ftz=True)
                    _exp2_65 = cute.math.exp2((row_max1[0] - merged_max1[0]), approx=True, ftz=True)
                    row_max0[0] = cutlass.Float32(merged_max0[0])
                    row_max1[0] = cutlass.Float32(merged_max1[0])
                    new_sum0[0] = cutlass.Float32(0.0)
                    new_sum1[0] = cutlass.Float32(0.0)
                    sa0_0[0] = cutlass.Float32(0.0)
                    sa1_1[0] = cutlass.Float32(0.0)
                    sb0_2[0] = cutlass.Float32(0.0)
                    sb1_3[0] = cutlass.Float32(0.0)
                    _exp2_66 = cute.math.exp2(((d_qk[0] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_67 = cute.math.exp2(((d_qk[1] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[0] = cutlass.Float32(_exp2_66)
                    d_qk[1] = cutlass.Float32(_exp2_67)
                    sa0_0[0] += cutlass.Float32((_exp2_66 + _exp2_67))
                    _exp2_68 = cute.math.exp2(((d_qk[2] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_69 = cute.math.exp2(((d_qk[3] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[2] = cutlass.Float32(_exp2_68)
                    d_qk[3] = cutlass.Float32(_exp2_69)
                    sa1_1[0] += cutlass.Float32((_exp2_68 + _exp2_69))
                    _exp2_70 = cute.math.exp2(((d_qk[4] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_71 = cute.math.exp2(((d_qk[5] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[4] = cutlass.Float32(_exp2_70)
                    d_qk[5] = cutlass.Float32(_exp2_71)
                    sa0_0[0] += cutlass.Float32((_exp2_70 + _exp2_71))
                    _exp2_72 = cute.math.exp2(((d_qk[6] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_73 = cute.math.exp2(((d_qk[7] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[6] = cutlass.Float32(_exp2_72)
                    d_qk[7] = cutlass.Float32(_exp2_73)
                    sa1_1[0] += cutlass.Float32((_exp2_72 + _exp2_73))
                    _exp2_74 = cute.math.exp2(((d_qk[8] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_75 = cute.math.exp2(((d_qk[9] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[8] = cutlass.Float32(_exp2_74)
                    d_qk[9] = cutlass.Float32(_exp2_75)
                    sa0_0[0] += cutlass.Float32((_exp2_74 + _exp2_75))
                    _exp2_76 = cute.math.exp2(((d_qk[10] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_77 = cute.math.exp2(((d_qk[11] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[10] = cutlass.Float32(_exp2_76)
                    d_qk[11] = cutlass.Float32(_exp2_77)
                    sa1_1[0] += cutlass.Float32((_exp2_76 + _exp2_77))
                    _exp2_78 = cute.math.exp2(((d_qk[12] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_79 = cute.math.exp2(((d_qk[13] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[12] = cutlass.Float32(_exp2_78)
                    d_qk[13] = cutlass.Float32(_exp2_79)
                    sa0_0[0] += cutlass.Float32((_exp2_78 + _exp2_79))
                    _exp2_80 = cute.math.exp2(((d_qk[14] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_81 = cute.math.exp2(((d_qk[15] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[14] = cutlass.Float32(_exp2_80)
                    d_qk[15] = cutlass.Float32(_exp2_81)
                    sa1_1[0] += cutlass.Float32((_exp2_80 + _exp2_81))
                    _exp2_82 = cute.math.exp2(((d_qk[16] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_83 = cute.math.exp2(((d_qk[17] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[16] = cutlass.Float32(_exp2_82)
                    d_qk[17] = cutlass.Float32(_exp2_83)
                    sa0_0[0] += cutlass.Float32((_exp2_82 + _exp2_83))
                    _exp2_84 = cute.math.exp2(((d_qk[18] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_85 = cute.math.exp2(((d_qk[19] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[18] = cutlass.Float32(_exp2_84)
                    d_qk[19] = cutlass.Float32(_exp2_85)
                    sa1_1[0] += cutlass.Float32((_exp2_84 + _exp2_85))
                    _exp2_86 = cute.math.exp2(((d_qk[20] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_87 = cute.math.exp2(((d_qk[21] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[20] = cutlass.Float32(_exp2_86)
                    d_qk[21] = cutlass.Float32(_exp2_87)
                    sa0_0[0] += cutlass.Float32((_exp2_86 + _exp2_87))
                    _exp2_88 = cute.math.exp2(((d_qk[22] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_89 = cute.math.exp2(((d_qk[23] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[22] = cutlass.Float32(_exp2_88)
                    d_qk[23] = cutlass.Float32(_exp2_89)
                    sa1_1[0] += cutlass.Float32((_exp2_88 + _exp2_89))
                    _exp2_90 = cute.math.exp2(((d_qk[24] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_91 = cute.math.exp2(((d_qk[25] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[24] = cutlass.Float32(_exp2_90)
                    d_qk[25] = cutlass.Float32(_exp2_91)
                    sa0_0[0] += cutlass.Float32((_exp2_90 + _exp2_91))
                    _exp2_92 = cute.math.exp2(((d_qk[26] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_93 = cute.math.exp2(((d_qk[27] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[26] = cutlass.Float32(_exp2_92)
                    d_qk[27] = cutlass.Float32(_exp2_93)
                    sa1_1[0] += cutlass.Float32((_exp2_92 + _exp2_93))
                    _exp2_94 = cute.math.exp2(((d_qk[28] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_95 = cute.math.exp2(((d_qk[29] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[28] = cutlass.Float32(_exp2_94)
                    d_qk[29] = cutlass.Float32(_exp2_95)
                    sa0_0[0] += cutlass.Float32((_exp2_94 + _exp2_95))
                    _exp2_96 = cute.math.exp2(((d_qk[30] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_97 = cute.math.exp2(((d_qk[31] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[30] = cutlass.Float32(_exp2_96)
                    d_qk[31] = cutlass.Float32(_exp2_97)
                    sa1_1[0] += cutlass.Float32((_exp2_96 + _exp2_97))
                    _exp2_98 = cute.math.exp2(((d_qk[32] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_99 = cute.math.exp2(((d_qk[33] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[32] = cutlass.Float32(_exp2_98)
                    d_qk[33] = cutlass.Float32(_exp2_99)
                    sb0_2[0] += cutlass.Float32((_exp2_98 + _exp2_99))
                    _exp2_100 = cute.math.exp2(((d_qk[34] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_101 = cute.math.exp2(((d_qk[35] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[34] = cutlass.Float32(_exp2_100)
                    d_qk[35] = cutlass.Float32(_exp2_101)
                    sb1_3[0] += cutlass.Float32((_exp2_100 + _exp2_101))
                    _exp2_102 = cute.math.exp2(((d_qk[36] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_103 = cute.math.exp2(((d_qk[37] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[36] = cutlass.Float32(_exp2_102)
                    d_qk[37] = cutlass.Float32(_exp2_103)
                    sb0_2[0] += cutlass.Float32((_exp2_102 + _exp2_103))
                    _exp2_104 = cute.math.exp2(((d_qk[38] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_105 = cute.math.exp2(((d_qk[39] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[38] = cutlass.Float32(_exp2_104)
                    d_qk[39] = cutlass.Float32(_exp2_105)
                    sb1_3[0] += cutlass.Float32((_exp2_104 + _exp2_105))
                    _exp2_106 = cute.math.exp2(((d_qk[40] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_107 = cute.math.exp2(((d_qk[41] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[40] = cutlass.Float32(_exp2_106)
                    d_qk[41] = cutlass.Float32(_exp2_107)
                    sb0_2[0] += cutlass.Float32((_exp2_106 + _exp2_107))
                    _exp2_108 = cute.math.exp2(((d_qk[42] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_109 = cute.math.exp2(((d_qk[43] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[42] = cutlass.Float32(_exp2_108)
                    d_qk[43] = cutlass.Float32(_exp2_109)
                    sb1_3[0] += cutlass.Float32((_exp2_108 + _exp2_109))
                    _exp2_110 = cute.math.exp2(((d_qk[44] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_111 = cute.math.exp2(((d_qk[45] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[44] = cutlass.Float32(_exp2_110)
                    d_qk[45] = cutlass.Float32(_exp2_111)
                    sb0_2[0] += cutlass.Float32((_exp2_110 + _exp2_111))
                    _exp2_112 = cute.math.exp2(((d_qk[46] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_113 = cute.math.exp2(((d_qk[47] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[46] = cutlass.Float32(_exp2_112)
                    d_qk[47] = cutlass.Float32(_exp2_113)
                    sb1_3[0] += cutlass.Float32((_exp2_112 + _exp2_113))
                    _exp2_114 = cute.math.exp2(((d_qk[48] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_115 = cute.math.exp2(((d_qk[49] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[48] = cutlass.Float32(_exp2_114)
                    d_qk[49] = cutlass.Float32(_exp2_115)
                    sb0_2[0] += cutlass.Float32((_exp2_114 + _exp2_115))
                    _exp2_116 = cute.math.exp2(((d_qk[50] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_117 = cute.math.exp2(((d_qk[51] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[50] = cutlass.Float32(_exp2_116)
                    d_qk[51] = cutlass.Float32(_exp2_117)
                    sb1_3[0] += cutlass.Float32((_exp2_116 + _exp2_117))
                    _exp2_118 = cute.math.exp2(((d_qk[52] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_119 = cute.math.exp2(((d_qk[53] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[52] = cutlass.Float32(_exp2_118)
                    d_qk[53] = cutlass.Float32(_exp2_119)
                    sb0_2[0] += cutlass.Float32((_exp2_118 + _exp2_119))
                    _exp2_120 = cute.math.exp2(((d_qk[54] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_121 = cute.math.exp2(((d_qk[55] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[54] = cutlass.Float32(_exp2_120)
                    d_qk[55] = cutlass.Float32(_exp2_121)
                    sb1_3[0] += cutlass.Float32((_exp2_120 + _exp2_121))
                    _exp2_122 = cute.math.exp2(((d_qk[56] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_123 = cute.math.exp2(((d_qk[57] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[56] = cutlass.Float32(_exp2_122)
                    d_qk[57] = cutlass.Float32(_exp2_123)
                    sb0_2[0] += cutlass.Float32((_exp2_122 + _exp2_123))
                    _exp2_124 = cute.math.exp2(((d_qk[58] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_125 = cute.math.exp2(((d_qk[59] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[58] = cutlass.Float32(_exp2_124)
                    d_qk[59] = cutlass.Float32(_exp2_125)
                    sb1_3[0] += cutlass.Float32((_exp2_124 + _exp2_125))
                    _exp2_126 = cute.math.exp2(((d_qk[60] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    _exp2_127 = cute.math.exp2(((d_qk[61] * scale_log2) - merged_max0[0]), approx=True, ftz=True)
                    d_qk[60] = cutlass.Float32(_exp2_126)
                    d_qk[61] = cutlass.Float32(_exp2_127)
                    sb0_2[0] += cutlass.Float32((_exp2_126 + _exp2_127))
                    _exp2_128 = cute.math.exp2(((d_qk[62] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    _exp2_129 = cute.math.exp2(((d_qk[63] * scale_log2) - merged_max1[0]), approx=True, ftz=True)
                    d_qk[62] = cutlass.Float32(_exp2_128)
                    d_qk[63] = cutlass.Float32(_exp2_129)
                    sb1_3[0] += cutlass.Float32((_exp2_128 + _exp2_129))
                    new_sum0[0] += cutlass.Float32((sa0_0[0] + (sb0_2[0] if (nxt_has2 != 0) else 0.0)))
                    new_sum1[0] += cutlass.Float32((sa1_1[0] + (sb1_3[0] if (nxt_has2 != 0) else 0.0)))
                    _shfl_xor_20 = cute.arch.shuffle_sync_bfly(new_sum0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    new_sum0[0] += cutlass.Float32(_shfl_xor_20)
                    _shfl_xor_21 = cute.arch.shuffle_sync_bfly(new_sum0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    new_sum0[0] += cutlass.Float32(_shfl_xor_21)
                    _shfl_xor_22 = cute.arch.shuffle_sync_bfly(new_sum1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                    new_sum1[0] += cutlass.Float32(_shfl_xor_22)
                    _shfl_xor_23 = cute.arch.shuffle_sync_bfly(new_sum1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                    new_sum1[0] += cutlass.Float32(_shfl_xor_23)
                    cute.nvgpu.warpgroup.wait_group(0)
                    if prims.elect_sync():
                        cute.arch.mbarrier_arrive(v_empty_addr + (cur_pos[0] % 3))
                    _vote_0 = cutlass.Int32(cute.arch.vote_any_sync(cutlass.Boolean(((_exp2_64 != 1.0) | (_exp2_65 != 1.0))), cutlass.Uint32(0xFFFFFFFF)))
                    need_rescale = cutlass.Int32(_vote_0)
                    if (need_rescale != 0):
                        d_o[0] = cutlass.Float32((d_o[0] * _exp2_64))
                        d_o[1] = cutlass.Float32((d_o[1] * _exp2_64))
                        d_o[4] = cutlass.Float32((d_o[4] * _exp2_64))
                        d_o[5] = cutlass.Float32((d_o[5] * _exp2_64))
                        d_o[8] = cutlass.Float32((d_o[8] * _exp2_64))
                        d_o[9] = cutlass.Float32((d_o[9] * _exp2_64))
                        d_o[12] = cutlass.Float32((d_o[12] * _exp2_64))
                        d_o[13] = cutlass.Float32((d_o[13] * _exp2_64))
                        d_o[16] = cutlass.Float32((d_o[16] * _exp2_64))
                        d_o[17] = cutlass.Float32((d_o[17] * _exp2_64))
                        d_o[20] = cutlass.Float32((d_o[20] * _exp2_64))
                        d_o[21] = cutlass.Float32((d_o[21] * _exp2_64))
                        d_o[24] = cutlass.Float32((d_o[24] * _exp2_64))
                        d_o[25] = cutlass.Float32((d_o[25] * _exp2_64))
                        d_o[28] = cutlass.Float32((d_o[28] * _exp2_64))
                        d_o[29] = cutlass.Float32((d_o[29] * _exp2_64))
                        d_o[32] = cutlass.Float32((d_o[32] * _exp2_64))
                        d_o[33] = cutlass.Float32((d_o[33] * _exp2_64))
                        d_o[36] = cutlass.Float32((d_o[36] * _exp2_64))
                        d_o[37] = cutlass.Float32((d_o[37] * _exp2_64))
                        d_o[40] = cutlass.Float32((d_o[40] * _exp2_64))
                        d_o[41] = cutlass.Float32((d_o[41] * _exp2_64))
                        d_o[44] = cutlass.Float32((d_o[44] * _exp2_64))
                        d_o[45] = cutlass.Float32((d_o[45] * _exp2_64))
                        d_o[48] = cutlass.Float32((d_o[48] * _exp2_64))
                        d_o[49] = cutlass.Float32((d_o[49] * _exp2_64))
                        d_o[52] = cutlass.Float32((d_o[52] * _exp2_64))
                        d_o[53] = cutlass.Float32((d_o[53] * _exp2_64))
                        d_o[56] = cutlass.Float32((d_o[56] * _exp2_64))
                        d_o[57] = cutlass.Float32((d_o[57] * _exp2_64))
                        d_o[60] = cutlass.Float32((d_o[60] * _exp2_64))
                        d_o[61] = cutlass.Float32((d_o[61] * _exp2_64))
                        d_o[2] = cutlass.Float32((d_o[2] * _exp2_65))
                        d_o[3] = cutlass.Float32((d_o[3] * _exp2_65))
                        d_o[6] = cutlass.Float32((d_o[6] * _exp2_65))
                        d_o[7] = cutlass.Float32((d_o[7] * _exp2_65))
                        d_o[10] = cutlass.Float32((d_o[10] * _exp2_65))
                        d_o[11] = cutlass.Float32((d_o[11] * _exp2_65))
                        d_o[14] = cutlass.Float32((d_o[14] * _exp2_65))
                        d_o[15] = cutlass.Float32((d_o[15] * _exp2_65))
                        d_o[18] = cutlass.Float32((d_o[18] * _exp2_65))
                        d_o[19] = cutlass.Float32((d_o[19] * _exp2_65))
                        d_o[22] = cutlass.Float32((d_o[22] * _exp2_65))
                        d_o[23] = cutlass.Float32((d_o[23] * _exp2_65))
                        d_o[26] = cutlass.Float32((d_o[26] * _exp2_65))
                        d_o[27] = cutlass.Float32((d_o[27] * _exp2_65))
                        d_o[30] = cutlass.Float32((d_o[30] * _exp2_65))
                        d_o[31] = cutlass.Float32((d_o[31] * _exp2_65))
                        d_o[34] = cutlass.Float32((d_o[34] * _exp2_65))
                        d_o[35] = cutlass.Float32((d_o[35] * _exp2_65))
                        d_o[38] = cutlass.Float32((d_o[38] * _exp2_65))
                        d_o[39] = cutlass.Float32((d_o[39] * _exp2_65))
                        d_o[42] = cutlass.Float32((d_o[42] * _exp2_65))
                        d_o[43] = cutlass.Float32((d_o[43] * _exp2_65))
                        d_o[46] = cutlass.Float32((d_o[46] * _exp2_65))
                        d_o[47] = cutlass.Float32((d_o[47] * _exp2_65))
                        d_o[50] = cutlass.Float32((d_o[50] * _exp2_65))
                        d_o[51] = cutlass.Float32((d_o[51] * _exp2_65))
                        d_o[54] = cutlass.Float32((d_o[54] * _exp2_65))
                        d_o[55] = cutlass.Float32((d_o[55] * _exp2_65))
                        d_o[58] = cutlass.Float32((d_o[58] * _exp2_65))
                        d_o[59] = cutlass.Float32((d_o[59] * _exp2_65))
                        d_o[62] = cutlass.Float32((d_o[62] * _exp2_65))
                        d_o[63] = cutlass.Float32((d_o[63] * _exp2_65))
                    row_sum0[0] = cutlass.Float32(((row_sum0[0] * _exp2_64) + new_sum0[0]))
                    row_sum1[0] = cutlass.Float32(((row_sum1[0] * _exp2_65) + new_sum1[0]))
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
                    _bf16x2_48 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[32]), cutlass.Float32(d_qk[33])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[32]), cutlass.Float32(d_qk[33])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[16] = cutlass.Uint32(_bf16x2_48)
                    _bf16x2_49 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[34]), cutlass.Float32(d_qk[35])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[34]), cutlass.Float32(d_qk[35])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[17] = cutlass.Uint32(_bf16x2_49)
                    _bf16x2_50 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[36]), cutlass.Float32(d_qk[37])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[36]), cutlass.Float32(d_qk[37])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[18] = cutlass.Uint32(_bf16x2_50)
                    _bf16x2_51 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[38]), cutlass.Float32(d_qk[39])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[38]), cutlass.Float32(d_qk[39])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[19] = cutlass.Uint32(_bf16x2_51)
                    _bf16x2_52 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[40]), cutlass.Float32(d_qk[41])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[40]), cutlass.Float32(d_qk[41])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[20] = cutlass.Uint32(_bf16x2_52)
                    _bf16x2_53 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[42]), cutlass.Float32(d_qk[43])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[42]), cutlass.Float32(d_qk[43])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[21] = cutlass.Uint32(_bf16x2_53)
                    _bf16x2_54 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[44]), cutlass.Float32(d_qk[45])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[44]), cutlass.Float32(d_qk[45])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[22] = cutlass.Uint32(_bf16x2_54)
                    _bf16x2_55 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[46]), cutlass.Float32(d_qk[47])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[46]), cutlass.Float32(d_qk[47])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[23] = cutlass.Uint32(_bf16x2_55)
                    _bf16x2_56 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[48]), cutlass.Float32(d_qk[49])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[48]), cutlass.Float32(d_qk[49])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[24] = cutlass.Uint32(_bf16x2_56)
                    _bf16x2_57 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[50]), cutlass.Float32(d_qk[51])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[50]), cutlass.Float32(d_qk[51])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[25] = cutlass.Uint32(_bf16x2_57)
                    _bf16x2_58 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[52]), cutlass.Float32(d_qk[53])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[52]), cutlass.Float32(d_qk[53])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[26] = cutlass.Uint32(_bf16x2_58)
                    _bf16x2_59 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[54]), cutlass.Float32(d_qk[55])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[54]), cutlass.Float32(d_qk[55])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[27] = cutlass.Uint32(_bf16x2_59)
                    _bf16x2_60 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[56]), cutlass.Float32(d_qk[57])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[56]), cutlass.Float32(d_qk[57])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[28] = cutlass.Uint32(_bf16x2_60)
                    _bf16x2_61 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[58]), cutlass.Float32(d_qk[59])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[58]), cutlass.Float32(d_qk[59])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[29] = cutlass.Uint32(_bf16x2_61)
                    _bf16x2_62 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[60]), cutlass.Float32(d_qk[61])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[60]), cutlass.Float32(d_qk[61])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[30] = cutlass.Uint32(_bf16x2_62)
                    _bf16x2_63 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[62]), cutlass.Float32(d_qk[63])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[62]), cutlass.Float32(d_qk[63])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                    p_bf16[31] = cutlass.Uint32(_bf16x2_63)
                    if (nxt_has2 == 0):
                        p_bf16[16] = cutlass.Uint32(0)
                        p_bf16[17] = cutlass.Uint32(0)
                        p_bf16[18] = cutlass.Uint32(0)
                        p_bf16[19] = cutlass.Uint32(0)
                        p_bf16[20] = cutlass.Uint32(0)
                        p_bf16[21] = cutlass.Uint32(0)
                        p_bf16[22] = cutlass.Uint32(0)
                        p_bf16[23] = cutlass.Uint32(0)
                        p_bf16[24] = cutlass.Uint32(0)
                        p_bf16[25] = cutlass.Uint32(0)
                        p_bf16[26] = cutlass.Uint32(0)
                        p_bf16[27] = cutlass.Uint32(0)
                        p_bf16[28] = cutlass.Uint32(0)
                        p_bf16[29] = cutlass.Uint32(0)
                        p_bf16[30] = cutlass.Uint32(0)
                        p_bf16[31] = cutlass.Uint32(0)
                    prev[0] = cutlass.Int32(nxt_pos)
                    cur_pos[0] = cutlass.Int32(nxt_pos)
                    cur_has2[0] = cutlass.Int32(nxt_has2)
                last_stage = cutlass.Int32((cur_pos[0] % 3))
                while not prims.mbarrier_wait_parity(v_full_addr + last_stage, (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(cur_pos[0]).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                cute.nvgpu.warpgroup.fence()
                _wgmma_b_0_6_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_a_addr + cutlass.Uint32((last_stage * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                _wgmma_b_0_6 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw)))
                _wgmma_38_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_38 = cutlass_llvm.inline_asm(
                    _wgmma_38_ty,
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
                        cutlass.Uint64(_wgmma_b_0_6).ir_value(),
                        cutlass.Uint32(p_bf16[(0) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(0) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(0) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(0) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_38, position=[63]))
                _wgmma_39_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_39 = cutlass_llvm.inline_asm(
                    _wgmma_39_ty,
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
                        cutlass.Uint64((_wgmma_b_0_6 + 128)).ir_value(),
                        cutlass.Uint32(p_bf16[(4) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(4) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(4) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(4) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_39, position=[63]))
                _wgmma_40_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_40 = cutlass_llvm.inline_asm(
                    _wgmma_40_ty,
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
                        cutlass.Uint64((_wgmma_b_0_6 + 256)).ir_value(),
                        cutlass.Uint32(p_bf16[(8) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(8) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(8) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(8) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_40, position=[63]))
                _wgmma_41_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_41 = cutlass_llvm.inline_asm(
                    _wgmma_41_ty,
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
                        cutlass.Uint64((_wgmma_b_0_6 + 384)).ir_value(),
                        cutlass.Uint32(p_bf16[(12) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(12) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(12) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(12) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_41, position=[63]))
                _wgmma_b_0_7_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_b_addr + cutlass.Uint32((last_stage * 32768)))) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
                _wgmma_b_0_7 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw)))
                _wgmma_42_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_42 = cutlass_llvm.inline_asm(
                    _wgmma_42_ty,
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
                        cutlass.Uint32(p_bf16[(16) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(16) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(16) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(16) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_42, position=[63]))
                _wgmma_43_ty = cutlass_llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 64)
                _wgmma_43 = cutlass_llvm.inline_asm(
                    _wgmma_43_ty,
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
                        cutlass.Uint32(p_bf16[(20) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(20) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(20) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(20) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                d_o[(0) + 0] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[0]))
                d_o[(0) + 1] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[1]))
                d_o[(0) + 2] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[2]))
                d_o[(0) + 3] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[3]))
                d_o[(0) + 4] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[4]))
                d_o[(0) + 5] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[5]))
                d_o[(0) + 6] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[6]))
                d_o[(0) + 7] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[7]))
                d_o[(0) + 8] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[8]))
                d_o[(0) + 9] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[9]))
                d_o[(0) + 10] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[10]))
                d_o[(0) + 11] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[11]))
                d_o[(0) + 12] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[12]))
                d_o[(0) + 13] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[13]))
                d_o[(0) + 14] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[14]))
                d_o[(0) + 15] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[15]))
                d_o[(0) + 16] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[16]))
                d_o[(0) + 17] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[17]))
                d_o[(0) + 18] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[18]))
                d_o[(0) + 19] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[19]))
                d_o[(0) + 20] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[20]))
                d_o[(0) + 21] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[21]))
                d_o[(0) + 22] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[22]))
                d_o[(0) + 23] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[23]))
                d_o[(0) + 24] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[24]))
                d_o[(0) + 25] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[25]))
                d_o[(0) + 26] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[26]))
                d_o[(0) + 27] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[27]))
                d_o[(0) + 28] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[28]))
                d_o[(0) + 29] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[29]))
                d_o[(0) + 30] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[30]))
                d_o[(0) + 31] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[31]))
                d_o[(0) + 32] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[32]))
                d_o[(0) + 33] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[33]))
                d_o[(0) + 34] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[34]))
                d_o[(0) + 35] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[35]))
                d_o[(0) + 36] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[36]))
                d_o[(0) + 37] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[37]))
                d_o[(0) + 38] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[38]))
                d_o[(0) + 39] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[39]))
                d_o[(0) + 40] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[40]))
                d_o[(0) + 41] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[41]))
                d_o[(0) + 42] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[42]))
                d_o[(0) + 43] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[43]))
                d_o[(0) + 44] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[44]))
                d_o[(0) + 45] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[45]))
                d_o[(0) + 46] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[46]))
                d_o[(0) + 47] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[47]))
                d_o[(0) + 48] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[48]))
                d_o[(0) + 49] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[49]))
                d_o[(0) + 50] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[50]))
                d_o[(0) + 51] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[51]))
                d_o[(0) + 52] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[52]))
                d_o[(0) + 53] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[53]))
                d_o[(0) + 54] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[54]))
                d_o[(0) + 55] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[55]))
                d_o[(0) + 56] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[56]))
                d_o[(0) + 57] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[57]))
                d_o[(0) + 58] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[58]))
                d_o[(0) + 59] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[59]))
                d_o[(0) + 60] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[60]))
                d_o[(0) + 61] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[61]))
                d_o[(0) + 62] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[62]))
                d_o[(0) + 63] = cutlass.Float32(cutlass_llvm.extractvalue(cutlass.Float32.mlir_type, _wgmma_43, position=[63]))
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
                        cutlass.Uint64((_wgmma_b_0_7 + 256)).ir_value(),
                        cutlass.Uint32(p_bf16[(24) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(24) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(24) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(24) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                        cutlass.Uint64((_wgmma_b_0_7 + 384)).ir_value(),
                        cutlass.Uint32(p_bf16[(28) + 0]).ir_value(),
                        cutlass.Uint32(p_bf16[(28) + 1]).ir_value(),
                        cutlass.Uint32(p_bf16[(28) + 2]).ir_value(),
                        cutlass.Uint32(p_bf16[(28) + 3]).ir_value(),
                    ],
                    asm_string='{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, {$129, $130, $131, $132}, $128, 1, 1, 1, 1;\n}\n',
                    constraints='=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,l,r,r,r,r,~{memory}',
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
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive(v_empty_addr + (cur_pos[0] % 3))
                prev[0] = cutlass.Int32(cur_pos[0])
            tile_end = cutlass.Int32((gbase[0] + n_seq_m))
            tail_lo = cutlass.Int32((prev[0] + 1))
            for p_3 in cutlass.range(cutlass.Int32(tail_lo), cutlass.Int32(tile_end), cutlass.Int32(1), unroll=1, unroll_full=False):
                while not prims.mbarrier_wait_parity(k_full_addr + (p_3 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_3).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive(k_empty_addr + (p_3 % 3))
                while not prims.mbarrier_wait_parity(v_full_addr + (p_3 % 3), (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(p_3).ir_value(), cutlass.Int32(3).ir_value())) & 1), prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive(v_empty_addr + (p_3 % 3))
            prev[0] = cutlass.Int32((tile_end - 1))
            gbase[0] = cutlass.Int32(tile_end)
            store_wg[0] = cutlass.Int32(1)
            partner_idle = cutlass.Int32((n_other == 0))
            self_idle = cutlass.Int32((n_own == 0))
            skip_merge = cutlass.Int32((1 if (partner_idle != 0) else (1 if (self_idle != 0) else 0)))
            _if_condition_46 = cutlass.Boolean((self_idle != 0))
            store_wg[0] = cutlass.Int32(cutlass.select_(_if_condition_46, cutlass.Int32(0), store_wg[0]))
            do_merge[0] = cutlass.Int32((1 if (is_split != 0) else 0))
            _if_condition_47 = cutlass.Boolean((skip_merge != 0))
            do_merge[0] = cutlass.Int32(cutlass.select_(_if_condition_47, cutlass.Int32(0), do_merge[0]))
            a0[0] = cutlass.Float32(1.0)
            a1[0] = cutlass.Float32(1.0)
            b0[0] = cutlass.Float32(0.0)
            b1[0] = cutlass.Float32(0.0)
            if (do_merge[0] != 0):
                prims.barrier_cta_sync(9, thread_count=256)
                if (cwg == 1):
                    prims.store_ext(cutlass.Float32(d_o[0]).ir_value(), (merge_o + (tid_wg)))
                    prims.store_ext(cutlass.Float32(d_o[1]).ir_value(), (merge_o + ((128 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[2]).ir_value(), (merge_o + ((256 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[3]).ir_value(), (merge_o + ((384 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[4]).ir_value(), (merge_o + ((512 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[5]).ir_value(), (merge_o + ((640 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[6]).ir_value(), (merge_o + ((768 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[7]).ir_value(), (merge_o + ((896 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[8]).ir_value(), (merge_o + ((1024 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[9]).ir_value(), (merge_o + ((1152 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[10]).ir_value(), (merge_o + ((1280 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[11]).ir_value(), (merge_o + ((1408 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[12]).ir_value(), (merge_o + ((1536 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[13]).ir_value(), (merge_o + ((1664 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[14]).ir_value(), (merge_o + ((1792 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[15]).ir_value(), (merge_o + ((1920 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[16]).ir_value(), (merge_o + ((2048 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[17]).ir_value(), (merge_o + ((2176 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[18]).ir_value(), (merge_o + ((2304 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[19]).ir_value(), (merge_o + ((2432 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[20]).ir_value(), (merge_o + ((2560 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[21]).ir_value(), (merge_o + ((2688 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[22]).ir_value(), (merge_o + ((2816 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[23]).ir_value(), (merge_o + ((2944 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[24]).ir_value(), (merge_o + ((3072 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[25]).ir_value(), (merge_o + ((3200 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[26]).ir_value(), (merge_o + ((3328 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[27]).ir_value(), (merge_o + ((3456 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[28]).ir_value(), (merge_o + ((3584 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[29]).ir_value(), (merge_o + ((3712 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[30]).ir_value(), (merge_o + ((3840 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[31]).ir_value(), (merge_o + ((3968 + tid_wg))))
                    if ((lane & 3) == 0):
                        prims.store_ext(cutlass.Float32(row_max0[0]).ir_value(), (merge_ml + ((quad * 4))))
                        prims.store_ext(cutlass.Float32(row_max1[0]).ir_value(), (merge_ml + (((quad * 4) + 1))))
                        prims.store_ext(cutlass.Float32(row_sum0[0]).ir_value(), (merge_ml + (((quad * 4) + 2))))
                        prims.store_ext(cutlass.Float32(row_sum1[0]).ir_value(), (merge_ml + (((quad * 4) + 3))))
                    store_wg[0] = cutlass.Int32(0)
                prims.barrier_cta_sync(9, thread_count=256)
                if (cwg == 0):
                    pm0 = cutlass.Float32(_merge_ml[(quad * 4)])
                    pm1 = cutlass.Float32(_merge_ml[((quad * 4) + 1)])
                    pl0 = cutlass.Float32(_merge_ml[((quad * 4) + 2)])
                    pl1 = cutlass.Float32(_merge_ml[((quad * 4) + 3)])
                    _max_134 = cute.arch.fmax(row_max0[0], pm0, ftz=False)
                    mm0 = cutlass.Float32(_max_134)
                    _max_135 = cute.arch.fmax(row_max1[0], pm1, ftz=False)
                    mm1 = cutlass.Float32(_max_135)
                    _exp2_130 = cute.math.exp2((row_max0[0] - mm0), approx=True, ftz=True)
                    a0[0] = cutlass.Float32((0.0 if (row_max0[0] == (0 - float("inf"))) else _exp2_130))
                    _exp2_131 = cute.math.exp2((row_max1[0] - mm1), approx=True, ftz=True)
                    a1[0] = cutlass.Float32((0.0 if (row_max1[0] == (0 - float("inf"))) else _exp2_131))
                    _exp2_132 = cute.math.exp2((pm0 - mm0), approx=True, ftz=True)
                    b0[0] = cutlass.Float32((0.0 if (pm0 == (0 - float("inf"))) else _exp2_132))
                    _exp2_133 = cute.math.exp2((pm1 - mm1), approx=True, ftz=True)
                    b1[0] = cutlass.Float32((0.0 if (pm1 == (0 - float("inf"))) else _exp2_133))
                    po = cutlass.Float32(_merge_o[tid_wg])
                    d_o[0] = cutlass.Float32(((d_o[0] * a0[0]) + (po * b0[0])))
                    po_0 = cutlass.Float32(_merge_o[(128 + tid_wg)])
                    d_o[1] = cutlass.Float32(((d_o[1] * a0[0]) + (po_0 * b0[0])))
                    po_1 = cutlass.Float32(_merge_o[(256 + tid_wg)])
                    d_o[2] = cutlass.Float32(((d_o[2] * a1[0]) + (po_1 * b1[0])))
                    po_2 = cutlass.Float32(_merge_o[(384 + tid_wg)])
                    d_o[3] = cutlass.Float32(((d_o[3] * a1[0]) + (po_2 * b1[0])))
                    po_3 = cutlass.Float32(_merge_o[(512 + tid_wg)])
                    d_o[4] = cutlass.Float32(((d_o[4] * a0[0]) + (po_3 * b0[0])))
                    po_4 = cutlass.Float32(_merge_o[(640 + tid_wg)])
                    d_o[5] = cutlass.Float32(((d_o[5] * a0[0]) + (po_4 * b0[0])))
                    po_5 = cutlass.Float32(_merge_o[(768 + tid_wg)])
                    d_o[6] = cutlass.Float32(((d_o[6] * a1[0]) + (po_5 * b1[0])))
                    po_6 = cutlass.Float32(_merge_o[(896 + tid_wg)])
                    d_o[7] = cutlass.Float32(((d_o[7] * a1[0]) + (po_6 * b1[0])))
                    po_7 = cutlass.Float32(_merge_o[(1024 + tid_wg)])
                    d_o[8] = cutlass.Float32(((d_o[8] * a0[0]) + (po_7 * b0[0])))
                    po_8 = cutlass.Float32(_merge_o[(1152 + tid_wg)])
                    d_o[9] = cutlass.Float32(((d_o[9] * a0[0]) + (po_8 * b0[0])))
                    po_9 = cutlass.Float32(_merge_o[(1280 + tid_wg)])
                    d_o[10] = cutlass.Float32(((d_o[10] * a1[0]) + (po_9 * b1[0])))
                    po_10 = cutlass.Float32(_merge_o[(1408 + tid_wg)])
                    d_o[11] = cutlass.Float32(((d_o[11] * a1[0]) + (po_10 * b1[0])))
                    po_11 = cutlass.Float32(_merge_o[(1536 + tid_wg)])
                    d_o[12] = cutlass.Float32(((d_o[12] * a0[0]) + (po_11 * b0[0])))
                    po_12 = cutlass.Float32(_merge_o[(1664 + tid_wg)])
                    d_o[13] = cutlass.Float32(((d_o[13] * a0[0]) + (po_12 * b0[0])))
                    po_13 = cutlass.Float32(_merge_o[(1792 + tid_wg)])
                    d_o[14] = cutlass.Float32(((d_o[14] * a1[0]) + (po_13 * b1[0])))
                    po_14 = cutlass.Float32(_merge_o[(1920 + tid_wg)])
                    d_o[15] = cutlass.Float32(((d_o[15] * a1[0]) + (po_14 * b1[0])))
                    po_15 = cutlass.Float32(_merge_o[(2048 + tid_wg)])
                    d_o[16] = cutlass.Float32(((d_o[16] * a0[0]) + (po_15 * b0[0])))
                    po_16 = cutlass.Float32(_merge_o[(2176 + tid_wg)])
                    d_o[17] = cutlass.Float32(((d_o[17] * a0[0]) + (po_16 * b0[0])))
                    po_17 = cutlass.Float32(_merge_o[(2304 + tid_wg)])
                    d_o[18] = cutlass.Float32(((d_o[18] * a1[0]) + (po_17 * b1[0])))
                    po_18 = cutlass.Float32(_merge_o[(2432 + tid_wg)])
                    d_o[19] = cutlass.Float32(((d_o[19] * a1[0]) + (po_18 * b1[0])))
                    po_19 = cutlass.Float32(_merge_o[(2560 + tid_wg)])
                    d_o[20] = cutlass.Float32(((d_o[20] * a0[0]) + (po_19 * b0[0])))
                    po_20 = cutlass.Float32(_merge_o[(2688 + tid_wg)])
                    d_o[21] = cutlass.Float32(((d_o[21] * a0[0]) + (po_20 * b0[0])))
                    po_21 = cutlass.Float32(_merge_o[(2816 + tid_wg)])
                    d_o[22] = cutlass.Float32(((d_o[22] * a1[0]) + (po_21 * b1[0])))
                    po_22 = cutlass.Float32(_merge_o[(2944 + tid_wg)])
                    d_o[23] = cutlass.Float32(((d_o[23] * a1[0]) + (po_22 * b1[0])))
                    po_23 = cutlass.Float32(_merge_o[(3072 + tid_wg)])
                    d_o[24] = cutlass.Float32(((d_o[24] * a0[0]) + (po_23 * b0[0])))
                    po_24 = cutlass.Float32(_merge_o[(3200 + tid_wg)])
                    d_o[25] = cutlass.Float32(((d_o[25] * a0[0]) + (po_24 * b0[0])))
                    po_25 = cutlass.Float32(_merge_o[(3328 + tid_wg)])
                    d_o[26] = cutlass.Float32(((d_o[26] * a1[0]) + (po_25 * b1[0])))
                    po_26 = cutlass.Float32(_merge_o[(3456 + tid_wg)])
                    d_o[27] = cutlass.Float32(((d_o[27] * a1[0]) + (po_26 * b1[0])))
                    po_27 = cutlass.Float32(_merge_o[(3584 + tid_wg)])
                    d_o[28] = cutlass.Float32(((d_o[28] * a0[0]) + (po_27 * b0[0])))
                    po_28 = cutlass.Float32(_merge_o[(3712 + tid_wg)])
                    d_o[29] = cutlass.Float32(((d_o[29] * a0[0]) + (po_28 * b0[0])))
                    po_29 = cutlass.Float32(_merge_o[(3840 + tid_wg)])
                    d_o[30] = cutlass.Float32(((d_o[30] * a1[0]) + (po_29 * b1[0])))
                    po_30 = cutlass.Float32(_merge_o[(3968 + tid_wg)])
                    d_o[31] = cutlass.Float32(((d_o[31] * a1[0]) + (po_30 * b1[0])))
                    row_sum0[0] = cutlass.Float32(((row_sum0[0] * a0[0]) + (pl0 * b0[0])))
                    row_sum1[0] = cutlass.Float32(((row_sum1[0] * a1[0]) + (pl1 * b1[0])))
                prims.barrier_cta_sync(9, thread_count=256)
                if (cwg == 1):
                    prims.store_ext(cutlass.Float32(d_o[32]).ir_value(), (merge_o + (tid_wg)))
                    prims.store_ext(cutlass.Float32(d_o[33]).ir_value(), (merge_o + ((128 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[34]).ir_value(), (merge_o + ((256 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[35]).ir_value(), (merge_o + ((384 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[36]).ir_value(), (merge_o + ((512 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[37]).ir_value(), (merge_o + ((640 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[38]).ir_value(), (merge_o + ((768 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[39]).ir_value(), (merge_o + ((896 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[40]).ir_value(), (merge_o + ((1024 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[41]).ir_value(), (merge_o + ((1152 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[42]).ir_value(), (merge_o + ((1280 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[43]).ir_value(), (merge_o + ((1408 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[44]).ir_value(), (merge_o + ((1536 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[45]).ir_value(), (merge_o + ((1664 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[46]).ir_value(), (merge_o + ((1792 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[47]).ir_value(), (merge_o + ((1920 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[48]).ir_value(), (merge_o + ((2048 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[49]).ir_value(), (merge_o + ((2176 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[50]).ir_value(), (merge_o + ((2304 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[51]).ir_value(), (merge_o + ((2432 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[52]).ir_value(), (merge_o + ((2560 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[53]).ir_value(), (merge_o + ((2688 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[54]).ir_value(), (merge_o + ((2816 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[55]).ir_value(), (merge_o + ((2944 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[56]).ir_value(), (merge_o + ((3072 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[57]).ir_value(), (merge_o + ((3200 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[58]).ir_value(), (merge_o + ((3328 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[59]).ir_value(), (merge_o + ((3456 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[60]).ir_value(), (merge_o + ((3584 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[61]).ir_value(), (merge_o + ((3712 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[62]).ir_value(), (merge_o + ((3840 + tid_wg))))
                    prims.store_ext(cutlass.Float32(d_o[63]).ir_value(), (merge_o + ((3968 + tid_wg))))
                prims.barrier_cta_sync(9, thread_count=256)
                if (cwg == 0):
                    po_h = cutlass.Float32(_merge_o[tid_wg])
                    d_o[32] = cutlass.Float32(((d_o[32] * a0[0]) + (po_h * b0[0])))
                    po_h_0 = cutlass.Float32(_merge_o[(128 + tid_wg)])
                    d_o[33] = cutlass.Float32(((d_o[33] * a0[0]) + (po_h_0 * b0[0])))
                    po_h_1 = cutlass.Float32(_merge_o[(256 + tid_wg)])
                    d_o[34] = cutlass.Float32(((d_o[34] * a1[0]) + (po_h_1 * b1[0])))
                    po_h_2 = cutlass.Float32(_merge_o[(384 + tid_wg)])
                    d_o[35] = cutlass.Float32(((d_o[35] * a1[0]) + (po_h_2 * b1[0])))
                    po_h_3 = cutlass.Float32(_merge_o[(512 + tid_wg)])
                    d_o[36] = cutlass.Float32(((d_o[36] * a0[0]) + (po_h_3 * b0[0])))
                    po_h_4 = cutlass.Float32(_merge_o[(640 + tid_wg)])
                    d_o[37] = cutlass.Float32(((d_o[37] * a0[0]) + (po_h_4 * b0[0])))
                    po_h_5 = cutlass.Float32(_merge_o[(768 + tid_wg)])
                    d_o[38] = cutlass.Float32(((d_o[38] * a1[0]) + (po_h_5 * b1[0])))
                    po_h_6 = cutlass.Float32(_merge_o[(896 + tid_wg)])
                    d_o[39] = cutlass.Float32(((d_o[39] * a1[0]) + (po_h_6 * b1[0])))
                    po_h_7 = cutlass.Float32(_merge_o[(1024 + tid_wg)])
                    d_o[40] = cutlass.Float32(((d_o[40] * a0[0]) + (po_h_7 * b0[0])))
                    po_h_8 = cutlass.Float32(_merge_o[(1152 + tid_wg)])
                    d_o[41] = cutlass.Float32(((d_o[41] * a0[0]) + (po_h_8 * b0[0])))
                    po_h_9 = cutlass.Float32(_merge_o[(1280 + tid_wg)])
                    d_o[42] = cutlass.Float32(((d_o[42] * a1[0]) + (po_h_9 * b1[0])))
                    po_h_10 = cutlass.Float32(_merge_o[(1408 + tid_wg)])
                    d_o[43] = cutlass.Float32(((d_o[43] * a1[0]) + (po_h_10 * b1[0])))
                    po_h_11 = cutlass.Float32(_merge_o[(1536 + tid_wg)])
                    d_o[44] = cutlass.Float32(((d_o[44] * a0[0]) + (po_h_11 * b0[0])))
                    po_h_12 = cutlass.Float32(_merge_o[(1664 + tid_wg)])
                    d_o[45] = cutlass.Float32(((d_o[45] * a0[0]) + (po_h_12 * b0[0])))
                    po_h_13 = cutlass.Float32(_merge_o[(1792 + tid_wg)])
                    d_o[46] = cutlass.Float32(((d_o[46] * a1[0]) + (po_h_13 * b1[0])))
                    po_h_14 = cutlass.Float32(_merge_o[(1920 + tid_wg)])
                    d_o[47] = cutlass.Float32(((d_o[47] * a1[0]) + (po_h_14 * b1[0])))
                    po_h_15 = cutlass.Float32(_merge_o[(2048 + tid_wg)])
                    d_o[48] = cutlass.Float32(((d_o[48] * a0[0]) + (po_h_15 * b0[0])))
                    po_h_16 = cutlass.Float32(_merge_o[(2176 + tid_wg)])
                    d_o[49] = cutlass.Float32(((d_o[49] * a0[0]) + (po_h_16 * b0[0])))
                    po_h_17 = cutlass.Float32(_merge_o[(2304 + tid_wg)])
                    d_o[50] = cutlass.Float32(((d_o[50] * a1[0]) + (po_h_17 * b1[0])))
                    po_h_18 = cutlass.Float32(_merge_o[(2432 + tid_wg)])
                    d_o[51] = cutlass.Float32(((d_o[51] * a1[0]) + (po_h_18 * b1[0])))
                    po_h_19 = cutlass.Float32(_merge_o[(2560 + tid_wg)])
                    d_o[52] = cutlass.Float32(((d_o[52] * a0[0]) + (po_h_19 * b0[0])))
                    po_h_20 = cutlass.Float32(_merge_o[(2688 + tid_wg)])
                    d_o[53] = cutlass.Float32(((d_o[53] * a0[0]) + (po_h_20 * b0[0])))
                    po_h_21 = cutlass.Float32(_merge_o[(2816 + tid_wg)])
                    d_o[54] = cutlass.Float32(((d_o[54] * a1[0]) + (po_h_21 * b1[0])))
                    po_h_22 = cutlass.Float32(_merge_o[(2944 + tid_wg)])
                    d_o[55] = cutlass.Float32(((d_o[55] * a1[0]) + (po_h_22 * b1[0])))
                    po_h_23 = cutlass.Float32(_merge_o[(3072 + tid_wg)])
                    d_o[56] = cutlass.Float32(((d_o[56] * a0[0]) + (po_h_23 * b0[0])))
                    po_h_24 = cutlass.Float32(_merge_o[(3200 + tid_wg)])
                    d_o[57] = cutlass.Float32(((d_o[57] * a0[0]) + (po_h_24 * b0[0])))
                    po_h_25 = cutlass.Float32(_merge_o[(3328 + tid_wg)])
                    d_o[58] = cutlass.Float32(((d_o[58] * a1[0]) + (po_h_25 * b1[0])))
                    po_h_26 = cutlass.Float32(_merge_o[(3456 + tid_wg)])
                    d_o[59] = cutlass.Float32(((d_o[59] * a1[0]) + (po_h_26 * b1[0])))
                    po_h_27 = cutlass.Float32(_merge_o[(3584 + tid_wg)])
                    d_o[60] = cutlass.Float32(((d_o[60] * a0[0]) + (po_h_27 * b0[0])))
                    po_h_28 = cutlass.Float32(_merge_o[(3712 + tid_wg)])
                    d_o[61] = cutlass.Float32(((d_o[61] * a0[0]) + (po_h_28 * b0[0])))
                    po_h_29 = cutlass.Float32(_merge_o[(3840 + tid_wg)])
                    d_o[62] = cutlass.Float32(((d_o[62] * a1[0]) + (po_h_29 * b1[0])))
                    po_h_30 = cutlass.Float32(_merge_o[(3968 + tid_wg)])
                    d_o[63] = cutlass.Float32(((d_o[63] * a1[0]) + (po_h_30 * b1[0])))
            if (store_wg[0] != 0):
                _rcp_0 = cute.math.rcp(row_sum0[0], approx=True, ftz=True)
                _rcp_1 = cute.math.rcp(row_sum1[0], approx=True, ftz=True)
                d_o[0] = cutlass.Float32((d_o[0] * _rcp_0))
                d_o[1] = cutlass.Float32((d_o[1] * _rcp_0))
                d_o[4] = cutlass.Float32((d_o[4] * _rcp_0))
                d_o[5] = cutlass.Float32((d_o[5] * _rcp_0))
                d_o[8] = cutlass.Float32((d_o[8] * _rcp_0))
                d_o[9] = cutlass.Float32((d_o[9] * _rcp_0))
                d_o[12] = cutlass.Float32((d_o[12] * _rcp_0))
                d_o[13] = cutlass.Float32((d_o[13] * _rcp_0))
                d_o[16] = cutlass.Float32((d_o[16] * _rcp_0))
                d_o[17] = cutlass.Float32((d_o[17] * _rcp_0))
                d_o[20] = cutlass.Float32((d_o[20] * _rcp_0))
                d_o[21] = cutlass.Float32((d_o[21] * _rcp_0))
                d_o[24] = cutlass.Float32((d_o[24] * _rcp_0))
                d_o[25] = cutlass.Float32((d_o[25] * _rcp_0))
                d_o[28] = cutlass.Float32((d_o[28] * _rcp_0))
                d_o[29] = cutlass.Float32((d_o[29] * _rcp_0))
                d_o[32] = cutlass.Float32((d_o[32] * _rcp_0))
                d_o[33] = cutlass.Float32((d_o[33] * _rcp_0))
                d_o[36] = cutlass.Float32((d_o[36] * _rcp_0))
                d_o[37] = cutlass.Float32((d_o[37] * _rcp_0))
                d_o[40] = cutlass.Float32((d_o[40] * _rcp_0))
                d_o[41] = cutlass.Float32((d_o[41] * _rcp_0))
                d_o[44] = cutlass.Float32((d_o[44] * _rcp_0))
                d_o[45] = cutlass.Float32((d_o[45] * _rcp_0))
                d_o[48] = cutlass.Float32((d_o[48] * _rcp_0))
                d_o[49] = cutlass.Float32((d_o[49] * _rcp_0))
                d_o[52] = cutlass.Float32((d_o[52] * _rcp_0))
                d_o[53] = cutlass.Float32((d_o[53] * _rcp_0))
                d_o[56] = cutlass.Float32((d_o[56] * _rcp_0))
                d_o[57] = cutlass.Float32((d_o[57] * _rcp_0))
                d_o[60] = cutlass.Float32((d_o[60] * _rcp_0))
                d_o[61] = cutlass.Float32((d_o[61] * _rcp_0))
                d_o[2] = cutlass.Float32((d_o[2] * _rcp_1))
                d_o[3] = cutlass.Float32((d_o[3] * _rcp_1))
                d_o[6] = cutlass.Float32((d_o[6] * _rcp_1))
                d_o[7] = cutlass.Float32((d_o[7] * _rcp_1))
                d_o[10] = cutlass.Float32((d_o[10] * _rcp_1))
                d_o[11] = cutlass.Float32((d_o[11] * _rcp_1))
                d_o[14] = cutlass.Float32((d_o[14] * _rcp_1))
                d_o[15] = cutlass.Float32((d_o[15] * _rcp_1))
                d_o[18] = cutlass.Float32((d_o[18] * _rcp_1))
                d_o[19] = cutlass.Float32((d_o[19] * _rcp_1))
                d_o[22] = cutlass.Float32((d_o[22] * _rcp_1))
                d_o[23] = cutlass.Float32((d_o[23] * _rcp_1))
                d_o[26] = cutlass.Float32((d_o[26] * _rcp_1))
                d_o[27] = cutlass.Float32((d_o[27] * _rcp_1))
                d_o[30] = cutlass.Float32((d_o[30] * _rcp_1))
                d_o[31] = cutlass.Float32((d_o[31] * _rcp_1))
                d_o[34] = cutlass.Float32((d_o[34] * _rcp_1))
                d_o[35] = cutlass.Float32((d_o[35] * _rcp_1))
                d_o[38] = cutlass.Float32((d_o[38] * _rcp_1))
                d_o[39] = cutlass.Float32((d_o[39] * _rcp_1))
                d_o[42] = cutlass.Float32((d_o[42] * _rcp_1))
                d_o[43] = cutlass.Float32((d_o[43] * _rcp_1))
                d_o[46] = cutlass.Float32((d_o[46] * _rcp_1))
                d_o[47] = cutlass.Float32((d_o[47] * _rcp_1))
                d_o[50] = cutlass.Float32((d_o[50] * _rcp_1))
                d_o[51] = cutlass.Float32((d_o[51] * _rcp_1))
                d_o[54] = cutlass.Float32((d_o[54] * _rcp_1))
                d_o[55] = cutlass.Float32((d_o[55] * _rcp_1))
                d_o[58] = cutlass.Float32((d_o[58] * _rcp_1))
                d_o[59] = cutlass.Float32((d_o[59] * _rcp_1))
                d_o[62] = cutlass.Float32((d_o[62] * _rcp_1))
                d_o[63] = cutlass.Float32((d_o[63] * _rcp_1))
                qj = cutlass.Int32((lane & 3))
                qj1 = cutlass.Int32((qj & 1))
                qj2 = cutlass.Int32((qj & 2))
                o_vec = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base = cutlass.Int32((((head_m * seqlen_q) + (my_qb * 64)) * 128))
                m_local_r = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_64 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[0]), cutlass.Float32(d_o[1])))[1]), cutlass.Float32(((cutlass.Float32(d_o[0]), cutlass.Float32(d_o[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_64)
                _bf16x2_65 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[4]), cutlass.Float32(d_o[5])))[1]), cutlass.Float32(((cutlass.Float32(d_o[4]), cutlass.Float32(d_o[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_65)
                _bf16x2_66 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[8]), cutlass.Float32(d_o[9])))[1]), cutlass.Float32(((cutlass.Float32(d_o[8]), cutlass.Float32(d_o[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_66)
                _bf16x2_67 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[12]), cutlass.Float32(d_o[13])))[1]), cutlass.Float32(((cutlass.Float32(d_o[12]), cutlass.Float32(d_o[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_67)
                _shfl_xor_24 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_24)
                _shfl_xor_25 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_25)
                _shfl_xor_26 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_26)
                _shfl_xor_27 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_27)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_28 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_28)
                _shfl_xor_29 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_29)
                _shfl_xor_30 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_30)
                _shfl_xor_31 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_31)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off = cutlass.Int32(((o_row_base + (m_local_r * 128)) + (qj * 8)))
                _gmem_store_raw_48 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_48.ir_value(), O + o_off)
                _bf16x2_68 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[16]), cutlass.Float32(d_o[17])))[1]), cutlass.Float32(((cutlass.Float32(d_o[16]), cutlass.Float32(d_o[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_68)
                _bf16x2_69 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[20]), cutlass.Float32(d_o[21])))[1]), cutlass.Float32(((cutlass.Float32(d_o[20]), cutlass.Float32(d_o[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_69)
                _bf16x2_70 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[24]), cutlass.Float32(d_o[25])))[1]), cutlass.Float32(((cutlass.Float32(d_o[24]), cutlass.Float32(d_o[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_70)
                _bf16x2_71 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[28]), cutlass.Float32(d_o[29])))[1]), cutlass.Float32(((cutlass.Float32(d_o[28]), cutlass.Float32(d_o[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_71)
                _shfl_xor_32 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_32)
                _shfl_xor_33 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_33)
                _shfl_xor_34 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_34)
                _shfl_xor_35 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_35)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_36 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_36)
                _shfl_xor_37 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_37)
                _shfl_xor_38 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_38)
                _shfl_xor_39 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_39)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_0 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((4 + qj) * 8)))
                _gmem_store_raw_49 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_49.ir_value(), O + o_off_0)
                _bf16x2_72 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[32]), cutlass.Float32(d_o[33])))[1]), cutlass.Float32(((cutlass.Float32(d_o[32]), cutlass.Float32(d_o[33])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_72)
                _bf16x2_73 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[36]), cutlass.Float32(d_o[37])))[1]), cutlass.Float32(((cutlass.Float32(d_o[36]), cutlass.Float32(d_o[37])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_73)
                _bf16x2_74 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[40]), cutlass.Float32(d_o[41])))[1]), cutlass.Float32(((cutlass.Float32(d_o[40]), cutlass.Float32(d_o[41])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_74)
                _bf16x2_75 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[44]), cutlass.Float32(d_o[45])))[1]), cutlass.Float32(((cutlass.Float32(d_o[44]), cutlass.Float32(d_o[45])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_75)
                _shfl_xor_40 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_40)
                _shfl_xor_41 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_41)
                _shfl_xor_42 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_42)
                _shfl_xor_43 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_43)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_44 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_44)
                _shfl_xor_45 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_45)
                _shfl_xor_46 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_46)
                _shfl_xor_47 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_47)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_1 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((8 + qj) * 8)))
                _gmem_store_raw_50 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_50.ir_value(), O + o_off_1)
                _bf16x2_76 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[48]), cutlass.Float32(d_o[49])))[1]), cutlass.Float32(((cutlass.Float32(d_o[48]), cutlass.Float32(d_o[49])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_76)
                _bf16x2_77 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[52]), cutlass.Float32(d_o[53])))[1]), cutlass.Float32(((cutlass.Float32(d_o[52]), cutlass.Float32(d_o[53])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_77)
                _bf16x2_78 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[56]), cutlass.Float32(d_o[57])))[1]), cutlass.Float32(((cutlass.Float32(d_o[56]), cutlass.Float32(d_o[57])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_78)
                _bf16x2_79 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[60]), cutlass.Float32(d_o[61])))[1]), cutlass.Float32(((cutlass.Float32(d_o[60]), cutlass.Float32(d_o[61])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_79)
                _shfl_xor_48 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_48)
                _shfl_xor_49 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_49)
                _shfl_xor_50 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_50)
                _shfl_xor_51 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_51)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_52 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_52)
                _shfl_xor_53 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_53)
                _shfl_xor_54 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_54)
                _shfl_xor_55 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_55)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_2 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((12 + qj) * 8)))
                _gmem_store_raw_51 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_51.ir_value(), O + o_off_2)
                m_local_r_3 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_80 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[2]), cutlass.Float32(d_o[3])))[1]), cutlass.Float32(((cutlass.Float32(d_o[2]), cutlass.Float32(d_o[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_80)
                _bf16x2_81 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[6]), cutlass.Float32(d_o[7])))[1]), cutlass.Float32(((cutlass.Float32(d_o[6]), cutlass.Float32(d_o[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_81)
                _bf16x2_82 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[10]), cutlass.Float32(d_o[11])))[1]), cutlass.Float32(((cutlass.Float32(d_o[10]), cutlass.Float32(d_o[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_82)
                _bf16x2_83 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[14]), cutlass.Float32(d_o[15])))[1]), cutlass.Float32(((cutlass.Float32(d_o[14]), cutlass.Float32(d_o[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_83)
                _shfl_xor_56 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_56)
                _shfl_xor_57 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_57)
                _shfl_xor_58 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_58)
                _shfl_xor_59 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_59)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_60 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_60)
                _shfl_xor_61 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_61)
                _shfl_xor_62 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_62)
                _shfl_xor_63 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_63)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_4 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + (qj * 8)))
                _gmem_store_raw_52 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_52.ir_value(), O + o_off_4)
                _bf16x2_84 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[18]), cutlass.Float32(d_o[19])))[1]), cutlass.Float32(((cutlass.Float32(d_o[18]), cutlass.Float32(d_o[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_84)
                _bf16x2_85 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[22]), cutlass.Float32(d_o[23])))[1]), cutlass.Float32(((cutlass.Float32(d_o[22]), cutlass.Float32(d_o[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_85)
                _bf16x2_86 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[26]), cutlass.Float32(d_o[27])))[1]), cutlass.Float32(((cutlass.Float32(d_o[26]), cutlass.Float32(d_o[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_86)
                _bf16x2_87 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[30]), cutlass.Float32(d_o[31])))[1]), cutlass.Float32(((cutlass.Float32(d_o[30]), cutlass.Float32(d_o[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_87)
                _shfl_xor_64 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_64)
                _shfl_xor_65 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_65)
                _shfl_xor_66 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_66)
                _shfl_xor_67 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_67)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_68 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_68)
                _shfl_xor_69 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_69)
                _shfl_xor_70 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_70)
                _shfl_xor_71 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_71)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_5 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((4 + qj) * 8)))
                _gmem_store_raw_53 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_53.ir_value(), O + o_off_5)
                _bf16x2_88 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[34]), cutlass.Float32(d_o[35])))[1]), cutlass.Float32(((cutlass.Float32(d_o[34]), cutlass.Float32(d_o[35])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_88)
                _bf16x2_89 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[38]), cutlass.Float32(d_o[39])))[1]), cutlass.Float32(((cutlass.Float32(d_o[38]), cutlass.Float32(d_o[39])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_89)
                _bf16x2_90 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[42]), cutlass.Float32(d_o[43])))[1]), cutlass.Float32(((cutlass.Float32(d_o[42]), cutlass.Float32(d_o[43])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_90)
                _bf16x2_91 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[46]), cutlass.Float32(d_o[47])))[1]), cutlass.Float32(((cutlass.Float32(d_o[46]), cutlass.Float32(d_o[47])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_91)
                _shfl_xor_72 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_72)
                _shfl_xor_73 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_73)
                _shfl_xor_74 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_74)
                _shfl_xor_75 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_75)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_76 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_76)
                _shfl_xor_77 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_77)
                _shfl_xor_78 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_78)
                _shfl_xor_79 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_79)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_6 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((8 + qj) * 8)))
                _gmem_store_raw_54 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_54.ir_value(), O + o_off_6)
                _bf16x2_92 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[50]), cutlass.Float32(d_o[51])))[1]), cutlass.Float32(((cutlass.Float32(d_o[50]), cutlass.Float32(d_o[51])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_92)
                _bf16x2_93 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[54]), cutlass.Float32(d_o[55])))[1]), cutlass.Float32(((cutlass.Float32(d_o[54]), cutlass.Float32(d_o[55])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_93)
                _bf16x2_94 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[58]), cutlass.Float32(d_o[59])))[1]), cutlass.Float32(((cutlass.Float32(d_o[58]), cutlass.Float32(d_o[59])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_94)
                _bf16x2_95 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[62]), cutlass.Float32(d_o[63])))[1]), cutlass.Float32(((cutlass.Float32(d_o[62]), cutlass.Float32(d_o[63])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_95)
                _shfl_xor_80 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_80)
                _shfl_xor_81 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_81)
                _shfl_xor_82 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_82)
                _shfl_xor_83 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_83)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_84 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_84)
                _shfl_xor_85 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_85)
                _shfl_xor_86 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_86)
                _shfl_xor_87 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_87)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_7 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((12 + qj) * 8)))
                _gmem_store_raw_55 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_55.ir_value(), O + o_off_7)
            cute.arch.sync_warp()
            if prims.elect_sync():
                cute.arch.mbarrier_arrive(meta_empty_addr + slot_m)
    if _cake_ldparam_b64(_hdr__base + 0, 'u64') != cutlass.Uint64(hdr__slot_0):
        cutlass_llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string='trap;',
            constraints='~{memory}',
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        )
    if _cake_ldparam_b64(_hdr__base + 3448, 'u64') != cutlass.Uint64(hdr__slot_431):
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
def launch_vsa_sm90_bf16_fwd(Q: cute.Tensor, _cake_tma_Q_dim_0: cutlass.Int64, _cake_tma_Q_dim_1: cutlass.Int64, _cake_tma_Q_dim_2: cutlass.Int64, _cake_tma_Q_stride16_0: cutlass.Int64, _cake_tma_Q_stride16_1: cutlass.Int64, K: cute.Tensor, _cake_tma_K_dim_0: cutlass.Int64, _cake_tma_K_dim_1: cutlass.Int64, _cake_tma_K_dim_2: cutlass.Int64, _cake_tma_K_stride16_0: cutlass.Int64, _cake_tma_K_stride16_1: cutlass.Int64, Vt: cute.Tensor, _cake_tma_Vt_dim_0: cutlass.Int64, _cake_tma_Vt_dim_1: cutlass.Int64, _cake_tma_Vt_dim_2: cutlass.Int64, _cake_tma_Vt_dim_3: cutlass.Int64, _cake_tma_Vt_stride16_0: cutlass.Int64, _cake_tma_Vt_stride16_1: cutlass.Int64, _cake_tma_Vt_stride16_2: cutlass.Int64, O: cute.Tensor, meta: cute.Tensor, hdr__slot_0: cutlass.Uint64, hdr__slot_1: cutlass.Uint64, hdr__slot_2: cutlass.Uint64, hdr__slot_3: cutlass.Uint64, hdr__slot_4: cutlass.Uint64, hdr__slot_5: cutlass.Uint64, hdr__slot_6: cutlass.Uint64, hdr__slot_7: cutlass.Uint64, hdr__slot_8: cutlass.Uint64, hdr__slot_9: cutlass.Uint64, hdr__slot_10: cutlass.Uint64, hdr__slot_11: cutlass.Uint64, hdr__slot_12: cutlass.Uint64, hdr__slot_13: cutlass.Uint64, hdr__slot_14: cutlass.Uint64, hdr__slot_15: cutlass.Uint64, hdr__slot_16: cutlass.Uint64, hdr__slot_17: cutlass.Uint64, hdr__slot_18: cutlass.Uint64, hdr__slot_19: cutlass.Uint64, hdr__slot_20: cutlass.Uint64, hdr__slot_21: cutlass.Uint64, hdr__slot_22: cutlass.Uint64, hdr__slot_23: cutlass.Uint64, hdr__slot_24: cutlass.Uint64, hdr__slot_25: cutlass.Uint64, hdr__slot_26: cutlass.Uint64, hdr__slot_27: cutlass.Uint64, hdr__slot_28: cutlass.Uint64, hdr__slot_29: cutlass.Uint64, hdr__slot_30: cutlass.Uint64, hdr__slot_31: cutlass.Uint64, hdr__slot_32: cutlass.Uint64, hdr__slot_33: cutlass.Uint64, hdr__slot_34: cutlass.Uint64, hdr__slot_35: cutlass.Uint64, hdr__slot_36: cutlass.Uint64, hdr__slot_37: cutlass.Uint64, hdr__slot_38: cutlass.Uint64, hdr__slot_39: cutlass.Uint64, hdr__slot_40: cutlass.Uint64, hdr__slot_41: cutlass.Uint64, hdr__slot_42: cutlass.Uint64, hdr__slot_43: cutlass.Uint64, hdr__slot_44: cutlass.Uint64, hdr__slot_45: cutlass.Uint64, hdr__slot_46: cutlass.Uint64, hdr__slot_47: cutlass.Uint64, hdr__slot_48: cutlass.Uint64, hdr__slot_49: cutlass.Uint64, hdr__slot_50: cutlass.Uint64, hdr__slot_51: cutlass.Uint64, hdr__slot_52: cutlass.Uint64, hdr__slot_53: cutlass.Uint64, hdr__slot_54: cutlass.Uint64, hdr__slot_55: cutlass.Uint64, hdr__slot_56: cutlass.Uint64, hdr__slot_57: cutlass.Uint64, hdr__slot_58: cutlass.Uint64, hdr__slot_59: cutlass.Uint64, hdr__slot_60: cutlass.Uint64, hdr__slot_61: cutlass.Uint64, hdr__slot_62: cutlass.Uint64, hdr__slot_63: cutlass.Uint64, hdr__slot_64: cutlass.Uint64, hdr__slot_65: cutlass.Uint64, hdr__slot_66: cutlass.Uint64, hdr__slot_67: cutlass.Uint64, hdr__slot_68: cutlass.Uint64, hdr__slot_69: cutlass.Uint64, hdr__slot_70: cutlass.Uint64, hdr__slot_71: cutlass.Uint64, hdr__slot_72: cutlass.Uint64, hdr__slot_73: cutlass.Uint64, hdr__slot_74: cutlass.Uint64, hdr__slot_75: cutlass.Uint64, hdr__slot_76: cutlass.Uint64, hdr__slot_77: cutlass.Uint64, hdr__slot_78: cutlass.Uint64, hdr__slot_79: cutlass.Uint64, hdr__slot_80: cutlass.Uint64, hdr__slot_81: cutlass.Uint64, hdr__slot_82: cutlass.Uint64, hdr__slot_83: cutlass.Uint64, hdr__slot_84: cutlass.Uint64, hdr__slot_85: cutlass.Uint64, hdr__slot_86: cutlass.Uint64, hdr__slot_87: cutlass.Uint64, hdr__slot_88: cutlass.Uint64, hdr__slot_89: cutlass.Uint64, hdr__slot_90: cutlass.Uint64, hdr__slot_91: cutlass.Uint64, hdr__slot_92: cutlass.Uint64, hdr__slot_93: cutlass.Uint64, hdr__slot_94: cutlass.Uint64, hdr__slot_95: cutlass.Uint64, hdr__slot_96: cutlass.Uint64, hdr__slot_97: cutlass.Uint64, hdr__slot_98: cutlass.Uint64, hdr__slot_99: cutlass.Uint64, hdr__slot_100: cutlass.Uint64, hdr__slot_101: cutlass.Uint64, hdr__slot_102: cutlass.Uint64, hdr__slot_103: cutlass.Uint64, hdr__slot_104: cutlass.Uint64, hdr__slot_105: cutlass.Uint64, hdr__slot_106: cutlass.Uint64, hdr__slot_107: cutlass.Uint64, hdr__slot_108: cutlass.Uint64, hdr__slot_109: cutlass.Uint64, hdr__slot_110: cutlass.Uint64, hdr__slot_111: cutlass.Uint64, hdr__slot_112: cutlass.Uint64, hdr__slot_113: cutlass.Uint64, hdr__slot_114: cutlass.Uint64, hdr__slot_115: cutlass.Uint64, hdr__slot_116: cutlass.Uint64, hdr__slot_117: cutlass.Uint64, hdr__slot_118: cutlass.Uint64, hdr__slot_119: cutlass.Uint64, hdr__slot_120: cutlass.Uint64, hdr__slot_121: cutlass.Uint64, hdr__slot_122: cutlass.Uint64, hdr__slot_123: cutlass.Uint64, hdr__slot_124: cutlass.Uint64, hdr__slot_125: cutlass.Uint64, hdr__slot_126: cutlass.Uint64, hdr__slot_127: cutlass.Uint64, hdr__slot_128: cutlass.Uint64, hdr__slot_129: cutlass.Uint64, hdr__slot_130: cutlass.Uint64, hdr__slot_131: cutlass.Uint64, hdr__slot_132: cutlass.Uint64, hdr__slot_133: cutlass.Uint64, hdr__slot_134: cutlass.Uint64, hdr__slot_135: cutlass.Uint64, hdr__slot_136: cutlass.Uint64, hdr__slot_137: cutlass.Uint64, hdr__slot_138: cutlass.Uint64, hdr__slot_139: cutlass.Uint64, hdr__slot_140: cutlass.Uint64, hdr__slot_141: cutlass.Uint64, hdr__slot_142: cutlass.Uint64, hdr__slot_143: cutlass.Uint64, hdr__slot_144: cutlass.Uint64, hdr__slot_145: cutlass.Uint64, hdr__slot_146: cutlass.Uint64, hdr__slot_147: cutlass.Uint64, hdr__slot_148: cutlass.Uint64, hdr__slot_149: cutlass.Uint64, hdr__slot_150: cutlass.Uint64, hdr__slot_151: cutlass.Uint64, hdr__slot_152: cutlass.Uint64, hdr__slot_153: cutlass.Uint64, hdr__slot_154: cutlass.Uint64, hdr__slot_155: cutlass.Uint64, hdr__slot_156: cutlass.Uint64, hdr__slot_157: cutlass.Uint64, hdr__slot_158: cutlass.Uint64, hdr__slot_159: cutlass.Uint64, hdr__slot_160: cutlass.Uint64, hdr__slot_161: cutlass.Uint64, hdr__slot_162: cutlass.Uint64, hdr__slot_163: cutlass.Uint64, hdr__slot_164: cutlass.Uint64, hdr__slot_165: cutlass.Uint64, hdr__slot_166: cutlass.Uint64, hdr__slot_167: cutlass.Uint64, hdr__slot_168: cutlass.Uint64, hdr__slot_169: cutlass.Uint64, hdr__slot_170: cutlass.Uint64, hdr__slot_171: cutlass.Uint64, hdr__slot_172: cutlass.Uint64, hdr__slot_173: cutlass.Uint64, hdr__slot_174: cutlass.Uint64, hdr__slot_175: cutlass.Uint64, hdr__slot_176: cutlass.Uint64, hdr__slot_177: cutlass.Uint64, hdr__slot_178: cutlass.Uint64, hdr__slot_179: cutlass.Uint64, hdr__slot_180: cutlass.Uint64, hdr__slot_181: cutlass.Uint64, hdr__slot_182: cutlass.Uint64, hdr__slot_183: cutlass.Uint64, hdr__slot_184: cutlass.Uint64, hdr__slot_185: cutlass.Uint64, hdr__slot_186: cutlass.Uint64, hdr__slot_187: cutlass.Uint64, hdr__slot_188: cutlass.Uint64, hdr__slot_189: cutlass.Uint64, hdr__slot_190: cutlass.Uint64, hdr__slot_191: cutlass.Uint64, hdr__slot_192: cutlass.Uint64, hdr__slot_193: cutlass.Uint64, hdr__slot_194: cutlass.Uint64, hdr__slot_195: cutlass.Uint64, hdr__slot_196: cutlass.Uint64, hdr__slot_197: cutlass.Uint64, hdr__slot_198: cutlass.Uint64, hdr__slot_199: cutlass.Uint64, hdr__slot_200: cutlass.Uint64, hdr__slot_201: cutlass.Uint64, hdr__slot_202: cutlass.Uint64, hdr__slot_203: cutlass.Uint64, hdr__slot_204: cutlass.Uint64, hdr__slot_205: cutlass.Uint64, hdr__slot_206: cutlass.Uint64, hdr__slot_207: cutlass.Uint64, hdr__slot_208: cutlass.Uint64, hdr__slot_209: cutlass.Uint64, hdr__slot_210: cutlass.Uint64, hdr__slot_211: cutlass.Uint64, hdr__slot_212: cutlass.Uint64, hdr__slot_213: cutlass.Uint64, hdr__slot_214: cutlass.Uint64, hdr__slot_215: cutlass.Uint64, hdr__slot_216: cutlass.Uint64, hdr__slot_217: cutlass.Uint64, hdr__slot_218: cutlass.Uint64, hdr__slot_219: cutlass.Uint64, hdr__slot_220: cutlass.Uint64, hdr__slot_221: cutlass.Uint64, hdr__slot_222: cutlass.Uint64, hdr__slot_223: cutlass.Uint64, hdr__slot_224: cutlass.Uint64, hdr__slot_225: cutlass.Uint64, hdr__slot_226: cutlass.Uint64, hdr__slot_227: cutlass.Uint64, hdr__slot_228: cutlass.Uint64, hdr__slot_229: cutlass.Uint64, hdr__slot_230: cutlass.Uint64, hdr__slot_231: cutlass.Uint64, hdr__slot_232: cutlass.Uint64, hdr__slot_233: cutlass.Uint64, hdr__slot_234: cutlass.Uint64, hdr__slot_235: cutlass.Uint64, hdr__slot_236: cutlass.Uint64, hdr__slot_237: cutlass.Uint64, hdr__slot_238: cutlass.Uint64, hdr__slot_239: cutlass.Uint64, hdr__slot_240: cutlass.Uint64, hdr__slot_241: cutlass.Uint64, hdr__slot_242: cutlass.Uint64, hdr__slot_243: cutlass.Uint64, hdr__slot_244: cutlass.Uint64, hdr__slot_245: cutlass.Uint64, hdr__slot_246: cutlass.Uint64, hdr__slot_247: cutlass.Uint64, hdr__slot_248: cutlass.Uint64, hdr__slot_249: cutlass.Uint64, hdr__slot_250: cutlass.Uint64, hdr__slot_251: cutlass.Uint64, hdr__slot_252: cutlass.Uint64, hdr__slot_253: cutlass.Uint64, hdr__slot_254: cutlass.Uint64, hdr__slot_255: cutlass.Uint64, hdr__slot_256: cutlass.Uint64, hdr__slot_257: cutlass.Uint64, hdr__slot_258: cutlass.Uint64, hdr__slot_259: cutlass.Uint64, hdr__slot_260: cutlass.Uint64, hdr__slot_261: cutlass.Uint64, hdr__slot_262: cutlass.Uint64, hdr__slot_263: cutlass.Uint64, hdr__slot_264: cutlass.Uint64, hdr__slot_265: cutlass.Uint64, hdr__slot_266: cutlass.Uint64, hdr__slot_267: cutlass.Uint64, hdr__slot_268: cutlass.Uint64, hdr__slot_269: cutlass.Uint64, hdr__slot_270: cutlass.Uint64, hdr__slot_271: cutlass.Uint64, hdr__slot_272: cutlass.Uint64, hdr__slot_273: cutlass.Uint64, hdr__slot_274: cutlass.Uint64, hdr__slot_275: cutlass.Uint64, hdr__slot_276: cutlass.Uint64, hdr__slot_277: cutlass.Uint64, hdr__slot_278: cutlass.Uint64, hdr__slot_279: cutlass.Uint64, hdr__slot_280: cutlass.Uint64, hdr__slot_281: cutlass.Uint64, hdr__slot_282: cutlass.Uint64, hdr__slot_283: cutlass.Uint64, hdr__slot_284: cutlass.Uint64, hdr__slot_285: cutlass.Uint64, hdr__slot_286: cutlass.Uint64, hdr__slot_287: cutlass.Uint64, hdr__slot_288: cutlass.Uint64, hdr__slot_289: cutlass.Uint64, hdr__slot_290: cutlass.Uint64, hdr__slot_291: cutlass.Uint64, hdr__slot_292: cutlass.Uint64, hdr__slot_293: cutlass.Uint64, hdr__slot_294: cutlass.Uint64, hdr__slot_295: cutlass.Uint64, hdr__slot_296: cutlass.Uint64, hdr__slot_297: cutlass.Uint64, hdr__slot_298: cutlass.Uint64, hdr__slot_299: cutlass.Uint64, hdr__slot_300: cutlass.Uint64, hdr__slot_301: cutlass.Uint64, hdr__slot_302: cutlass.Uint64, hdr__slot_303: cutlass.Uint64, hdr__slot_304: cutlass.Uint64, hdr__slot_305: cutlass.Uint64, hdr__slot_306: cutlass.Uint64, hdr__slot_307: cutlass.Uint64, hdr__slot_308: cutlass.Uint64, hdr__slot_309: cutlass.Uint64, hdr__slot_310: cutlass.Uint64, hdr__slot_311: cutlass.Uint64, hdr__slot_312: cutlass.Uint64, hdr__slot_313: cutlass.Uint64, hdr__slot_314: cutlass.Uint64, hdr__slot_315: cutlass.Uint64, hdr__slot_316: cutlass.Uint64, hdr__slot_317: cutlass.Uint64, hdr__slot_318: cutlass.Uint64, hdr__slot_319: cutlass.Uint64, hdr__slot_320: cutlass.Uint64, hdr__slot_321: cutlass.Uint64, hdr__slot_322: cutlass.Uint64, hdr__slot_323: cutlass.Uint64, hdr__slot_324: cutlass.Uint64, hdr__slot_325: cutlass.Uint64, hdr__slot_326: cutlass.Uint64, hdr__slot_327: cutlass.Uint64, hdr__slot_328: cutlass.Uint64, hdr__slot_329: cutlass.Uint64, hdr__slot_330: cutlass.Uint64, hdr__slot_331: cutlass.Uint64, hdr__slot_332: cutlass.Uint64, hdr__slot_333: cutlass.Uint64, hdr__slot_334: cutlass.Uint64, hdr__slot_335: cutlass.Uint64, hdr__slot_336: cutlass.Uint64, hdr__slot_337: cutlass.Uint64, hdr__slot_338: cutlass.Uint64, hdr__slot_339: cutlass.Uint64, hdr__slot_340: cutlass.Uint64, hdr__slot_341: cutlass.Uint64, hdr__slot_342: cutlass.Uint64, hdr__slot_343: cutlass.Uint64, hdr__slot_344: cutlass.Uint64, hdr__slot_345: cutlass.Uint64, hdr__slot_346: cutlass.Uint64, hdr__slot_347: cutlass.Uint64, hdr__slot_348: cutlass.Uint64, hdr__slot_349: cutlass.Uint64, hdr__slot_350: cutlass.Uint64, hdr__slot_351: cutlass.Uint64, hdr__slot_352: cutlass.Uint64, hdr__slot_353: cutlass.Uint64, hdr__slot_354: cutlass.Uint64, hdr__slot_355: cutlass.Uint64, hdr__slot_356: cutlass.Uint64, hdr__slot_357: cutlass.Uint64, hdr__slot_358: cutlass.Uint64, hdr__slot_359: cutlass.Uint64, hdr__slot_360: cutlass.Uint64, hdr__slot_361: cutlass.Uint64, hdr__slot_362: cutlass.Uint64, hdr__slot_363: cutlass.Uint64, hdr__slot_364: cutlass.Uint64, hdr__slot_365: cutlass.Uint64, hdr__slot_366: cutlass.Uint64, hdr__slot_367: cutlass.Uint64, hdr__slot_368: cutlass.Uint64, hdr__slot_369: cutlass.Uint64, hdr__slot_370: cutlass.Uint64, hdr__slot_371: cutlass.Uint64, hdr__slot_372: cutlass.Uint64, hdr__slot_373: cutlass.Uint64, hdr__slot_374: cutlass.Uint64, hdr__slot_375: cutlass.Uint64, hdr__slot_376: cutlass.Uint64, hdr__slot_377: cutlass.Uint64, hdr__slot_378: cutlass.Uint64, hdr__slot_379: cutlass.Uint64, hdr__slot_380: cutlass.Uint64, hdr__slot_381: cutlass.Uint64, hdr__slot_382: cutlass.Uint64, hdr__slot_383: cutlass.Uint64, hdr__slot_384: cutlass.Uint64, hdr__slot_385: cutlass.Uint64, hdr__slot_386: cutlass.Uint64, hdr__slot_387: cutlass.Uint64, hdr__slot_388: cutlass.Uint64, hdr__slot_389: cutlass.Uint64, hdr__slot_390: cutlass.Uint64, hdr__slot_391: cutlass.Uint64, hdr__slot_392: cutlass.Uint64, hdr__slot_393: cutlass.Uint64, hdr__slot_394: cutlass.Uint64, hdr__slot_395: cutlass.Uint64, hdr__slot_396: cutlass.Uint64, hdr__slot_397: cutlass.Uint64, hdr__slot_398: cutlass.Uint64, hdr__slot_399: cutlass.Uint64, hdr__slot_400: cutlass.Uint64, hdr__slot_401: cutlass.Uint64, hdr__slot_402: cutlass.Uint64, hdr__slot_403: cutlass.Uint64, hdr__slot_404: cutlass.Uint64, hdr__slot_405: cutlass.Uint64, hdr__slot_406: cutlass.Uint64, hdr__slot_407: cutlass.Uint64, hdr__slot_408: cutlass.Uint64, hdr__slot_409: cutlass.Uint64, hdr__slot_410: cutlass.Uint64, hdr__slot_411: cutlass.Uint64, hdr__slot_412: cutlass.Uint64, hdr__slot_413: cutlass.Uint64, hdr__slot_414: cutlass.Uint64, hdr__slot_415: cutlass.Uint64, hdr__slot_416: cutlass.Uint64, hdr__slot_417: cutlass.Uint64, hdr__slot_418: cutlass.Uint64, hdr__slot_419: cutlass.Uint64, hdr__slot_420: cutlass.Uint64, hdr__slot_421: cutlass.Uint64, hdr__slot_422: cutlass.Uint64, hdr__slot_423: cutlass.Uint64, hdr__slot_424: cutlass.Uint64, hdr__slot_425: cutlass.Uint64, hdr__slot_426: cutlass.Uint64, hdr__slot_427: cutlass.Uint64, hdr__slot_428: cutlass.Uint64, hdr__slot_429: cutlass.Uint64, hdr__slot_430: cutlass.Uint64, hdr__slot_431: cutlass.Uint64, tile_stride: cutlass.Int32, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32, dbg: cute.Tensor, tl: cute.Tensor, grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32, stream: cuda.CUstream):
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
        [64, 64, 1],
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
    kernel_vsa_sm90_bf16_fwd(_tma_Q, _tma_K, _tma_Vt, hdr__slot_0, hdr__slot_1, hdr__slot_2, hdr__slot_3, hdr__slot_4, hdr__slot_5, hdr__slot_6, hdr__slot_7, hdr__slot_8, hdr__slot_9, hdr__slot_10, hdr__slot_11, hdr__slot_12, hdr__slot_13, hdr__slot_14, hdr__slot_15, hdr__slot_16, hdr__slot_17, hdr__slot_18, hdr__slot_19, hdr__slot_20, hdr__slot_21, hdr__slot_22, hdr__slot_23, hdr__slot_24, hdr__slot_25, hdr__slot_26, hdr__slot_27, hdr__slot_28, hdr__slot_29, hdr__slot_30, hdr__slot_31, hdr__slot_32, hdr__slot_33, hdr__slot_34, hdr__slot_35, hdr__slot_36, hdr__slot_37, hdr__slot_38, hdr__slot_39, hdr__slot_40, hdr__slot_41, hdr__slot_42, hdr__slot_43, hdr__slot_44, hdr__slot_45, hdr__slot_46, hdr__slot_47, hdr__slot_48, hdr__slot_49, hdr__slot_50, hdr__slot_51, hdr__slot_52, hdr__slot_53, hdr__slot_54, hdr__slot_55, hdr__slot_56, hdr__slot_57, hdr__slot_58, hdr__slot_59, hdr__slot_60, hdr__slot_61, hdr__slot_62, hdr__slot_63, hdr__slot_64, hdr__slot_65, hdr__slot_66, hdr__slot_67, hdr__slot_68, hdr__slot_69, hdr__slot_70, hdr__slot_71, hdr__slot_72, hdr__slot_73, hdr__slot_74, hdr__slot_75, hdr__slot_76, hdr__slot_77, hdr__slot_78, hdr__slot_79, hdr__slot_80, hdr__slot_81, hdr__slot_82, hdr__slot_83, hdr__slot_84, hdr__slot_85, hdr__slot_86, hdr__slot_87, hdr__slot_88, hdr__slot_89, hdr__slot_90, hdr__slot_91, hdr__slot_92, hdr__slot_93, hdr__slot_94, hdr__slot_95, hdr__slot_96, hdr__slot_97, hdr__slot_98, hdr__slot_99, hdr__slot_100, hdr__slot_101, hdr__slot_102, hdr__slot_103, hdr__slot_104, hdr__slot_105, hdr__slot_106, hdr__slot_107, hdr__slot_108, hdr__slot_109, hdr__slot_110, hdr__slot_111, hdr__slot_112, hdr__slot_113, hdr__slot_114, hdr__slot_115, hdr__slot_116, hdr__slot_117, hdr__slot_118, hdr__slot_119, hdr__slot_120, hdr__slot_121, hdr__slot_122, hdr__slot_123, hdr__slot_124, hdr__slot_125, hdr__slot_126, hdr__slot_127, hdr__slot_128, hdr__slot_129, hdr__slot_130, hdr__slot_131, hdr__slot_132, hdr__slot_133, hdr__slot_134, hdr__slot_135, hdr__slot_136, hdr__slot_137, hdr__slot_138, hdr__slot_139, hdr__slot_140, hdr__slot_141, hdr__slot_142, hdr__slot_143, hdr__slot_144, hdr__slot_145, hdr__slot_146, hdr__slot_147, hdr__slot_148, hdr__slot_149, hdr__slot_150, hdr__slot_151, hdr__slot_152, hdr__slot_153, hdr__slot_154, hdr__slot_155, hdr__slot_156, hdr__slot_157, hdr__slot_158, hdr__slot_159, hdr__slot_160, hdr__slot_161, hdr__slot_162, hdr__slot_163, hdr__slot_164, hdr__slot_165, hdr__slot_166, hdr__slot_167, hdr__slot_168, hdr__slot_169, hdr__slot_170, hdr__slot_171, hdr__slot_172, hdr__slot_173, hdr__slot_174, hdr__slot_175, hdr__slot_176, hdr__slot_177, hdr__slot_178, hdr__slot_179, hdr__slot_180, hdr__slot_181, hdr__slot_182, hdr__slot_183, hdr__slot_184, hdr__slot_185, hdr__slot_186, hdr__slot_187, hdr__slot_188, hdr__slot_189, hdr__slot_190, hdr__slot_191, hdr__slot_192, hdr__slot_193, hdr__slot_194, hdr__slot_195, hdr__slot_196, hdr__slot_197, hdr__slot_198, hdr__slot_199, hdr__slot_200, hdr__slot_201, hdr__slot_202, hdr__slot_203, hdr__slot_204, hdr__slot_205, hdr__slot_206, hdr__slot_207, hdr__slot_208, hdr__slot_209, hdr__slot_210, hdr__slot_211, hdr__slot_212, hdr__slot_213, hdr__slot_214, hdr__slot_215, hdr__slot_216, hdr__slot_217, hdr__slot_218, hdr__slot_219, hdr__slot_220, hdr__slot_221, hdr__slot_222, hdr__slot_223, hdr__slot_224, hdr__slot_225, hdr__slot_226, hdr__slot_227, hdr__slot_228, hdr__slot_229, hdr__slot_230, hdr__slot_231, hdr__slot_232, hdr__slot_233, hdr__slot_234, hdr__slot_235, hdr__slot_236, hdr__slot_237, hdr__slot_238, hdr__slot_239, hdr__slot_240, hdr__slot_241, hdr__slot_242, hdr__slot_243, hdr__slot_244, hdr__slot_245, hdr__slot_246, hdr__slot_247, hdr__slot_248, hdr__slot_249, hdr__slot_250, hdr__slot_251, hdr__slot_252, hdr__slot_253, hdr__slot_254, hdr__slot_255, hdr__slot_256, hdr__slot_257, hdr__slot_258, hdr__slot_259, hdr__slot_260, hdr__slot_261, hdr__slot_262, hdr__slot_263, hdr__slot_264, hdr__slot_265, hdr__slot_266, hdr__slot_267, hdr__slot_268, hdr__slot_269, hdr__slot_270, hdr__slot_271, hdr__slot_272, hdr__slot_273, hdr__slot_274, hdr__slot_275, hdr__slot_276, hdr__slot_277, hdr__slot_278, hdr__slot_279, hdr__slot_280, hdr__slot_281, hdr__slot_282, hdr__slot_283, hdr__slot_284, hdr__slot_285, hdr__slot_286, hdr__slot_287, hdr__slot_288, hdr__slot_289, hdr__slot_290, hdr__slot_291, hdr__slot_292, hdr__slot_293, hdr__slot_294, hdr__slot_295, hdr__slot_296, hdr__slot_297, hdr__slot_298, hdr__slot_299, hdr__slot_300, hdr__slot_301, hdr__slot_302, hdr__slot_303, hdr__slot_304, hdr__slot_305, hdr__slot_306, hdr__slot_307, hdr__slot_308, hdr__slot_309, hdr__slot_310, hdr__slot_311, hdr__slot_312, hdr__slot_313, hdr__slot_314, hdr__slot_315, hdr__slot_316, hdr__slot_317, hdr__slot_318, hdr__slot_319, hdr__slot_320, hdr__slot_321, hdr__slot_322, hdr__slot_323, hdr__slot_324, hdr__slot_325, hdr__slot_326, hdr__slot_327, hdr__slot_328, hdr__slot_329, hdr__slot_330, hdr__slot_331, hdr__slot_332, hdr__slot_333, hdr__slot_334, hdr__slot_335, hdr__slot_336, hdr__slot_337, hdr__slot_338, hdr__slot_339, hdr__slot_340, hdr__slot_341, hdr__slot_342, hdr__slot_343, hdr__slot_344, hdr__slot_345, hdr__slot_346, hdr__slot_347, hdr__slot_348, hdr__slot_349, hdr__slot_350, hdr__slot_351, hdr__slot_352, hdr__slot_353, hdr__slot_354, hdr__slot_355, hdr__slot_356, hdr__slot_357, hdr__slot_358, hdr__slot_359, hdr__slot_360, hdr__slot_361, hdr__slot_362, hdr__slot_363, hdr__slot_364, hdr__slot_365, hdr__slot_366, hdr__slot_367, hdr__slot_368, hdr__slot_369, hdr__slot_370, hdr__slot_371, hdr__slot_372, hdr__slot_373, hdr__slot_374, hdr__slot_375, hdr__slot_376, hdr__slot_377, hdr__slot_378, hdr__slot_379, hdr__slot_380, hdr__slot_381, hdr__slot_382, hdr__slot_383, hdr__slot_384, hdr__slot_385, hdr__slot_386, hdr__slot_387, hdr__slot_388, hdr__slot_389, hdr__slot_390, hdr__slot_391, hdr__slot_392, hdr__slot_393, hdr__slot_394, hdr__slot_395, hdr__slot_396, hdr__slot_397, hdr__slot_398, hdr__slot_399, hdr__slot_400, hdr__slot_401, hdr__slot_402, hdr__slot_403, hdr__slot_404, hdr__slot_405, hdr__slot_406, hdr__slot_407, hdr__slot_408, hdr__slot_409, hdr__slot_410, hdr__slot_411, hdr__slot_412, hdr__slot_413, hdr__slot_414, hdr__slot_415, hdr__slot_416, hdr__slot_417, hdr__slot_418, hdr__slot_419, hdr__slot_420, hdr__slot_421, hdr__slot_422, hdr__slot_423, hdr__slot_424, hdr__slot_425, hdr__slot_426, hdr__slot_427, hdr__slot_428, hdr__slot_429, hdr__slot_430, hdr__slot_431, O.iterator, meta.iterator, tile_stride, seqlen_q, seqlen_k, scale_log2, dbg.iterator, tl.iterator).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(384, 1, 1),
        smem=232064,
        min_blocks_per_mp=1,
        stream=stream,
    )

def compile_program():
    return cute.compile(launch_vsa_sm90_bf16_fwd,
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
        make_fake_tensor(cutlass.Int32, (cute.sym_int64(symbol='meta'),), (1,), assumed_align=16),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
        cutlass.Uint64(0),
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
        cutlass.Int32(0),
        cutlass.Float32(0),
        make_fake_tensor(cutlass.Int32, (cute.sym_int64(symbol='dbg'),), (1,), assumed_align=16),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='tl'),), (1,), assumed_align=16),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_90a',
    )
