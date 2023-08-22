import os
import tvm
from tvm.script import ir as I, relax as R, tir as T


def FlashInferIRModuleGen(
    dtype_in: str, dtype_out: str, rotary_mode: int = 0, rope_inv_scale: float = 1.0
) -> tvm.IRModule:
    @I.ir_module
    class FlashInferIRModule:
        @R.function
        def decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype_in),
            k: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype_in),
            v: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype_in),
            tmp: R.Tensor((2 * 1024 * 1024,), "float32"),
        ) -> R.Tensor(("seq_len", "num_heads", "head_dim"), dtype_out):
            num_heads = T.int64()
            head_dim = T.int64()
            with R.dataflow():
                o = R.call_dps_packed(
                    "FlashInferSingleDecodeWithKVCache",
                    (q, k, v, tmp, rotary_mode, rope_inv_scale),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype_out),
                )
                R.output(o)
            return o

    ext_mod = tvm.runtime.load_static_library(
        os.path.dirname(os.path.realpath(__file__))
        + "/../build/CMakeFiles/tvm_binding.dir/src/tvm_wrapper.cu.o",
        ["FlashInferSingleDecodeWithKVCache"],
    )

    return FlashInferIRModule.with_attr("external_mods", [ext_mod])
