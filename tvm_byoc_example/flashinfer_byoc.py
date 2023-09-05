import os
import tvm
from tvm.script import ir as I, relax as R, tir as T

# fmt: off
def FlashInferIRModuleGen(dtype_in: str, dtype_out: str, qkv_layout: int = 0, rotary_mode: int = 0, rope_scale: float = 1.0, rope_theta: float = 1e4) -> tvm.IRModule:
    @I.ir_module
    class FlashInferIRModuleNHD:
        @R.function
        def decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype_in),
            k: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype_in),
            v: R.Tensor(("seq_len", "num_heads", "head_dim"), dtype_in),
            tmp: R.Tensor((2 * 1024 * 1024,), "float32"),
        ) -> R.Tensor(("num_heads", "head_dim"), dtype_out):
            num_heads = T.int64()
            head_dim = T.int64()
            dtype = dtype_in
            with R.dataflow():
                o = R.call_dps_packed(
                    "FlashInferSingleDecodeWithKVCache",
                    (q, k, v, tmp, 0, rotary_mode, rope_scale, rope_theta),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype_out),
                )
                R.output(o)
            return o

    @I.ir_module
    class FlashInferIRModuleHND:
        @R.function
        def decode(
            q: R.Tensor(("num_heads", "head_dim"), dtype_in),
            k: R.Tensor(("num_heads", "seq_len", "head_dim"), dtype_in),
            v: R.Tensor(("num_heads", "seq_len", "head_dim"), dtype_in),
            tmp: R.Tensor((2 * 1024 * 1024,), "float32"),
        ) -> R.Tensor(("num_heads", "head_dim"), dtype_out):
            num_heads = T.int64()
            head_dim = T.int64()
            dtype = dtype_in
            with R.dataflow():
                o = R.call_dps_packed(
                    "FlashInferSingleDecodeWithKVCache",
                    (q, k, v, tmp, 1, rotary_mode, rope_scale, rope_theta),
                    out_sinfo=R.Tensor((num_heads, head_dim), dtype_out),
                )
                R.output(o)
            return o

    ext_mod = tvm.runtime.load_static_library(
        os.path.dirname(os.path.realpath(__file__))
        + "/../build/CMakeFiles/tvm_binding.dir/src/tvm_wrapper.cu.o",
        ["FlashInferSingleDecodeWithKVCache"],
    )

    if qkv_layout == 0:
        return FlashInferIRModuleNHD.with_attr("external_mods", [ext_mod])
    elif qkv_layout == 1:
        return FlashInferIRModuleHND.with_attr("external_mods", [ext_mod])
    else:
        raise ValueError("qkv_layout should be 0 or 1")
