"""DeepSeek V4.1 prepared CSA1/CSA2 compression."""
from ..._lib.meta import OpMeta,Provenance,install_lazy_api
META=OpMeta(name="mla_compress",group="attention",api_style="planned",entry_points=("Caps","Plan","Binding","MlaCompressQuery","plan","bind","run","reference","is_supported"),dtypes=("bf16",),requires=("cutlass",),provenance=Provenance(repo="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash",commit="fb2764a5cf321eaa5070ca8f9e892818f477c16d",paths=("inference/model.py",)),test_path="tests/experimental/b12x/attention/test_mla_compress.py",since="1.3.0",notes="Prepared native CSA compression.")
install_lazy_api(globals(),META)
