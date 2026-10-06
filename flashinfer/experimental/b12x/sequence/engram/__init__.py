"""DeepSeek V4.1 compressed-token Engram hash and FP8 gather."""

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="engram",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "LookupBinding",
        "DiskTable",
        "EngramQuery",
        "Geometry",
        "build_geometry",
        "build_compressed_token_map",
        "plan",
        "bind",
        "bind_lookup",
        "run",
        "run_lookup",
        "run_lookups",
        "is_supported",
    ),
    dtypes=("int64", "float8_e4m3fn", "float8_e8m0fnu", "bfloat16"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash",
        commit="fb2764a5cf321eaa5070ca8f9e892818f477c16d",
        paths=("inference/engram.py", "inference/model.py"),
    ),
    test_path="tests/experimental/b12x/sequence/test_engram.py",
    since="1.3.0",
    notes="Immutable PCG64 Engram geometry and prepared native lookup.",
)
install_lazy_api(globals(), META)
