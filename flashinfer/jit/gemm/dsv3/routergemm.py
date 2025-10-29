from flashinfer.jit import gen_dsv3_router_gemm_module


def get_dsv3_router_gemm_module():
    module = gen_dsv3_router_gemm_module().build_and_load()

    return module
