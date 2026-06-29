#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> bsa_fused_fwd_blk64(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                               torch::Tensor q2k_block_index, int block_sparse_num,
                                               torch::Tensor block_sizes, float softmax_scale,
                                               torch::Tensor q2k_block_nums);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bsa_fused_fwd_blk64", &bsa_fused_fwd_blk64, "BSA fused attention forward blk=64 (C++ AOT)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("q2k_block_index"),
        py::arg("block_sparse_num"), py::arg("block_sizes"), py::arg("softmax_scale"),
        py::arg("q2k_block_nums") = torch::Tensor());
}
