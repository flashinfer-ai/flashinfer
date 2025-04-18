// flashinfer: adapted from sglang + vllm code
// refer to: sgl-kernel/csrc/common_extension.cc from sglang
#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

using fptr_t = int64_t;

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data,
                      int64_t rank, bool full_nvlink);
void dispose(fptr_t _fa);
int64_t meta_size();
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes, int64_t num_ctas);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs);
void register_graph_buffers(fptr_t _fa, const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
