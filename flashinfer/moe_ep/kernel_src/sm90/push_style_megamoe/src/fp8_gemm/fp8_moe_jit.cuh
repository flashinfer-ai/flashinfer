/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tensorrt_llm/deep_gemm/compiler.cuh>
#include <vector>

#if __has_include("fp8_moe_build_config.h")
#include "fp8_moe_build_config.h"
#else
#error "fp8_moe_build_config.h is required to identify the FP8 MoE JIT sources"
#endif

namespace flashinfer::sm90_push_fp8::jit {

inline std::filesystem::path grouped_kernel_cache_path(uint32_t shape_n, uint32_t shape_k,
                                                       uint32_t block_m, uint32_t block_n,
                                                       uint32_t block_k, uint32_t num_groups,
                                                       uint32_t num_stages,
                                                       uint32_t num_tma_multicast, bool swap_ab) {
  std::string const name =
      std::string(swap_ab ? "gemm_swapAB_" : "gemm_") + std::to_string(shape_n) + "_" +
      std::to_string(shape_k) + "_" + std::to_string(block_m) + "_" + std::to_string(block_n) +
      "_" + std::to_string(block_k) + "_" + std::to_string(num_groups) + "_" +
      std::to_string(num_stages) + std::to_string(num_groups) + "_" + std::to_string(num_stages) +
      "_" + std::to_string(num_tma_multicast) + "_GroupedWithOffset";
  return deep_gemm::jit::getCacheDir() / name;
}

inline bool grouped_kernel_cache_ready(uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
                                       uint32_t block_n, uint32_t block_k, uint32_t num_groups,
                                       uint32_t num_stages, uint32_t num_tma_multicast,
                                       bool swap_ab) {
  auto const path = grouped_kernel_cache_path(shape_n, shape_k, block_m, block_n, block_k,
                                              num_groups, num_stages, num_tma_multicast, swap_ab);
  return deep_gemm::jit::getGlobalRuntimeCache()[path.string()] != nullptr;
}

inline std::filesystem::path fc1_kernel_cache_path(uint32_t shape_n, uint32_t shape_k,
                                                   uint32_t block_m, uint32_t block_n,
                                                   uint32_t block_k, uint32_t num_groups,
                                                   uint32_t num_stages) {
  std::string const name = "sm90_push_fp8_fc1_" + std::to_string(shape_n) + "_" +
                           std::to_string(shape_k) + "_" + std::to_string(block_m) + "_" +
                           std::to_string(block_n) + "_" + std::to_string(block_k) + "_" +
                           std::to_string(num_groups) + "_" + std::to_string(num_stages) + "_" +
                           FLASHINFER_SM90_PUSH_FP8_MOE_SOURCE_DIGEST + "_GroupedWithOffset";
  return deep_gemm::jit::getCacheDir() / name;
}

inline std::string generate_fc1_kernel(uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
                                       uint32_t block_n, uint32_t block_k, uint32_t num_groups,
                                       uint32_t num_stages) {
  constexpr uint32_t kNumTMAThreads = 128;
  constexpr uint32_t kNumMathThreadsPerGroup = 128;

  return R"(
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <fp8_moe_fc1_fused.cuh>

using SchedulerType = flashinfer::sm90_push_fp8::Fp8MoeFc1Scheduler<)" +
         std::to_string(shape_n) + ", " + std::to_string(block_m) + ", " + std::to_string(block_n) +
         ", " + std::to_string(num_groups) + R"(, 1>;

__global__ void dummy_kernel() {
  void* ptr = (void*)&flashinfer::sm90_push_fp8::fp8_gemm_kernel_sm90_push_fc1_fused<)" +
         std::to_string(shape_n) + ", " + std::to_string(shape_k) + ", " + std::to_string(block_m) +
         ", " + std::to_string(block_n) + ", " + std::to_string(block_k) + ", " +
         std::to_string(num_groups) + ", " + std::to_string(num_stages) + ", " +
         std::to_string(kNumTMAThreads) + ", " + std::to_string(kNumMathThreadsPerGroup) +
         R"(, 1, SchedulerType, deep_gemm::GroupedWithOffsetSchedulerInput>;
}
)";
}

inline std::string shell_quote(std::string const& value) {
#ifdef _WIN32
  std::string quoted = "\"";
  for (char c : value) quoted += c == '"' ? std::string("\\\"") : std::string(1, c);
  return quoted + "\"";
#else
  std::string quoted = "'";
  for (char c : value) quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return quoted + "'";
#endif
}

class Fp8MoeFc1Compiler {
 public:
  static Fp8MoeFc1Compiler& getInstance() {
    static Fp8MoeFc1Compiler compiler;
    return compiler;
  }

  deep_gemm::jit::Runtime* build(uint32_t shape_n, uint32_t shape_k, uint32_t block_m,
                                 uint32_t block_n, uint32_t block_k, uint32_t num_groups,
                                 uint32_t num_stages) {
    int const sm_version = tensorrt_llm::common::getSMVersion();
    if (sm_version != 90) {
      TLLM_THROW("SM90 push FP8 MoE GEMM requires SM90, got SM%d", sm_version);
    }
    auto const& include_dirs = deep_gemm::jit::getJitIncludeDirs();
    TLLM_CHECK_WITH_INFO(!include_dirs.empty(),
                         "SM90 push FP8 MoE JIT include directories are not configured");

    std::filesystem::path const path =
        fc1_kernel_cache_path(shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages);
    std::string const name = path.filename().string();
    auto& runtime_cache = deep_gemm::jit::getGlobalRuntimeCache();
    if (auto* runtime = runtime_cache[path.string()]) return runtime;

    std::filesystem::path const tmp_path =
        deep_gemm::jit::getTmpDir() / (name + "_" + deep_gemm::jit::generateUniqueId());
    std::filesystem::path const cubin_path = path / deep_gemm::jit::kKernelName;
    std::filesystem::path const tmp_cubin_path = tmp_path / deep_gemm::jit::kKernelName;
    std::filesystem::create_directories(tmp_path);
    std::filesystem::create_directories(path);

    std::filesystem::path const source_path = tmp_path / "kernel.cu";
    {
      std::ofstream source(source_path);
      source << generate_fc1_kernel(shape_n, shape_k, block_m, block_n, block_k, num_groups,
                                    num_stages);
      TLLM_CHECK_WITH_INFO(source.good(), "failed to write SM90 push FP8 MoE JIT source");
    }

    std::vector<std::string> command = {
        deep_gemm::jit::getNvccCompiler(),
        source_path.string(),
        "-o",
        tmp_cubin_path.string(),
        "-std=c++17",
        "--gpu-architecture=sm_90a",
        "-O3",
        "-cubin",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-allow-expensive-optimizations=true",
        "--ptxas-options=--register-usage-level=10",
        "--diag-suppress=161,174,177,940",
        "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1",
        "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1",
        "--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi",
    };
    for (auto const& dir : include_dirs) command.push_back("-I" + dir.string());

    std::string shell_command;
    for (auto const& argument : command) shell_command += shell_quote(argument) + " ";
#ifdef _WIN32
    FILE* pipe = _popen((shell_command + " 2>&1").c_str(), "r");
#else
    FILE* pipe = popen((shell_command + " 2>&1").c_str(), "r");
#endif
    TLLM_CHECK_WITH_INFO(pipe != nullptr, "failed to start nvcc for SM90 push FP8 MoE kernel");
    std::array<char, 4096> buffer{};
    std::string log;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
      log += buffer.data();
    }
#ifdef _WIN32
    int const status = _pclose(pipe);
#else
    int const status = pclose(pipe);
#endif
    if (status != 0 || !std::filesystem::exists(tmp_cubin_path)) {
      std::filesystem::remove_all(tmp_path);
      throw std::runtime_error("SM90 push FP8 MoE nvcc compilation failed:\n" + log);
    }

    try {
      std::filesystem::rename(tmp_cubin_path, cubin_path);
    } catch (std::filesystem::filesystem_error const&) {
      if (!std::filesystem::exists(cubin_path)) {
        std::filesystem::remove_all(tmp_path);
        throw;
      }
    }
    std::filesystem::remove_all(tmp_path);

    auto runtime = std::make_unique<deep_gemm::jit::Runtime>(
        path.string(), std::vector<char>(), deep_gemm::GemmType::GroupedWithOffset);
    auto* result = runtime.get();
    runtime_cache.set(path.string(), std::move(runtime));
    return result;
  }

  bool is_cached(uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
                 uint32_t block_k, uint32_t num_groups, uint32_t num_stages) const {
    auto const path =
        fc1_kernel_cache_path(shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages);
    return deep_gemm::jit::getGlobalRuntimeCache()[path.string()] != nullptr;
  }

 private:
  Fp8MoeFc1Compiler() {
    std::filesystem::create_directories(deep_gemm::jit::getTmpDir());
    std::filesystem::create_directories(deep_gemm::jit::getCacheDir());
  }
};

inline Fp8MoeFc1Compiler& get_fc1_compiler() { return Fp8MoeFc1Compiler::getInstance(); }

}  // namespace flashinfer::sm90_push_fp8::jit
