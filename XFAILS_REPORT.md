# FlashInfer Test Skips and Expected Failures Report

**Issue Type:** Test Coverage & Infrastructure  
**Priority:** High  
**Generated:** 2024-12-19  

## Executive Summary

This report identifies **143 test skips and expected failures** across the FlashInfer test suite, highlighting areas that require developer attention. These issues span hardware compatibility, missing features, parameter validation, and backend limitations.

## Critical Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Hardware Requirements** | 51 | 35.7% |
| **Feature Unsupported** | 31 | 21.7% |
| **Parameter Validation** | 21 | 14.7% |
| **Other Issues** | 34 | 23.8% |
| **Environment Issues** | 2 | 1.4% |
| **Backend Limitations** | 4 | 2.8% |
| **Total** | **143** | **100%** |

## High-Priority Issues Requiring Developer Action

### 🚨 Hardware Compatibility Crisis (51 issues)

The most critical concern is **extensive hardware compatibility problems** affecting modern GPU architectures:

#### SM90/100/110/120 GPU Support Issues
- **SM90A not supported**: 10 test failures across core functionality
- **SM110/120/121 limitations**: 6 TensorRT-LLM integration failures  
- **Compute capability requirements**: Multiple features require specific GPU generations
- **Backend-specific hardware restrictions**: Different backends have incompatible hardware requirements

**Impact**: Users with newer GPU hardware cannot fully utilize FlashInfer capabilities.

**Affected Components:**
- `tests/test_hopper.py` (6 failures)
- `tests/test_trtllm_gen_*.py` (multiple TensorRT-LLM integration issues)
- `tests/test_fp4_quantize.py` (FP4 quantization requires SM100+)
- `tests/test_blackwell_fmha.py` (Blackwell architecture support)

### 🔧 Missing Feature Support (31 issues)

Critical functionality gaps that limit FlashInfer's usability:

#### FlashAttention 3 Support
- **9 test failures** due to FA3 not being supported on target devices
- Affects attention sink and DeepSeek MLA functionality

#### Sequence Length Limitations  
- **Causal attention restrictions**: `qo_len > kv_len` not supported in multiple contexts
- **Variable length limitations**: Missing support for dynamic sequence handling

#### Backend Feature Gaps
- **TensorRT-LLM limitations**: Multiple unsupported feature combinations
- **CUDNN/Cutlass backend gaps**: Different backends support different feature sets

### ⚠️ Parameter Validation Issues (21 issues)

Insufficient parameter validation causing test failures:

#### Head Configuration Problems
- **`num_qo_heads` must be divisible by `num_kv_heads`**: 5 failures
- **`num_qo_heads` must be multiple of `num_kv_heads`**: 4 failures

#### Vocabulary and Sampling Issues  
- **`k` should be less than `vocab_size`**: 8 sampling-related failures

#### Block Sparse Configuration
- **Sequence length validation**: Block sizes must be smaller than sequence lengths

## Detailed Breakdown by Category

### Hardware Requirements (51 items)

<details>
<summary>Click to expand hardware issues</summary>

| Issue | Count | Files Affected |
|-------|-------|----------------|
| SM90A is not supported | 10 | `test_hopper.py`, `test_hopper_fp8_attention.py`, `test_jit_example.py` |
| trtllm-gen does not support SM110/SM120/SM121 GPUs | 6 | `test_attention_sink_blackwell.py`, `test_trtllm_gen_*.py` |
| PDL is only available for Hopper and later GPUs | 7 | `test_activation.py`, `test_norm.py` |
| Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8 | 6 | `test_fp4_quantize.py` |
| only SM100A and SM110A are supported | 3 | `test_blackwell_fmha.py` |
| XQA is only supported on SM90 GPUs | 2 | `test_xqa.py` |

</details>

### Feature Unsupported (31 items)

<details>
<summary>Click to expand unsupported features</summary>

| Issue | Count | Files Affected |
|-------|-------|----------------|
| FA3 is not supported on this device | 9 | `test_attention_sink.py`, `test_deepseek_mla.py` |
| qo_len > kv_len and causal is not supported | 5 | `test_batch_prefill_kernels.py`, `test_blackwell_fmha.py`, `test_single_prefill.py` |
| qo_len > kv_len not supported for causal attention | 3 | `test_deepseek_mla.py` |
| Mnnvl memory is not supported on this platform | 3 | `test_mnnvl_*.py` |

</details>

### Parameter Validation (21 items)

<details>
<summary>Click to expand parameter validation issues</summary>

| Issue | Count | Files Affected |
|-------|-------|----------------|
| k should be less than vocab_size | 8 | `test_logits_processor.py`, `test_sampling.py` |
| num_qo_heads must be divisible by num_kv_heads | 5 | `test_block_sparse.py`, `test_hopper.py` |
| num_qo_heads must be a multiple of num_kv_heads | 4 | `test_non_contiguous_*.py` |

</details>

## Immediate Action Items for Developers

### 🎯 Priority 1: Hardware Compatibility
1. **Audit SM90A support** - Determine why SM90A is marked as unsupported
2. **TensorRT-LLM SM110/120/121 support** - Coordinate with NVIDIA on compatibility
3. **Unified hardware requirement documentation** - Create clear compatibility matrix
4. **Fallback implementations** - Provide software alternatives where hardware features are unavailable

### 🎯 Priority 2: Feature Implementation  
1. **FlashAttention 3 integration** - Complete FA3 support across all target devices
2. **Causal attention with qo_len > kv_len** - Implement missing sequence length support
3. **Backend feature parity** - Ensure consistent feature support across CUDNN/Cutlass/TensorRT-LLM

### 🎯 Priority 3: Parameter Validation Enhancement
1. **Improve error messages** - Make parameter validation failures more informative
2. **Runtime parameter checking** - Add comprehensive validation before kernel launch
3. **Configuration helpers** - Provide utilities to validate parameter combinations

### 🎯 Priority 4: Test Infrastructure
1. **Conditional test execution** - Better hardware detection and graceful degradation
2. **Test categorization** - Separate hardware-dependent from logic tests
3. **CI/CD integration** - Track xfail reduction over time

## Monitoring and Tracking

This report should be regenerated regularly to track progress. Key metrics to monitor:

- **Total xfail count**: Target reduction from 143 to <50
- **Hardware compatibility**: Focus on reducing the 51 hardware-related issues
- **Feature coverage**: Track implementation of the 31 unsupported features
- **Parameter validation**: Improve the 21 validation issues

## Technical Debt Assessment

**Critical Technical Debt:**
- Hardware abstraction layer needs improvement
- Backend feature detection and fallback mechanisms missing
- Parameter validation scattered across codebase without central validation

**Estimated Effort:**
- Hardware compatibility fixes: 2-3 engineer-months
- Feature implementation: 3-4 engineer-months  
- Parameter validation overhaul: 1-2 engineer-months

## Conclusion

The FlashInfer test suite reveals significant technical debt in hardware compatibility and feature coverage. Addressing these 143 issues is crucial for:

1. **User Experience**: Ensuring FlashInfer works across modern GPU hardware
2. **Feature Completeness**: Providing comprehensive functionality
3. **Developer Productivity**: Reducing test failures and debugging overhead
4. **Project Maturity**: Moving towards a production-ready state

**Recommended Next Steps:**
1. Create GitHub issues for each high-priority category
2. Assign ownership for hardware compatibility workstream
3. Establish monthly tracking of xfail reduction progress
4. Integrate automated report generation into CI/CD pipeline

---

*Generated by FlashInfer xfails analysis tool. For questions or updates, contact the FlashInfer development team.*