/* Auto-generated public compact-FP32 binding selector. */
#ifndef FLASHINFER_FLASH_KDA_TARGET_MINOR
#error "JIT spec must define FLASHINFER_FLASH_KDA_TARGET_MINOR"
#endif
static_assert(FLASHINFER_FLASH_KDA_TARGET_MINOR == 3,
              "binding compiled for the wrong exact target");

#define FLASHKDA_GENERATED_BODY_FILE "flashkda_generated_fp32_compat_sm_103a_ui1_sf1_a757764a13.cu"
#define FLASHKDA_GENERATED_KERNEL kernel_flashkda_blackwell_prefill_fp32_state_initial
#define FLASHKDA_GENERATED_THREADS 384
#define FLASHKDA_GENERATED_SMEM_BYTES 226048
#define FLASHKDA_GENERATED_USE_PDL 0
#define FLASHKDA_FP32_COMPAT_USE_INITIAL_STATE 1
#define FLASHKDA_FP32_COMPAT_STORE_FINAL_STATE 1
#include "flashkda_generated_fp32_compat_binding.cuh"
