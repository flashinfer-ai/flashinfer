// FLASHINFER_CAKE_GROUPED_MXFP8_DEVICE_PLACEHOLDER
//
// Replace this file with the generated FP16 row2d program. It must define:
//
//   extern "C" __global__ void kernel_cake_grouped_mxfp8_quantize_f16(
//       const half* input, const int32_t* mask, fp8_e4m3* quantized,
//       uint8_t* scales, int32_t M, int32_t K, int32_t PADDED_K,
//       int32_t PM_TILES, int32_t PK_TILES, int32_t BLOCKS_PER_ROW,
//       uint64_t TOTAL_TASKS);
//
// Pointer pointee spellings may use generated private aliases; the host shim
// launches through cudaLaunchKernel and therefore binds the pointer ABI by
// address. Shape parameters are checked before int32 narrowing and all device
// offsets and TOTAL_TASKS must remain 64-bit.
