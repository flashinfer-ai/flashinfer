// Shared combine sizing for the fused FP4 paths. No arithmetic policy changes.
constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

constexpr uint32_t kNumChunkSlots = 3;
constexpr uint32_t kNumMaxRegistersForBuffer = 128;
constexpr uint32_t kNumChunks =
    (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and
     kHidden <= 32 * kNumMaxRegistersForBuffer)
        ? 1
        : 2;
constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
FI_MEGAMOE_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
FI_MEGAMOE_STATIC_ASSERT(kNumChunkSlots* kNumEpilogueWarps* kNumHiddenBytes / kNumChunks <=
                             SMEM_BEFORE_BARRIER_SIZE,
                         "Hidden is too large");
FI_MEGAMOE_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
FI_MEGAMOE_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0,
                         "Combine chunk must be divisible by 16 bytes");
FI_MEGAMOE_STATIC_ASSERT(kNumChunkUint4 % 32 == 0,
                         "Combine chunk must be a multiple of 32 16-byte elements");
FI_MEGAMOE_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");
