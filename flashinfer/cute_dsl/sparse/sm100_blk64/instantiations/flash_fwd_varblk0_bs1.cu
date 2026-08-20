// Explicit instantiation: HasVarBlockNums=false, HasBlockSizes=true
#include "../flash_fwd_launch_template.h"

template std::vector<torch::Tensor> flash::bsa_fused_fwd_blk64_launch<false, true>(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, torch::Tensor, float,
    torch::Tensor);
