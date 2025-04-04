/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/customAllReduceKernels.h>

#include "pytorch_extension_utils.h"

void AllReduceSum(at::Tensor data,
                  at::Tensor workspace,
                  int64_t world_size, 
                  int64_t rank, 
                  int64_t num_ctas) {
    printf("AllReduce called with num_ctas = %d\n", (int)num_ctas);

    float* workspace_ptr = workspace.data_ptr<float>();
    auto dtype = data.scalar_type();
    int hidden_size = data.size(-1);
    int token_num = data.numel() / hidden_size;
    auto fusion_op = tensorrt_llm::kernels::AllReduceFusionOp::NONE;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto params = tensorrt_llm::kernels::AllReduceParams::deserialize(
        reinterpret_cast<int64_t*>(workspace_ptr), 
        world_size, 
        rank, 
        dtype, 
        token_num, 
        hidden_size, 
        fusion_op
    );

    auto strat_config = tensorrt_llm::kernels::AllReduceStrategyConfig::PUSH_MODE;
    auto strat_type = tensorrt_llm::kernels::AllReduceStrategyType::AUTO;

    customAllReduce(params, dtype, strat_type, strat_config, fusion_op, stream);
}
