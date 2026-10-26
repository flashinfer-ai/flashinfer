"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Experimental Cake backend: MiniMax-M3 ICP decode indexer chain (fused FP8
# index scorer with in-epilogue top-16 + symmetric-memory exchange merge) for
# SM100 / SM103.  The public entry points are
# ``flashinfer.msa_ops.prepare_msa_icp_indexer``,
# ``create_msa_icp_indexer_workspace`` and ``destroy_msa_icp_indexer_workspace``;
# the generated programs, their host binding, the workspace and the JIT
# registration live in this package.
