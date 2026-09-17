# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target-owned JIT registry for generated GDN CP-prefill sources."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

from tvm_ffi import Shape

from . import env as jit_env
from .core import JitSpecNvcc, sm100a_nvcc_flags, sm103a_nvcc_flags

_PROGRAM = {'schema': 'flashinfer.gdn_cp.generated_program.v1', 'files': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_binding.cu', 'sha256': '0e72c09548ef01dc668f8ed253cbccce6a0a396d96f1b638b612ada7780e066e', 'bytes': 27153}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_binding.cu', 'sha256': 'c640c3ece7d098d8a8aad6d20dcee7d38f25fbb2f299b40d2d5ced8b6ece24e8', 'bytes': 11629}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_kernel.cu', 'sha256': 'd59613a4032bb994dab9bd05873d26a9a626ee59fe8552f1301d27cad3096db1', 'bytes': 10138}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_binding.cu', 'sha256': '505fb47fa16692388de329bc05e2f4f6c97a9c03365585864e34b9f5ce4f8fa9', 'bytes': 13399}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_binding.cu', 'sha256': '172170a6af62f21ddba08a29d301cc1bb3a4c45db8a1b5e3ae079eb9f405a0eb', 'bytes': 26193}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_binding.cu', 'sha256': 'cd6d75cd0b956cff81799b506b129f8429599e2bc240856104ef075cfdd051dc', 'bytes': 33858}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu', 'sha256': 'a9091c5c1b6b11702cd55e5e7cb71a859a8e16b9edc94b287cebaf83bd3b32ab', 'bytes': 178367}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_binding.cu', 'sha256': '443e87a4c3adae4fedc9a8225908e1a01e03c374d807da40b75bcd5f918f7a6b', 'bytes': 33858}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu', 'sha256': 'e6342f25051003f1ab28b69fd70d23c61c2aa83df5235080ab164d743fb3425f', 'bytes': 180425}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_binding.cu', 'sha256': '3eedf4bfe1ccc76266e692276dcc052813be90a4a9629deda900941721506fd3', 'bytes': 33858}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu', 'sha256': '352ea40763a110f25824e5ae55f286a3fd03d57108689db2cce34f5a9f9956d3', 'bytes': 178450}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_binding.cu', 'sha256': '2f3a76e28a5c58b95acbddea567da21a6db7c3ef4cf70c7f9c78560dc086f8db', 'bytes': 26196}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_binding.cu', 'sha256': 'd76a737bb3f31729c1d232cb7ced91dd68efea77f80eb83f8842f08a3d0a2324', 'bytes': 13399}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_kernel.cu', 'sha256': 'fc60a412aac9ec04ff820b120c595129be7729887e96a4aa315d00e8ffca1146', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_binding.cu', 'sha256': 'afd9ee87ab1b846255a0637b29ec758bc098e80f5f6eec48295167e09c8888d9', 'bytes': 33858}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_178671222ded9034a8eb_binding.cu', 'sha256': '02aab6049468bf6e4ad815408b04015bc4072cf21855cb7d61150119a1b25f36', 'bytes': 76394}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_4199d1d79d6a49edf869_binding.cu', 'sha256': '555cf3beb7bb9d249639ba8df23a0daf6215bdedd404454d3a56fa194ad2cbbe', 'bytes': 88768}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_73c66c5b63705fcf64f3_binding.cu', 'sha256': '2b989a44c077662af25afdef990706aeb41a43a771071379689590cee33b0cb4', 'bytes': 88768}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_80b5c0f5f45e95a8b86d_binding.cu', 'sha256': '09ed404134d2f1709183d7d4b7ab8f7b54eceb2b6848896717fc3675865e244e', 'bytes': 88771}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_83c723a6db4f0dc60fd6_binding.cu', 'sha256': 'fc8939974eb868132c9c50bc4f64dcfc14de9469cd833dcd03d8da033f529bcb', 'bytes': 88768}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_903968376ef39e03399b_binding.cu', 'sha256': 'd641ced9ca74c5df626ecb04e6c06f0beb8a01107229db5508af54d9441799b9', 'bytes': 88771}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_958ac07140e2a44bbe69_binding.cu', 'sha256': '0ca7748df49d10a6a7ce1c21bcd078a493c6e2d461fb78701e8d2dfad365f6fd', 'bytes': 88771}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_a76fd0df5ff8b2efbeae_binding.cu', 'sha256': '102c831324d18495dca3949b0224f8921e66a5336753979d1b82a9e6a1dff1d9', 'bytes': 88768}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_e6e240f866cdcb7f8a12_binding.cu', 'sha256': '5189f9d9b118827d4bdd9aaf483b12bb0dcd1a7ad807752017c50f8a793cb0bc', 'bytes': 88771}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_f1cc66cfcdfa6f9d66e9_binding.cu', 'sha256': '38bea3481f466f213e4a5edb5b86643faeb62f263fd21a7cde9fb8ed1e02cb3b', 'bytes': 88768}], 'modules': [{'arch': 'sm_103a', 'name': 'cake_gdn_cp_13316e89a213e54f11d7', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_13316e89a213e54f11d7', 'module_ident': 'cake_gdn_cp_13316e89a213e54f11d7_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 384, 'tma_workspace_arg_index': 13, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_binding.cu', 'sha256': '0e72c09548ef01dc668f8ed253cbccce6a0a396d96f1b638b612ada7780e066e', 'bytes': 27153}], 'program_sha256': '6b4eaa68604b7094dbc1b570d77c77548dd692371779255fccd62c77db06acd5'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_1fbdb4b8eac3c2561392', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_1fbdb4b8eac3c2561392', 'module_ident': 'cake_gdn_cp_1fbdb4b8eac3c2561392_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_kernel.cu', 'sha256': 'd59613a4032bb994dab9bd05873d26a9a626ee59fe8552f1301d27cad3096db1', 'bytes': 10138}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_binding.cu', 'sha256': 'c640c3ece7d098d8a8aad6d20dcee7d38f25fbb2f299b40d2d5ced8b6ece24e8', 'bytes': 11629}], 'program_sha256': '1efae95f760039fb720fb6a960f4c11c0522cf6da8eb25ffbe8388fc38ccd393'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_21d59ed39ef3b14a6924', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_21d59ed39ef3b14a6924', 'module_ident': 'cake_gdn_cp_21d59ed39ef3b14a6924_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_binding.cu', 'sha256': '505fb47fa16692388de329bc05e2f4f6c97a9c03365585864e34b9f5ce4f8fa9', 'bytes': 13399}], 'program_sha256': 'dc61d6e5c48a575a882b51fd7740a5b074ba8fd6e3cb980aa572bcc9bb344cd7'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_289b96a2fd636429e5e2', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_289b96a2fd636429e5e2', 'module_ident': 'cake_gdn_cp_289b96a2fd636429e5e2_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 256, 'tma_workspace_arg_index': 11, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_binding.cu', 'sha256': '172170a6af62f21ddba08a29d301cc1bb3a4c45db8a1b5e3ae079eb9f405a0eb', 'bytes': 26193}], 'program_sha256': 'b28d6a2a9279d0e95b25e01428a07f99e5eedc319e65e5f1a46c7a0c7b26e884'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_3377c622dc2ee6e0ca17', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_3377c622dc2ee6e0ca17', 'module_ident': 'cake_gdn_cp_3377c622dc2ee6e0ca17_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu', 'sha256': 'a9091c5c1b6b11702cd55e5e7cb71a859a8e16b9edc94b287cebaf83bd3b32ab', 'bytes': 178367}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_binding.cu', 'sha256': 'cd6d75cd0b956cff81799b506b129f8429599e2bc240856104ef075cfdd051dc', 'bytes': 33858}], 'program_sha256': '8a8d8afdfae891239459afd33912f78389d480ad92efdecee7304c89d3f7ed1a'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_7c846a200567707f263b', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_7c846a200567707f263b', 'module_ident': 'cake_gdn_cp_7c846a200567707f263b_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu', 'sha256': 'e6342f25051003f1ab28b69fd70d23c61c2aa83df5235080ab164d743fb3425f', 'bytes': 180425}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_binding.cu', 'sha256': '443e87a4c3adae4fedc9a8225908e1a01e03c374d807da40b75bcd5f918f7a6b', 'bytes': 33858}], 'program_sha256': '93d2aa37ab2e1cd031ea517b16381c8bdb0dce4699c0f433093b2e22c13cc71d'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_8b923b3b61c8982e0e22', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_8b923b3b61c8982e0e22', 'module_ident': 'cake_gdn_cp_8b923b3b61c8982e0e22_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu', 'sha256': '352ea40763a110f25824e5ae55f286a3fd03d57108689db2cce34f5a9f9956d3', 'bytes': 178450}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_binding.cu', 'sha256': '3eedf4bfe1ccc76266e692276dcc052813be90a4a9629deda900941721506fd3', 'bytes': 33858}], 'program_sha256': 'b68464daf7831905a9df625b453fa9a9a1b0b321e46c779714ec11f6b6597d19'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_9cc24f16037fc91ffb1b', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_9cc24f16037fc91ffb1b', 'module_ident': 'cake_gdn_cp_9cc24f16037fc91ffb1b_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 256, 'tma_workspace_arg_index': 11, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_binding.cu', 'sha256': '2f3a76e28a5c58b95acbddea567da21a6db7c3ef4cf70c7f9c78560dc086f8db', 'bytes': 26196}], 'program_sha256': 'dc4334a7217490c7049530bf5cf01f408200b14729f3c0bc44473ee807b2e22d'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_d469dc884c6b59ee6287', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_d469dc884c6b59ee6287', 'module_ident': 'cake_gdn_cp_d469dc884c6b59ee6287_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_kernel.cu', 'sha256': 'fc60a412aac9ec04ff820b120c595129be7729887e96a4aa315d00e8ffca1146', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_binding.cu', 'sha256': 'd76a737bb3f31729c1d232cb7ced91dd68efea77f80eb83f8842f08a3d0a2324', 'bytes': 13399}], 'program_sha256': '762e43035bc19c5a55732bda69c286b1692d2359d4ed212f3064a425977f9fdb'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_ea49a9f356264cabed47', 'role': 'kernel', 'translation_units': {'device': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_binding.cu'}, 'kernel_symbol': 'kernel_cake_gdn_cp_ea49a9f356264cabed47', 'module_ident': 'cake_gdn_cp_ea49a9f356264cabed47_sm_103a', 'ffi_entry': 'run', 'tma_workspace_bytes': 0, 'tma_workspace_arg_index': None, 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_binding.cu', 'sha256': 'afd9ee87ab1b846255a0637b29ec758bc098e80f5f6eec48295167e09c8888d9', 'bytes': 33858}], 'program_sha256': '4e27663bb57cd5a2902d117c468309150042eb68f0d124ee8ddde26f0f5310a7'}], 'sequences': [{'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_178671222ded9034a8eb', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_178671222ded9034a8eb_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_1fbdb4b8eac3c2561392_kernel.cu', 'sha256': 'd59613a4032bb994dab9bd05873d26a9a626ee59fe8552f1301d27cad3096db1', 'bytes': 10138}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_178671222ded9034a8eb_binding.cu', 'sha256': '02aab6049468bf6e4ad815408b04015bc4072cf21855cb7d61150119a1b25f36', 'bytes': 76394}], 'program_sha256': '014973f1ed3ba9578ef90d04da9a3860987d079c1b97309d3bffded7d2871783'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_4199d1d79d6a49edf869', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_4199d1d79d6a49edf869_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu', 'sha256': 'a9091c5c1b6b11702cd55e5e7cb71a859a8e16b9edc94b287cebaf83bd3b32ab', 'bytes': 178367}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_4199d1d79d6a49edf869_binding.cu', 'sha256': '555cf3beb7bb9d249639ba8df23a0daf6215bdedd404454d3a56fa194ad2cbbe', 'bytes': 88768}], 'program_sha256': 'd6898fdf36c37498080f81dfe5933050a88280807ad2c514c7e0f17755f6cb76'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_73c66c5b63705fcf64f3', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_73c66c5b63705fcf64f3_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu', 'sha256': '352ea40763a110f25824e5ae55f286a3fd03d57108689db2cce34f5a9f9956d3', 'bytes': 178450}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_73c66c5b63705fcf64f3_binding.cu', 'sha256': '2b989a44c077662af25afdef990706aeb41a43a771071379689590cee33b0cb4', 'bytes': 88768}], 'program_sha256': '3b24957d8324f56e90391a8a92ed542316ac44aa32d3e57274a746eb4aac8b12'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_80b5c0f5f45e95a8b86d', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_80b5c0f5f45e95a8b86d_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_8b923b3b61c8982e0e22_kernel.cu', 'sha256': '352ea40763a110f25824e5ae55f286a3fd03d57108689db2cce34f5a9f9956d3', 'bytes': 178450}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_80b5c0f5f45e95a8b86d_binding.cu', 'sha256': '09ed404134d2f1709183d7d4b7ab8f7b54eceb2b6848896717fc3675865e244e', 'bytes': 88771}], 'program_sha256': 'f3aef849f88bbf1f65f248cf723c9bc31ce044dc2a09b64d3376187f5e46a0c3'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_83c723a6db4f0dc60fd6', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_83c723a6db4f0dc60fd6_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_83c723a6db4f0dc60fd6_binding.cu', 'sha256': 'fc8939974eb868132c9c50bc4f64dcfc14de9469cd833dcd03d8da033f529bcb', 'bytes': 88768}], 'program_sha256': '09f682fe421c59271b92a82607eab228f98828bd09423b7c33eeeec0ed23d909'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_903968376ef39e03399b', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_903968376ef39e03399b_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_903968376ef39e03399b_binding.cu', 'sha256': 'd641ced9ca74c5df626ecb04e6c06f0beb8a01107229db5508af54d9441799b9', 'bytes': 88771}], 'program_sha256': 'd81d37743d4292d94dab54984630f6509407b67e9c3fa82a6f7414b8ce0436de'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_958ac07140e2a44bbe69', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_958ac07140e2a44bbe69_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu', 'sha256': 'e6342f25051003f1ab28b69fd70d23c61c2aa83df5235080ab164d743fb3425f', 'bytes': 180425}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_958ac07140e2a44bbe69_binding.cu', 'sha256': '0ca7748df49d10a6a7ce1c21bcd078a493c6e2d461fb78701e8d2dfad365f6fd', 'bytes': 88771}], 'program_sha256': '0a06a600c608d463d513e7db22bde54a5dc1e0e657cfef790a4897338dc3dadc'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_a76fd0df5ff8b2efbeae', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_a76fd0df5ff8b2efbeae_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_d469dc884c6b59ee6287_kernel.cu', 'sha256': 'fc60a412aac9ec04ff820b120c595129be7729887e96a4aa315d00e8ffca1146', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_ea49a9f356264cabed47_kernel.cu', 'sha256': 'fdca7b51a7f4dfb8ef2d73e0eddaf4c12ee9f88fc621b3b09478a8b2bdfaf0d2', 'bytes': 178457}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_a76fd0df5ff8b2efbeae_binding.cu', 'sha256': '102c831324d18495dca3949b0224f8921e66a5336753979d1b82a9e6a1dff1d9', 'bytes': 88768}], 'program_sha256': '05281d7d4becb28eeca6383fe6c6fde60b844782738d9f9a0b46293f6a6f2870'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_e6e240f866cdcb7f8a12', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_e6e240f866cdcb7f8a12_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_9cc24f16037fc91ffb1b_kernel.cu', 'sha256': 'ea5df4b650acae6c61e5786c2284f2d9ab403636ee26199301beb097da42d83d', 'bytes': 44919}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_3377c622dc2ee6e0ca17_kernel.cu', 'sha256': 'a9091c5c1b6b11702cd55e5e7cb71a859a8e16b9edc94b287cebaf83bd3b32ab', 'bytes': 178367}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_e6e240f866cdcb7f8a12_binding.cu', 'sha256': '5189f9d9b118827d4bdd9aaf483b12bb0dcd1a7ad807752017c50f8a793cb0bc', 'bytes': 88771}], 'program_sha256': '5c323ad7c092bf56eb14b92cf747d133f494066b39fc1f6b058840fbdc6a0310'}, {'arch': 'sm_103a', 'name': 'cake_gdn_cp_seq_f1cc66cfcdfa6f9d66e9', 'role': 'sequence', 'translation_units': {'devices': ['csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu'], 'binding': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_f1cc66cfcdfa6f9d66e9_binding.cu'}, 'ffi_entry': 'run', 'compile_flags': [], 'closure': [{'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_21d59ed39ef3b14a6924_kernel.cu', 'sha256': '0ff4fd7a99dca9612cd795a3f892c5c72781f94623195a8cbc4109e58fa19430', 'bytes': 65731}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_13316e89a213e54f11d7_kernel.cu', 'sha256': '4363958b084618d65adc60ffee74212a09610456a15be2f87d94867670ae0c2b', 'bytes': 113933}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_289b96a2fd636429e5e2_kernel.cu', 'sha256': '67a5e1b51dec2da4b38a71e539eb54cc93327821b2fcca4e2c4a637c33fcc428', 'bytes': 47963}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_7c846a200567707f263b_kernel.cu', 'sha256': 'e6342f25051003f1ab28b69fd70d23c61c2aa83df5235080ab164d743fb3425f', 'bytes': 180425}, {'path': 'csrc/gdn/gdn_cp/sm_103a/cake_gdn_cp_generated_seq_f1cc66cfcdfa6f9d66e9_binding.cu', 'sha256': '38bea3481f466f213e4a5edb5b86643faeb62f263fd21a7cde9fb8ed1e02cb3b', 'bytes': 88768}], 'program_sha256': '1f595c59264f32f04f132c4b70a5b22e7ba3e50d1437bf6634908a9b0ab5c865'}], 'templates': [{'arch': 'sm_103a', 'template': 'gdn_cp_fixup_simt_row4', 'module': {'name': 'cake_gdn_cp_1fbdb4b8eac3c2561392', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_fixup_utcmma128', 'module': {'name': 'cake_gdn_cp_9cc24f16037fc91ffb1b', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_fixup_utcmma64', 'module': {'name': 'cake_gdn_cp_289b96a2fd636429e5e2', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_mn_precompute_fp16', 'module': {'name': 'cake_gdn_cp_13316e89a213e54f11d7', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_prefill_equal_head_fp16', 'module': {'name': 'cake_gdn_cp_3377c622dc2ee6e0ca17', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_prefill_equal_head_h32_fp16', 'module': {'name': 'cake_gdn_cp_8b923b3b61c8982e0e22', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_prefill_fp16', 'module': {'name': 'cake_gdn_cp_ea49a9f356264cabed47', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_prefill_generic_fp16', 'module': {'name': 'cake_gdn_cp_7c846a200567707f263b', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_t_precompute_fp16', 'module': {'name': 'cake_gdn_cp_21d59ed39ef3b14a6924', 'role': 'kernel'}}, {'arch': 'sm_103a', 'template': 'gdn_cp_t_precompute_gb300_hv48_min6', 'module': {'name': 'cake_gdn_cp_d469dc884c6b59ee6287', 'role': 'kernel'}}]}
_STABLE_TEMPLATES = {
    "t_precompute": "gdn_cp_t_precompute_fp16",
    "t_precompute_gb300_hv48_min6": "gdn_cp_t_precompute_gb300_hv48_min6",
    "mn_precompute": "gdn_cp_mn_precompute_fp16",
    "state_fixup_simt_row4": "gdn_cp_fixup_simt_row4",
    "state_fixup_utcmma64": "gdn_cp_fixup_utcmma64",
    "state_fixup_utcmma128": "gdn_cp_fixup_utcmma128",
    "cp_prefill": "gdn_cp_prefill_fp16",
    "cp_prefill_checkpoint": "gdn_cp_prefill_fp16",
    "cp_prefill_equal_head": "gdn_cp_prefill_equal_head_fp16",
    "cp_prefill_equal_head_h32": "gdn_cp_prefill_equal_head_h32_fp16",
    "cp_prefill_equal_head_checkpoint": "gdn_cp_prefill_equal_head_fp16",
    "cp_prefill_generic": "gdn_cp_prefill_generic_fp16",
    "cp_prefill_generic_checkpoint": "gdn_cp_prefill_generic_fp16",
}


def _source_root() -> Path:
    sentinel = _PROGRAM["files"][0]["path"]
    candidates = (
        jit_env.FLASHINFER_CSRC_DIR.parent,
        Path(__file__).resolve().parents[2],
    )
    for root in candidates:
        if (root / sentinel).is_file():
            return root
    raise FileNotFoundError(f"generated GDN CP source {sentinel!r} was not found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delivered_path(relative: str) -> Path:
    path = _source_root() / relative
    if not path.is_file():
        raise RuntimeError(f"generated GDN CP source is missing: {relative!r}")
    return path


@functools.cache
def generated_gdn_cp_program() -> dict[str, Any]:
    if _PROGRAM.get("schema") != "flashinfer.gdn_cp.generated_program.v1":
        raise RuntimeError("generated GDN CP registry has an unsupported identity")
    for record in _PROGRAM.get("files", ()):
        source = _delivered_path(record["path"])
        if not source.is_file() or _sha256(source) != record["sha256"]:
            raise RuntimeError(f"generated GDN CP source drift at {source}")
    return _PROGRAM


def _record(collection: str, name: str, role: str, arch: str) -> dict[str, Any]:
    matches = [
        item
        for item in generated_gdn_cp_program()[collection]
        if item["name"] == name and item["role"] == role and item["arch"] == arch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"generated GDN CP {collection} record is not unique: {(name, role, arch)!r}"
        )
    return matches[0]


def _translation_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict) and isinstance(value.get("paths"), list):
        return list(value["paths"])
    raise TypeError(f"invalid generated translation-unit delivery: {value!r}")


_ARCH_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def _fast_divmod_shape(divisor: int) -> Shape:
    """Build the exact generated FastDivmod carrier without a producer-package dependency."""

    if not isinstance(divisor, int) or isinstance(divisor, bool) or not 1 <= divisor <= 0x7FFF_FFFF:
        raise ValueError(f"FastDivmod divisor must be in [1, 2147483647], got {divisor!r}")
    if divisor == 1:
        multiplier = shift_right = 0
    else:
        p = 31 + (divisor - 1).bit_length()
        multiplier = (((1 << p) + divisor - 1) // divisor) & 0xFFFF_FFFF
        shift_right = p - 32
    return Shape((divisor, multiplier, shift_right))


def _build(record: dict[str, Any], sources: list[str]):
    source_paths = [_delivered_path(path) for path in sources]
    arch_flags = _ARCH_FLAGS.get(record["arch"])
    if arch_flags is None:
        raise ValueError(f"unsupported generated GDN CP architecture: {record['arch']!r}")
    name = f"generated_gdn_cp_{record['name']}_{record['program_sha256'][:16]}"
    spec = JitSpecNvcc(
        name=name,
        sources=source_paths,
        extra_cflags=["-std=c++17", "-DNDEBUG", "-O3"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-DNDEBUG",
            "-O3",
            *arch_flags,
            *record.get("compile_flags", ()),
        ],
        extra_ldflags=None,
        extra_include_dirs=[
            _source_root() / "csrc",
            _source_root() / "csrc" / "include",
            _source_root() / "include",
        ],
        needs_device_linking=True,
    )
    module = spec.build_and_load()
    library = spec.get_library_path()
    if not library.is_file():
        raise RuntimeError(f"generated GDN CP library was not materialized for {record['name']!r}")
    identity = {
        "arch": record["arch"],
        "ffi_entry": record["ffi_entry"],
        "program_sha256": record["program_sha256"],
        "binary_sha256": _sha256(library),
        "binary_bytes": library.stat().st_size,
    }
    return module, identity


def _module_for_template(template: str, arch: str) -> dict[str, Any]:
    refs = {
        (record["module"]["name"], record["module"]["role"])
        for record in generated_gdn_cp_program()["templates"]
        if record["arch"] == arch and record["template"] == template
    }
    if len(refs) != 1:
        raise ValueError(f"template {template!r} does not select one generated module for {arch}")
    name, role = next(iter(refs))
    matches = [
        item
        for item in generated_gdn_cp_program()["modules"]
        if item["name"] == name and item["role"] == role and item["arch"] == arch
    ]
    if len(matches) != 1:
        raise ValueError(f"generated module reference {(name, role, arch)!r} is not unique")
    return matches[0]


@functools.cache
def load_generated_gdn_cp_kernel(name: str, arch: str):
    template = _STABLE_TEMPLATES.get(name)
    if template is None:
        return None
    if not any(record["arch"] == arch for record in generated_gdn_cp_program()["templates"]):
        return None
    record = _module_for_template(template, arch)
    translation = record["translation_units"]
    sources = [
        *_translation_paths(translation["device"]),
        *_translation_paths(translation["binding"]),
    ]
    module, _identity = _build(record, sources)
    entry = module[record["ffi_entry"]]
    if template not in (
        "gdn_cp_prefill_fp16",
        "gdn_cp_prefill_equal_head_fp16",
        "gdn_cp_prefill_equal_head_h32_fp16",
        "gdn_cp_prefill_generic_fp16",
    ):
        return entry

    def launch_with_fast_divmod(*args):
        # Preserve the existing FlashInfer private launch call while inserting
        # the two target-owned carriers required by the generated Stage-4 ABI.
        if len(args) != 20:
            raise TypeError(f"generated GDN CP prefill expected 20 legacy arguments, got {len(args)}")
        cp_chunk_len = args[10]
        num_sab_heads = args[15]
        return entry(
            *args[:16],
            _fast_divmod_shape(cp_chunk_len),
            _fast_divmod_shape(num_sab_heads),
            *args[16:],
        )

    return launch_with_fast_divmod


def prepare_generated_gdn_cp_kernel(name: str, arch: str, *, device):
    """Bind caller-owned TMA descriptors to one prepared kernel instance."""

    entry = load_generated_gdn_cp_kernel(name, arch)
    if entry is None:
        return None
    record = _module_for_template(_STABLE_TEMPLATES[name], arch)
    workspace_bytes = record["tma_workspace_bytes"]
    if not workspace_bytes:
        return entry, ()

    import torch

    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    workspace_index = record["tma_workspace_arg_index"]

    def launch_with_workspace(*args):
        return entry(*args[:workspace_index], workspace, *args[workspace_index:])

    return launch_with_workspace, (workspace,)


@functools.cache
def load_generated_gdn_cp_sequence(name: str, role: str, arch: str):
    record = _record("sequences", name, role, arch)
    translation = record["translation_units"]
    sources = [
        *(path for item in translation["devices"] for path in _translation_paths(item)),
        *_translation_paths(translation["binding"]),
    ]
    module, _identity = _build(record, sources)
    return module[record["ffi_entry"]]


def _record_sources(collection: str, record: dict[str, Any]) -> list[str]:
    translation = record["translation_units"]
    if collection == "modules":
        return [
            *_translation_paths(translation["device"]),
            *_translation_paths(translation["binding"]),
        ]
    return [
        *(path for item in translation["devices"] for path in _translation_paths(item)),
        *_translation_paths(translation["binding"]),
    ]


@functools.cache
def _artifact_identity(collection: str, name: str, role: str, arch: str) -> dict[str, Any]:
    record = _record(collection, name, role, arch)
    _module, identity = _build(record, _record_sources(collection, record))
    return identity


def build_all_generated_gdn_cp() -> dict[str, dict[str, Any]]:
    program = generated_gdn_cp_program()
    identities: dict[str, dict[str, Any]] = {}
    for collection, kind in (("modules", "module"), ("sequences", "sequence")):
        for record in program[collection]:
            label = f"{kind}:{record['name']}:{record['role']}"
            identities[label] = _artifact_identity(
                collection, record["name"], record["role"], record["arch"]
            )
    return identities


__all__ = [
    "build_all_generated_gdn_cp",
    "generated_gdn_cp_program",
    "load_generated_gdn_cp_kernel",
    "prepare_generated_gdn_cp_kernel",
    "load_generated_gdn_cp_sequence",
]
