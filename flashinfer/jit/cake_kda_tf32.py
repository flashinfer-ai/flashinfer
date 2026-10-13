"""Native JIT loading for the generated KDA prefill schedules (SM100a and SM103a).

One kernel source serves both architectures where the generated bodies agree
(architecture-specific lowering sits under exact ``__CUDA_ARCH__`` guards); each
architecture still compiles its own module. Argument plans and factory kwargs are
stored once and expanded into ``MODULES`` / ``FACTORIES`` at import.
"""

from functools import cache
from pathlib import Path

import torch

from . import env as jit_env
from .core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

ARCHES = ("sm_100a", "sm_103a")
_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
_SHIM_HEADER = "csrc/kda/tf32/cake_kda_prefill_host_shim.cuh"

# Host argument plans shared by the generated bindings: (kind, name) per argument.
ARG_PLANS = {
    "P00": (
        ("buffer", "q"),
        ("tma_buffer", "q_tma"),
        ("buffer", "k"),
        ("tma_buffer", "k_tma"),
        ("buffer", "v"),
        ("tma_buffer", "v_tma"),
        ("buffer", "g"),
        ("tma_buffer", "g_tma"),
        ("buffer", "beta"),
        ("tma_buffer", "beta_tma"),
        ("buffer", "A_log"),
        ("buffer", "dt_bias"),
        ("buffer", "cu_seqlens"),
        ("buffer", "seq_order"),
        ("buffer", "initial_state"),
        ("buffer", "out"),
        ("tma_buffer", "out_tma"),
        ("buffer", "final_state"),
        ("parameter", "num_heads"),
        ("parameter", "use_initial_state"),
        ("parameter", "store_final_state"),
        ("parameter", "scale"),
        ("parameter", "lower_bound"),
        ("parameter", "state_indices_addr"),
        ("parameter", "state_checkpoints_addr"),
        ("parameter", "checkpoint_cu_starts_addr"),
        ("parameter", "beta_token_stride"),
        ("parameter", "state_slot_stride"),
        ("parameter", "use_state_indices"),
        ("parameter", "checkpoint_every_n_tokens"),
        ("buffer", "cu_chunk_offsets"),
        ("buffer", "chunk_state"),
        ("buffer", "state_checkpoint_needed"),
        ("buffer", "tape_qd"),
        ("buffer", "tape_kd"),
        ("buffer", "tape_kr"),
        ("buffer", "tape_j"),
        ("buffer", "tape_restore_factor"),
        ("buffer", "tape_e"),
        ("buffer", "tape_x"),
        ("buffer", "tape_r"),
        ("buffer", "norm_inv_out"),
        ("buffer", "decay_out"),
        ("buffer", "beta_active_out"),
        ("buffer", "initial_state_f32"),
        ("buffer", "zero_workspace"),
        ("parameter", "zero_words"),
        ("parameter", "num_sequences"),
        ("tma_buffer", "state_checkpoints_tma"),
        ("buffer", "final_state_f32"),
        ("parameter", "g_token_stride"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P01": (
        ("buffer", "items"),
        ("parameter", "num_heads"),
        ("buffer", "carry_hi"),
        ("tma_buffer", "carry_hi_tma"),
        ("buffer", "carry_lo"),
        ("tma_buffer", "carry_lo_tma"),
        ("buffer", "maps"),
        ("tma_buffer", "maps_tma"),
        ("buffer", "map_final"),
        ("tma_buffer", "map_final_tma"),
        ("buffer", "op_qd"),
        ("buffer", "op_kd"),
        ("buffer", "op_ft"),
        ("buffer", "op_inv"),
        ("buffer", "op_vec"),
        ("buffer", "out"),
        ("tma_buffer", "out_tma"),
        ("buffer", "rows"),
        ("tma_buffer", "rows_tma"),
        ("buffer", "row_starts"),
        ("buffer", "final"),
        ("tma_buffer", "final_tma"),
        ("buffer", "pair"),
        ("tma_buffer", "pair_tma"),
        ("buffer", "dpair"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P02": (
        ("buffer", "q"),
        ("tma_buffer", "q_tma"),
        ("buffer", "k"),
        ("tma_buffer", "k_tma"),
        ("buffer", "v"),
        ("tma_buffer", "v_tma"),
        ("buffer", "g"),
        ("tma_buffer", "g_tma"),
        ("buffer", "beta"),
        ("tma_buffer", "beta_tma"),
        ("buffer", "A_log"),
        ("buffer", "dt_bias"),
        ("buffer", "cu_seqlens"),
        ("buffer", "seq_order"),
        ("buffer", "initial_state"),
        ("buffer", "out"),
        ("tma_buffer", "out_tma"),
        ("buffer", "final_state"),
        ("parameter", "state_indices_addr"),
        ("parameter", "state_slot_stride"),
        ("parameter", "use_state_indices"),
        ("buffer", "initial_state_f32"),
        ("buffer", "final_state_f32"),
        ("parameter", "state_checkpoints_addr"),
        ("parameter", "checkpoint_cu_starts_addr"),
        ("buffer", "beta_active_out"),
        ("parameter", "beta_token_stride"),
        ("parameter", "g_token_stride"),
        ("parameter", "checkpoint_every_n_tokens"),
        ("parameter", "num_heads"),
        ("parameter", "use_initial_state"),
        ("parameter", "store_final_state"),
        ("parameter", "scale"),
        ("parameter", "lower_bound"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P03": (
        ("buffer", "items"),
        ("parameter", "num_heads"),
        ("buffer", "pair"),
        ("tma_buffer", "pair_tma"),
        ("buffer", "dpair"),
        ("buffer", "maps"),
        ("tma_buffer", "maps_tma"),
        ("buffer", "map_final"),
        ("tma_buffer", "map_final_tma"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P04": (
        ("buffer", "split_state"),
        ("buffer", "map_state_bf16"),
        ("buffer", "carry"),
        ("buffer", "carry_hi"),
        ("buffer", "carry_lo"),
        ("buffer", "final_state"),
        ("parameter", "write_final_state"),
        ("parameter", "num_heads"),
        ("buffer", "part_cu_seqlens"),
        ("buffer", "final_indices"),
        ("parameter", "final_slot_stride"),
        ("buffer", "final_state_bf16"),
        ("buffer", "rows"),
        ("buffer", "row_starts"),
        ("parameter", "write_rows"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P05": (
        ("buffer", "q"),
        ("tma_buffer", "q_tma"),
        ("buffer", "k"),
        ("tma_buffer", "k_tma"),
        ("buffer", "raw_gate"),
        ("tma_buffer", "raw_gate_tma"),
        ("buffer", "beta_logits"),
        ("buffer", "beta_active_f32"),
        ("tma_buffer", "beta_logits_tma"),
        ("buffer", "a_log"),
        ("buffer", "dt_bias"),
        ("buffer", "cu_seqlens"),
        ("buffer", "cu_chunks"),
        ("buffer", "chunk_to_seq"),
        ("buffer", "ws_qd"),
        ("tma_buffer", "ws_qd_tma"),
        ("buffer", "ws_kd"),
        ("tma_buffer", "ws_kd_tma"),
        ("buffer", "ws_w"),
        ("tma_buffer", "ws_w_tma"),
        ("buffer", "ws_qk_t"),
        ("buffer", "ws_diag"),
        ("parameter", "total_chunks"),
        ("parameter", "num_heads"),
        ("parameter", "gate_lower_bound"),
        ("parameter", "beta_token_stride"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P06": (
        ("buffer", "q"),
        ("tma_buffer", "q_tma"),
        ("buffer", "k"),
        ("tma_buffer", "k_tma"),
        ("buffer", "raw_gate"),
        ("tma_buffer", "raw_gate_tma"),
        ("buffer", "beta_logits"),
        ("buffer", "beta_active_f32"),
        ("parameter", "affine_cache_token_offset"),
        ("parameter", "affine_cache_part_offset"),
        ("tma_buffer", "beta_logits_tma"),
        ("buffer", "a_log"),
        ("buffer", "dt_bias"),
        ("buffer", "cu_seqlens"),
        ("buffer", "seq_order"),
        ("buffer", "v"),
        ("tma_buffer", "v_tma"),
        ("buffer", "out"),
        ("buffer", "initial_state"),
        ("buffer", "final_state"),
        ("buffer", "initial_state_f32"),
        ("buffer", "final_state_f32"),
        ("parameter", "state_indices_addr"),
        ("parameter", "state_slot_stride"),
        ("parameter", "use_state_indices"),
        ("parameter", "use_initial_state"),
        ("parameter", "store_final_state"),
        ("tma_buffer", "state_checkpoints_tma"),
        ("buffer", "state_checkpoints"),
        ("buffer", "checkpoint_cu_starts"),
        ("parameter", "checkpoint_every_n_tokens"),
        ("parameter", "scale"),
        ("parameter", "num_heads"),
        ("parameter", "gate_lower_bound"),
        ("parameter", "beta_token_stride"),
        ("buffer", "task_ids"),
        ("buffer", "task_offsets"),
        ("buffer", "task_token_starts"),
        ("buffer", "task_token_counts"),
        ("buffer", "task_state_sources"),
        ("buffer", "task_state_destinations"),
        ("buffer", "mid_state_f32"),
        ("buffer", "mid_state_ready"),
        ("tma_buffer", "owner_packet_tma"),
        ("tma_buffer", "owner_packet_tail_tma"),
        ("buffer", "map_output_f32"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P07": (
        ("buffer", "ws_qd"),
        ("tma_buffer", "ws_qd_tma"),
        ("buffer", "ws_kd"),
        ("tma_buffer", "ws_kd_tma"),
        ("buffer", "ws_w"),
        ("tma_buffer", "ws_w_tma"),
        ("buffer", "ws_qk"),
        ("tma_buffer", "ws_qk_tma"),
        ("buffer", "ws_diag"),
        ("tma_buffer", "ws_diag_tma"),
        ("buffer", "v"),
        ("tma_buffer", "v_tma"),
        ("buffer", "cu_seqlens"),
        ("buffer", "cu_chunks"),
        ("buffer", "seq_order"),
        ("buffer", "initial_state"),
        ("buffer", "out"),
        ("tma_buffer", "out_tma"),
        ("buffer", "final_state"),
        ("parameter", "num_heads"),
        ("parameter", "use_initial_state"),
        ("parameter", "store_final_state"),
        ("parameter", "scale"),
        ("parameter", "state_indices_addr"),
        ("parameter", "state_slot_stride"),
        ("parameter", "use_state_indices"),
        ("buffer", "initial_state_f32"),
        ("buffer", "final_state_f32"),
        ("buffer", "state_checkpoints"),
        ("buffer", "checkpoint_cu_starts"),
        ("parameter", "checkpoint_every_n_tokens"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P08": (
        ("buffer", "split_state"),
        ("buffer", "map_state_bf16"),
        ("buffer", "carry"),
        ("buffer", "final_state"),
        ("parameter", "write_final_state"),
        ("parameter", "num_heads"),
        ("buffer", "part_cu_seqlens"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
    "P09": (
        ("buffer", "carry"),
        ("tma_buffer", "carry_tma"),
        ("buffer", "coefficients"),
        ("tma_buffer", "coefficients_tma"),
        ("buffer", "out"),
        ("buffer", "token_starts"),
        ("buffer", "token_counts"),
        ("buffer", "part_ids"),
        ("parameter", "num_heads"),
        ("workspace", "tma_descriptor_workspace"),
        ("grid", "grid_x"),
        ("grid", "grid_y"),
        ("grid", "grid_z"),
    ),
}

# module name -> (family directory, source digest, architectures, argument plan,
# TMA descriptor workspace bytes). Sources are csrc/kda/<dir>/cake_kda_<dir>_<digest>_{kernel,binding}.cu.
_MODULE_ROWS = {
    "cake_kda_bf16_02c6dae5d13691ae05c3": (
        "bf16",
        "62fb6024a358c8cb62daad22e105799391696c9c53a5b386b789d0c5e5884b6f",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_03ea50d078dbf34830bb": (
        "bf16",
        "c3f815e1e149d8b18ae19a166c7e11b5177e58f67b7809304933de6d77df25b1",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_0e474910f4729fac1027": (
        "bf16",
        "b917b554912ff27f758e9bddb2664d4064234c45b56a1cdcbf2325a363ce8d2f",
        ("sm_100a", "sm_103a"),
        "P01",
        1024,
    ),
    "cake_kda_bf16_14d1b3745aa9f0079482": (
        "bf16",
        "0c7f31828dd9e93e3d5522907aa28fabf2b9d410431924dd9355aaa96132173b",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_15be1db8a50ee4d9fd16": (
        "bf16",
        "51243fb161d66ee57bbc73c409dd622057f2e5b64445a31c59da251ec448568a",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_19a78237678977631fc7": (
        "bf16",
        "827cef267457f6d6b1cef7da13a149b5d57e9aab3b3dd88c3f3d83208d00031b",
        ("sm_100a", "sm_103a"),
        "P02",
        768,
    ),
    "cake_kda_bf16_2064d4aca94aef841cf8": (
        "bf16",
        "b5fc008b09efaaec7552c7de81c1b2cf2ca39321171a96d7203388c3dc32c578",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_2b11bfa0df216f854964": (
        "bf16",
        "d4ff22207792366eab60b789e452c65302005372ca900f47ed636c5332834dd9",
        ("sm_100a", "sm_103a"),
        "P03",
        384,
    ),
    "cake_kda_bf16_2b5f2c6b7cfc10ffa244": (
        "bf16",
        "7e2e6ee2298c5db04889ef1104e834ad93dec65ab462f9c53fe0603aaa1cdde0",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_3bd0815de00cd1122604": (
        "bf16",
        "9fce75ac82991b5dcb6cfd734aa15bf08717a4c57cb6fc20ed1d5d003926c020",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_496bf5563f7746744ffe": (
        "bf16",
        "1693d074ef717aef606cf66deaf3e10821bc7c9e511e7c3a4ff873a0e0b7723c",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_4ab394a09c63f264d7bd": (
        "bf16",
        "23fd208fea39c017ecc91084ea0658c7ff954968435c8b9f38076dcbcf768d6f",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_4e7e4609dadc8f6bcdfe": (
        "bf16",
        "a699400d4afdd0ce83bbfcb640e7a935de16229b66bfa5a31af5f0c435eda9b1",
        ("sm_100a", "sm_103a"),
        "P04",
        0,
    ),
    "cake_kda_bf16_50c0f288f4c8d7ca03a8": (
        "bf16",
        "fdbf49039086ea0ab88c0722741e49314dfc4289f27296f14121ef8ae81ecf09",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_51e813006f17147a02a5": (
        "bf16",
        "cacd2d67bfb19acab4380c4d3e096a6a0d1c3d0eefa6a274789fc43c54eff7b9",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_5535e79c76a18330f6f1": (
        "bf16",
        "0f326333dfcf09bc9e6fa72ab63b4cfe06869865b5087cd313ecfac5b404a0db",
        ("sm_100a", "sm_103a"),
        "P02",
        768,
    ),
    "cake_kda_bf16_5639519f5c942ed86e4f": (
        "bf16",
        "4a2354a3ebea77e12c8f0596f295444041ea67d8558c96d0cea41444efc99b6d",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_5926ccfc26ae75ec8391": (
        "bf16",
        "d5fd10ada12686e56a75ad17c67fa1c5e59dd3b18093b887914010f754f6f778",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_5be344cbddc29563f1ee": (
        "bf16",
        "e62d89eb5037fad1c55b9366b59c7657188d8a0e1ab70f1ae8ebb26ce8a22ce3",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_73414b477c864609350c": (
        "bf16",
        "fe0a65e1b6040161f67f802f7ff38b7b9d89e03c24664d8fe2f0e13826735185",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_7386cf449b08cfe77567": (
        "bf16",
        "8ec1ab7949d4176a144f16708296b3603d7edd934ee823237e77c6d28b284978",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_82dff1c305abe6367034": (
        "bf16",
        "c5b575af972ee37a9e12927304f0b9466edb84a2bf843598e8f7171c767a578b",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_893740628d53688ac269": (
        "bf16",
        "9ca3d761d899c7f54b0bf5b0f3ae2e6ab7b01169fe266afbd558ce0cdee1f8a2",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_898e5dc6fc0e2cc2ac9e": (
        "bf16",
        "c676a5f28b6f7d3b64021ed408ae8ed711961b8eb9e13cccd7a6df8088620cb1",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_8e9c0962021058a3005d": (
        "bf16",
        "f7694bdba544d4a92785b6824762eefe2f8b1e25602cb7b68070d19044a7e2ad",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_9556ac079590cd311172": (
        "bf16",
        "d4870b4beceb26b9378cbaa162941b1f3060a3a34aacb01e7f66b9a1a73809dd",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_96dd33788d497a79e142": (
        "bf16",
        "fc2c343905a5694cae94eae5727ae03dd02643f04b7380d81f88d048fc4238e8",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_9a336afc613e4f9ee1c7": (
        "bf16",
        "19e15f849f6972486b140dae77d700b900c5cd8a1eb5113615dc141cd3056068",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_acd84ad6d68fb21576b7": (
        "bf16",
        "92eb50f4e575c45052d9b1a707ab4043bb5e39712e2d704eca7f75bf8bdc19f5",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_b38d251875e94cb4aaa6": (
        "bf16",
        "f0285c78e04876b17bdb0ae69518e6fb429554c80cec156e44f39744fad2b9b6",
        ("sm_100a", "sm_103a"),
        "P04",
        0,
    ),
    "cake_kda_bf16_b52b37142605bb183504": (
        "bf16",
        "bccc43c204dd1e07c9606c2ed06ffa8c8fbc2d2d6ec7d704f4997331b6a7d1a5",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_b7a36d7e13872a35abe8": (
        "bf16",
        "ec3a314576ec3c63a428fb85f80f60ccbd95e5d3d5da87aa4ab229a11b3a01d9",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_bd558220666f207e0109": (
        "bf16",
        "c1bf3d46c0f9d6b6dec741dad6aeeafe388a8e9f04b272b411167920ec3ddd6f",
        ("sm_100a", "sm_103a"),
        "P01",
        1024,
    ),
    "cake_kda_bf16_c410d3932af8bc84666e": (
        "bf16",
        "d6d61ba137a855707d222768a32ed35304152d016f6ab54ce0b4cab6bbad35cf",
        ("sm_103a",),
        "P00",
        896,
    ),
    "cake_kda_bf16_c4f30cd09e56ce40d46c": (
        "bf16",
        "3ffa95259bae3a488de811461c72d30d3a9896363e2f67a084d6bb41173e8ab7",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_c528a0e1a9896ea82166": (
        "bf16",
        "fd0784553d4d888edd2a65c039895ca0f05bc340220707c98630f0a631c7ae40",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_c7c8fcf8dd88d37499d3": (
        "bf16",
        "cecf9ec32ef34d6b70c2978487715bf227593448435cf0e163d23c662d039d0f",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_cb22ba3c5cee4d5fa5fd": (
        "bf16",
        "c383123b8290c37224f68bb4789ff6de356db191189bbb34b10f2467f084487f",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_ce67a310d4f8a80a2cbf": (
        "bf16",
        "32d53b4c71a89e3688e42f9f1a4e3c9b06c6cbb6f472147999b7b30cac321d07",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_d1ce1e4b43672482991c": (
        "bf16",
        "0ad7db8816e35db97942e073e53526bc015bd67bae3d434d5f589edd876a6a0b",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_d22df3843e4ae69e26a7": (
        "bf16",
        "f9a2ce904d26a929b4235dc2abbc5c1bbe04bcc29b0743476863c129aa645de5",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_d3fc3337b10cdf6bf7d6": (
        "bf16",
        "1dcd1f92c8941333cedbaa26d406b60dded4804d87a88feac472028a6630c861",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_db50c13c6d4aa462575c": (
        "bf16",
        "e3dc29ea3e0a8d619fc7d75bfa0d5849bb1579bf87b47da7d9ed91dd806c690a",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_e3cd435a92ef7a053b72": (
        "bf16",
        "aee36048c045c3eb94bf80d0fe2aab882c2db78b448f637c8cadad70b7808fdd",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_eeb61146f150d9e0d825": (
        "bf16",
        "fbb569221b26b16a2963d7cb34998e0f6851fbe66c5600c3a5eadc519c1bf66a",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_f66adfecf2022fc198b6": (
        "bf16",
        "2cf036fe11faad73e7ac87aa51fedb5df7738b4550c580e58f8e2d15fdc12f23",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_f74a288bb51b335a1ccf": (
        "bf16",
        "c7d9c377486ac3b2f9b898896f55593fb6b6807a63c9c76813d5c77ce98e05ef",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_f7cf9b795943c5803edb": (
        "bf16",
        "dee4a03f1940f39613b5101abefd3b940c72c13672c0fa2201cf66c7aee7addd",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_fb073f88984adc2962a9": (
        "bf16",
        "bc1d9243080ec7944715f1471491b5d4b65cccb570f720a4c7ffc9943eb40bf8",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_fc66c809d573029352f7": (
        "bf16",
        "43dd64bbdf1bc9d8102aad406793ec3e7d804da4bb2cb50f695ca89248be9f8b",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_bf16_fd04381524dba62b317d": (
        "bf16",
        "395869bcfb1fafe283671aa154b4c52f4ee6cc0ba227b70f748cfa457be77207",
        ("sm_100a", "sm_103a"),
        "P00",
        896,
    ),
    "cake_kda_tf32_064455c977597cb9ed09": (
        "tf32",
        "4c99177c07ace7591e287d6d59191c460b0038d21d7697a719b133f5316847ff",
        ("sm_100a", "sm_103a"),
        "P05",
        896,
    ),
    "cake_kda_tf32_0fd45fbaac397bbacead": (
        "tf32",
        "12803094a6532d7d485f52f8a46877b86249c37c80feaf014cbf182dc84dc5b8",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_110a02399894fe49f121": (
        "tf32",
        "095bd9aa2cdee2c2c4c385bbd6b06a7f08910cb4e066922e5a9cd01a42a9d321",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_116b3374286bc679aa2a": (
        "tf32",
        "ea088cb31c6390f9a82f101f421a357e121f55e292b08c186f237b5c974fa3b6",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_1c5791474807c70f0769": (
        "tf32",
        "3d86251533a484c214da6dbd74f67a85119edf8c03d8c7ecad2cbef7f423a1b0",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_1c5969b9cc4b5ffb2477": (
        "tf32",
        "de0b648b181aea585317a33cbb59f25b403246d990627eb3e6cdab1a9c874ab0",
        ("sm_100a", "sm_103a"),
        "P07",
        896,
    ),
    "cake_kda_tf32_20fe382d8e26baba025f": (
        "tf32",
        "8c9b5bf0fe92d7019f7d08f543e32a256a7e03d683da4d7d3b5ab288ee71d730",
        ("sm_100a", "sm_103a"),
        "P07",
        896,
    ),
    "cake_kda_tf32_29e446f2642575503eea": (
        "tf32",
        "13d49901e42bbadbd8e8a22bffe61095c491acbfc893466a0dd647543a607a80",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_32cd512ec43c2530f170": (
        "tf32",
        "de45e5779221499b4fca8178ae5978c301d146b0f716eb1c561363d7f6e327d7",
        ("sm_100a", "sm_103a"),
        "P05",
        896,
    ),
    "cake_kda_tf32_380f74127da276b25cf1": (
        "tf32",
        "83d5b769f4ecd1825483a3d09349e9e95d14d3fe125dfb25559cfc17dad6a705",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_3a4f2589ba67a3e6401c": (
        "tf32",
        "361d3bf46ffa5ba381c2162a42b355cfbfa8ffeb0159d19ca01557f27c0810a0",
        ("sm_100a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_3b4bd35adcfa0f948de2": (
        "tf32",
        "2fe63c12a8b7535f696d880d5a9cfe26a51eb33a21585f698d09d089d7022b38",
        ("sm_100a", "sm_103a"),
        "P08",
        0,
    ),
    "cake_kda_tf32_3e3a1d311719d83b943a": (
        "tf32",
        "0b9e331c323128496f9c052c2db2be589d5d0893303854d1ad4052d64871a106",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_4374d24862aa57344e4f": (
        "tf32",
        "49252b8acc3d5b1acefaed3f94b8367860dbaadd9fe3f9b99d32883f1e60a7d4",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_637eee609dc3847cb602": (
        "tf32",
        "4409b00951db691ca33b764132c53a194cabef81f6f94dde882398b1fdc2e1da",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_656dad29691bf799ad49": (
        "tf32",
        "ff102053bf9f119f38635d80d95a8baf879a2bae58b5c633e2edbe945bed5162",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_67623610c7d269c8e455": (
        "tf32",
        "4434066f7f1b6aae62ab53df58e3987bf70dc53d8c5a87d196461674cbcf4a9e",
        ("sm_100a", "sm_103a"),
        "P09",
        256,
    ),
    "cake_kda_tf32_688090ae18b66d4aae0f": (
        "tf32",
        "c32f7c95ae09b258fda8b0619acf43969b48c31be2f0bec1efec7ad7b2e54852",
        ("sm_100a", "sm_103a"),
        "P07",
        896,
    ),
    "cake_kda_tf32_8bd403eee9a4578a3f2a": (
        "tf32",
        "1b9a9cf31e75651065d599f14529afb60899a3e29b045529c44ad98d91618ec8",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_9a6496ea94cd4302f8cd": (
        "tf32",
        "39a2d34aff6b2793f048ebaaafc57abde68ac0ec0d9f4faaf7f567c69c35c398",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_9b03dea068c354118714": (
        "tf32",
        "0aaa66f8b0fff559171a9d75b17859de6cf328b991c01274a3419eb656c327bd",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_9f57d4b208cc67725a80": (
        "tf32",
        "cf85be937c60c2f43b0cc374c2e4ac8e19bf9f973962d7224a0094d93786c258",
        ("sm_100a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_a6e1e706ef047611b36d": (
        "tf32",
        "667be027941f89be5127b8d4fc6c7bf9a8203b314d6c550fec2b07bf6e84fb05",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_a755dd9542165c5ee3c9": (
        "tf32",
        "ac2bd32c98dcfb11feed3b3ccac14797e2625b35c14ba59475812ec2641971a2",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_baf76efae0b1113677ea": (
        "tf32",
        "b6beda89dac6ab15d4e6112deae14c4a0eddb6863a26e546b9378f7b0824305c",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_c48c48b5db0e008fd946": (
        "tf32",
        "ed27d260f4765acbb3c477d97df9ea0e8f2813abc82edba26d7c65b5aabea2df",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_cc9e6f2d5ed58f1f3084": (
        "tf32",
        "572fd9b9998720563112d5c68df0101895e01e09bbdb475d52448d0c27f5d7fd",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_d4bf3dee30e8e3c2fb80": (
        "tf32",
        "aa4d61bbbcd9a21d62b25651d953d1239bdd239892e3d30b101e0ab2e429ab62",
        ("sm_100a", "sm_103a"),
        "P07",
        896,
    ),
    "cake_kda_tf32_d8cb0becd1d3911b0b8d": (
        "tf32",
        "dd0f76773ac27a364fcf5df27f69697e3b1af22c2f77595b41b23945752e90f8",
        ("sm_100a", "sm_103a"),
        "P07",
        896,
    ),
    "cake_kda_tf32_e21b8a42c1cf4a1b7326": (
        "tf32",
        "d30d8c96eeb096e09d369fb4362f7b004019396996cbf154515b6bddd784ec27",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_e6ba645edc4387b89652": (
        "tf32",
        "347cf634794cd5219f51e176a34ef4edd2b16544078450cc38aa1a49d2116253",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_e8117b26c217f6a939a8": (
        "tf32",
        "9839e3aa0018bebfd6602ad870087e6569a7a9fc11652a2c298ae69a3151df3c",
        ("sm_103a",),
        "P06",
        1024,
    ),
    "cake_kda_tf32_fb720910739b9a8d69d1": (
        "tf32",
        "5631aff0c79ab029e651a941cb484c602c9b853fccab131b625b56d5c4a4ce27",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
    "cake_kda_tf32_ff7bf41421e07fc5a457": (
        "tf32",
        "8c3b14205a9ffc1d6bb1736e329eb5d7a8f0d6e562e4bfae418f027e14352380",
        ("sm_100a", "sm_103a"),
        "P06",
        1024,
    ),
}


def _module_record(directory, digest, arches, plan, tma_workspace_bytes):
    stem = f"cake_kda_{directory}_{digest}"
    return {
        "arches": arches,
        "sources": [
            f"csrc/kda/{directory}/{stem}_kernel.cu",
            f"csrc/kda/{directory}/{stem}_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [list(slot) for slot in ARG_PLANS[plan]],
        "tma_workspace_bytes": tma_workspace_bytes,
        "cache_name": stem,
    }


MODULES = {name: _module_record(*row) for name, row in _MODULE_ROWS.items()}

# Factory specializations per schedule family. A row is
# (positional args, kwarg-signature index, kwargs that differ from the family
# defaults, module name or {arch: module name}).
_FACTORY_DEFAULTS = {
    "compiled_affine_output_projection": {},
    "compiled_bf16_fused_m128": {
        "active_beta_f32": False,
        "affine_main_indexed_initial": False,
        "affine_main_indexed_initial_bf16": False,
        "affine_operator_export": False,
        "backend": "cuda_cpp",
        "checkpoint_accumulate": True,
        "checkpoint_dtype_is_fp32": True,
        "checkpoint_tma": False,
        "early_n32_state_pack": False,
        "fp32_state_carrier": True,
        "gate_kind": "lower_bound",
        "generic_register_inverse": False,
        "logical_page64": False,
        "n16_short_four_stage": False,
        "n32_ft_slab": True,
        "n32_prediction_first": False,
        "pair_packed_beta": False,
        "pdl_publish_final_state": False,
        "pdl_wait_initial_state_f32": False,
        "scalar_beta": False,
        "serving_native_abi": True,
        "state_dtype_is_fp32": True,
        "tensor_state_decay": False,
    },
    "compiled_bf16_fused_m64": {
        "active_beta_f32": False,
        "state_dtype_is_fp32": True,
    },
    "compiled_flashkda_affine_apply_m128": {
        "pairmap": True,
    },
    "compiled_flashkda_map_prefix_m128": {},
    "compiled_flashkda_split_scan_bf16_m128": {
        "backend": "cuda_cpp",
        "carry_hilo": True,
        "compute_dtype": "bf16",
        "use_pdl": True,
    },
    "compiled_tf32_bt16_chain_m64_fp32_state": {
        "compact_output": False,
        "serving_native_abi": True,
        "split_prediction": False,
        "write_checkpoints": True,
    },
    "compiled_tf32_bt16_prepare": {
        "active_beta_f32": False,
    },
    "compiled_tf32_fused_n32": {
        "active_beta_f32": False,
        "affine_factor_cache": 0,
        "affine_main_indexed_initial": False,
        "affine_main_indexed_initial_bf16": False,
        "affine_map_only": False,
        "affine_map_output": False,
        "checkpoint_tma": False,
        "compact_state": False,
        "owner_helpers": 7,
        "pdl_publish_final_state": False,
        "pdl_wait_initial_state_f32": False,
        "prep_stages": 3,
        "round_tf32_operands": False,
        "state_dtype_is_fp32": True,
        "unbounded_softplus": False,
        "value_rows": 128,
        "write_checkpoints": True,
    },
}

_FACTORY_SIGNATURES = {
    "compiled_affine_output_projection": ((),),
    "compiled_bf16_fused_m128": (
        (
            "active_beta_f32",
            "affine_main_indexed_initial",
            "affine_main_indexed_initial_bf16",
            "affine_operator_export",
            "backend",
            "checkpoint_tma",
            "early_n32_state_pack",
            "gate_kind",
            "generic_register_inverse",
            "logical_page64",
            "n16_short_four_stage",
            "n32_ft_slab",
            "n32_prediction_first",
            "pair_packed_beta",
            "pdl_publish_final_state",
            "pdl_wait_initial_state_f32",
            "scalar_beta",
            "serving_native_abi",
            "state_dtype_is_fp32",
            "tensor_state_decay",
        ),
        (
            "active_beta_f32",
            "affine_main_indexed_initial",
            "affine_main_indexed_initial_bf16",
            "affine_operator_export",
            "backend",
            "checkpoint_accumulate",
            "checkpoint_dtype_is_fp32",
            "checkpoint_tma",
            "early_n32_state_pack",
            "gate_kind",
            "generic_register_inverse",
            "logical_page64",
            "n16_short_four_stage",
            "n32_ft_slab",
            "n32_prediction_first",
            "pair_packed_beta",
            "pdl_publish_final_state",
            "pdl_wait_initial_state_f32",
            "scalar_beta",
            "serving_native_abi",
            "state_dtype_is_fp32",
            "tensor_state_decay",
        ),
        (
            "active_beta_f32",
            "affine_main_indexed_initial",
            "affine_main_indexed_initial_bf16",
            "affine_operator_export",
            "backend",
            "checkpoint_dtype_is_fp32",
            "checkpoint_tma",
            "early_n32_state_pack",
            "gate_kind",
            "generic_register_inverse",
            "logical_page64",
            "n16_short_four_stage",
            "n32_ft_slab",
            "n32_prediction_first",
            "pair_packed_beta",
            "pdl_publish_final_state",
            "pdl_wait_initial_state_f32",
            "scalar_beta",
            "serving_native_abi",
            "state_dtype_is_fp32",
            "tensor_state_decay",
        ),
        (
            "active_beta_f32",
            "affine_main_indexed_initial",
            "affine_main_indexed_initial_bf16",
            "affine_operator_export",
            "backend",
            "checkpoint_tma",
            "early_n32_state_pack",
            "fp32_state_carrier",
            "gate_kind",
            "generic_register_inverse",
            "logical_page64",
            "n16_short_four_stage",
            "n32_ft_slab",
            "n32_prediction_first",
            "pair_packed_beta",
            "pdl_publish_final_state",
            "pdl_wait_initial_state_f32",
            "scalar_beta",
            "serving_native_abi",
            "state_dtype_is_fp32",
            "tensor_state_decay",
        ),
    ),
    "compiled_bf16_fused_m64": (("active_beta_f32", "state_dtype_is_fp32"),),
    "compiled_flashkda_affine_apply_m128": (("pairmap",), ()),
    "compiled_flashkda_map_prefix_m128": ((),),
    "compiled_flashkda_split_scan_bf16_m128": (
        ("backend", "carry_hilo", "compute_dtype", "use_pdl"),
        ("backend", "compute_dtype", "use_pdl"),
    ),
    "compiled_tf32_bt16_chain_m64_fp32_state": (
        (
            "compact_output",
            "serving_native_abi",
            "split_prediction",
            "write_checkpoints",
        ),
    ),
    "compiled_tf32_bt16_prepare": (("active_beta_f32",),),
    "compiled_tf32_fused_n32": (
        (
            "active_beta_f32",
            "affine_factor_cache",
            "affine_main_indexed_initial",
            "affine_main_indexed_initial_bf16",
            "affine_map_only",
            "affine_map_output",
            "checkpoint_tma",
            "compact_state",
            "pdl_publish_final_state",
            "pdl_wait_initial_state_f32",
            "prep_stages",
            "round_tf32_operands",
            "state_dtype_is_fp32",
            "unbounded_softplus",
            "value_rows",
            "write_checkpoints",
        ),
        (
            "active_beta_f32",
            "owner_helpers",
            "prep_stages",
            "round_tf32_operands",
            "state_dtype_is_fp32",
            "unbounded_softplus",
            "value_rows",
            "write_checkpoints",
        ),
    ),
}

_FACTORY_ROWS = {
    "compiled_affine_output_projection": [
        ((), 0, {}, "cake_kda_tf32_67623610c7d269c8e455"),
    ],
    "compiled_bf16_fused_m128": [
        (
            (16,),
            0,
            {"n32_ft_slab": False, "scalar_beta": True},
            "cake_kda_bf16_b7a36d7e13872a35abe8",
        ),
        (
            (16,),
            0,
            {"checkpoint_tma": True, "n32_ft_slab": False, "scalar_beta": True},
            "cake_kda_bf16_51e813006f17147a02a5",
        ),
        (
            (16,),
            0,
            {
                "active_beta_f32": True,
                "checkpoint_tma": True,
                "n32_ft_slab": False,
                "scalar_beta": True,
            },
            "cake_kda_bf16_3bd0815de00cd1122604",
        ),
        (
            (32,),
            1,
            {"gate_kind": "unbounded_softplus", "pdl_wait_initial_state_f32": True},
            "cake_kda_bf16_e3cd435a92ef7a053b72",
        ),
        (
            (32,),
            1,
            {
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_wait_initial_state_f32": True,
            },
            "cake_kda_bf16_5926ccfc26ae75ec8391",
        ),
        (
            (32,),
            2,
            {"gate_kind": "unbounded_softplus", "n32_ft_slab": False},
            "cake_kda_bf16_893740628d53688ac269",
        ),
        (
            (32,),
            2,
            {
                "gate_kind": "unbounded_softplus",
                "n32_ft_slab": False,
                "pair_packed_beta": True,
            },
            "cake_kda_bf16_496bf5563f7746744ffe",
        ),
        (
            (32,),
            2,
            {"gate_kind": "unbounded_softplus", "pdl_wait_initial_state_f32": True},
            "cake_kda_bf16_c7c8fcf8dd88d37499d3",
        ),
        (
            (32,),
            2,
            {
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_wait_initial_state_f32": True,
            },
            "cake_kda_bf16_b52b37142605bb183504",
        ),
        (
            (32,),
            3,
            {"gate_kind": "unbounded_softplus", "pdl_wait_initial_state_f32": True},
            "cake_kda_bf16_96dd33788d497a79e142",
        ),
        (
            (32,),
            3,
            {
                "gate_kind": "unbounded_softplus",
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_acd84ad6d68fb21576b7",
        ),
        (
            (32,),
            3,
            {
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_wait_initial_state_f32": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_73414b477c864609350c",
        ),
        (
            (32,),
            3,
            {
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "serving_native_abi": False,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_ce67a310d4f8a80a2cbf",
        ),
        (
            (32,),
            3,
            {
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_82dff1c305abe6367034",
        ),
        ((32,), 0, {"n32_ft_slab": False}, "cake_kda_bf16_02c6dae5d13691ae05c3"),
        (
            (32,),
            0,
            {"generic_register_inverse": True, "n32_ft_slab": False},
            "cake_kda_bf16_2064d4aca94aef841cf8",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "n32_ft_slab": False,
                "scalar_beta": True,
            },
            "cake_kda_bf16_5be344cbddc29563f1ee",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "n32_ft_slab": False,
                "pair_packed_beta": True,
                "scalar_beta": True,
            },
            "cake_kda_bf16_9556ac079590cd311172",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "n32_ft_slab": False,
                "n32_prediction_first": True,
            },
            "cake_kda_bf16_c410d3932af8bc84666e",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pdl_wait_initial_state_f32": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_8e9c0962021058a3005d",
        ),
        (
            (32,),
            0,
            {"generic_register_inverse": True, "pdl_wait_initial_state_f32": True},
            "cake_kda_bf16_2b5f2c6b7cfc10ffa244",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pdl_wait_initial_state_f32": True,
                "scalar_beta": True,
            },
            "cake_kda_bf16_f66adfecf2022fc198b6",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "serving_native_abi": False,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_03ea50d078dbf34830bb",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_5639519f5c942ed86e4f",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "scalar_beta": True,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_f74a288bb51b335a1ccf",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pair_packed_beta": True,
                "pdl_wait_initial_state_f32": True,
                "scalar_beta": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_14d1b3745aa9f0079482",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pair_packed_beta": True,
                "pdl_wait_initial_state_f32": True,
                "scalar_beta": True,
            },
            "cake_kda_bf16_db50c13c6d4aa462575c",
        ),
        (
            (32,),
            0,
            {
                "generic_register_inverse": True,
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "pdl_wait_initial_state_f32": True,
                "scalar_beta": True,
                "serving_native_abi": False,
                "state_dtype_is_fp32": False,
            },
            "cake_kda_bf16_eeb61146f150d9e0d825",
        ),
        (
            (32,),
            0,
            {"gate_kind": "unbounded_softplus", "n32_ft_slab": False},
            "cake_kda_bf16_9a336afc613e4f9ee1c7",
        ),
        (
            (32,),
            0,
            {
                "early_n32_state_pack": True,
                "generic_register_inverse": True,
                "n32_ft_slab": False,
                "scalar_beta": True,
            },
            "cake_kda_bf16_fd04381524dba62b317d",
        ),
        (
            (32,),
            2,
            {
                "affine_main_indexed_initial": True,
                "gate_kind": "unbounded_softplus",
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_fb073f88984adc2962a9",
        ),
        (
            (32,),
            2,
            {
                "affine_main_indexed_initial": True,
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_fc66c809d573029352f7",
        ),
        (
            (32,),
            3,
            {
                "affine_main_indexed_initial": True,
                "gate_kind": "unbounded_softplus",
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_898e5dc6fc0e2cc2ac9e",
        ),
        (
            (32,),
            3,
            {
                "affine_main_indexed_initial": True,
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_d22df3843e4ae69e26a7",
        ),
        (
            (32,),
            0,
            {
                "affine_main_indexed_initial": True,
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_c528a0e1a9896ea82166",
        ),
        (
            (32,),
            0,
            {
                "affine_main_indexed_initial": True,
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_50c0f288f4c8d7ca03a8",
        ),
        (
            (32,),
            0,
            {
                "affine_main_indexed_initial": True,
                "generic_register_inverse": True,
                "pdl_publish_final_state": True,
                "scalar_beta": True,
            },
            "cake_kda_bf16_4ab394a09c63f264d7bd",
        ),
        (
            (32,),
            0,
            {
                "affine_main_indexed_initial": True,
                "generic_register_inverse": True,
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "scalar_beta": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_d1ce1e4b43672482991c",
        ),
        (
            (32,),
            0,
            {
                "affine_main_indexed_initial": True,
                "generic_register_inverse": True,
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "scalar_beta": True,
            },
            "cake_kda_bf16_7386cf449b08cfe77567",
        ),
        (
            (32,),
            2,
            {
                "affine_main_indexed_initial": True,
                "affine_operator_export": True,
                "gate_kind": "unbounded_softplus",
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_15be1db8a50ee4d9fd16",
        ),
        (
            (32,),
            2,
            {
                "affine_main_indexed_initial": True,
                "affine_operator_export": True,
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_cb22ba3c5cee4d5fa5fd",
        ),
        (
            (32,),
            3,
            {
                "affine_main_indexed_initial": True,
                "affine_operator_export": True,
                "gate_kind": "unbounded_softplus",
                "pdl_publish_final_state": True,
            },
            "cake_kda_bf16_d3fc3337b10cdf6bf7d6",
        ),
        (
            (32,),
            3,
            {
                "affine_main_indexed_initial": True,
                "affine_operator_export": True,
                "gate_kind": "unbounded_softplus",
                "pair_packed_beta": True,
                "pdl_publish_final_state": True,
                "serving_native_abi": False,
            },
            "cake_kda_bf16_c4f30cd09e56ce40d46c",
        ),
        (
            (32,),
            0,
            {
                "active_beta_f32": True,
                "generic_register_inverse": True,
                "logical_page64": True,
                "n32_ft_slab": False,
                "scalar_beta": True,
            },
            "cake_kda_bf16_f7cf9b795943c5803edb",
        ),
    ],
    "compiled_bf16_fused_m64": [
        ((), 0, {}, "cake_kda_bf16_5535e79c76a18330f6f1"),
        ((), 0, {"active_beta_f32": True}, "cake_kda_bf16_19a78237678977631fc7"),
    ],
    "compiled_flashkda_affine_apply_m128": [
        ((), 0, {}, "cake_kda_bf16_0e474910f4729fac1027"),
        ((), 1, {}, "cake_kda_bf16_bd558220666f207e0109"),
    ],
    "compiled_flashkda_map_prefix_m128": [
        ((), 0, {}, "cake_kda_bf16_2b11bfa0df216f854964"),
    ],
    "compiled_flashkda_split_scan_bf16_m128": [
        ((), 0, {}, "cake_kda_bf16_4e7e4609dadc8f6bcdfe"),
        ((), 1, {}, "cake_kda_bf16_b38d251875e94cb4aaa6"),
        ((), 1, {"compute_dtype": "tf32"}, "cake_kda_tf32_3b4bd35adcfa0f948de2"),
    ],
    "compiled_tf32_bt16_chain_m64_fp32_state": [
        ((), 0, {"write_checkpoints": False}, "cake_kda_tf32_20fe382d8e26baba025f"),
        ((), 0, {}, "cake_kda_tf32_1c5969b9cc4b5ffb2477"),
        (
            (),
            0,
            {"split_prediction": True, "write_checkpoints": False},
            "cake_kda_tf32_688090ae18b66d4aae0f",
        ),
        ((), 0, {"split_prediction": True}, "cake_kda_tf32_d4bf3dee30e8e3c2fb80"),
        ((), 0, {"compact_output": True}, "cake_kda_tf32_d8cb0becd1d3911b0b8d"),
    ],
    "compiled_tf32_bt16_prepare": [
        ((), 0, {}, "cake_kda_tf32_32cd512ec43c2530f170"),
        ((), 0, {"active_beta_f32": True}, "cake_kda_tf32_064455c977597cb9ed09"),
    ],
    "compiled_tf32_fused_n32": [
        (
            (),
            0,
            {"prep_stages": 1, "write_checkpoints": False},
            "cake_kda_tf32_116b3374286bc679aa2a",
        ),
        ((), 0, {"prep_stages": 1}, "cake_kda_tf32_1c5791474807c70f0769"),
        (
            (),
            0,
            {"prep_stages": 1, "value_rows": 64, "write_checkpoints": False},
            "cake_kda_tf32_e21b8a42c1cf4a1b7326",
        ),
        (
            (),
            0,
            {"prep_stages": 1, "value_rows": 64},
            "cake_kda_tf32_110a02399894fe49f121",
        ),
        (
            (),
            0,
            {"prep_stages": 1, "unbounded_softplus": True},
            "cake_kda_tf32_9f57d4b208cc67725a80",
        ),
        (
            (),
            0,
            {"prep_stages": 1, "unbounded_softplus": True, "value_rows": 64},
            "cake_kda_tf32_656dad29691bf799ad49",
        ),
        ((), 0, {"write_checkpoints": False}, "cake_kda_tf32_637eee609dc3847cb602"),
        ((), 0, {}, "cake_kda_tf32_9b03dea068c354118714"),
        (
            (),
            0,
            {"value_rows": 64, "write_checkpoints": False},
            "cake_kda_tf32_8bd403eee9a4578a3f2a",
        ),
        ((), 0, {"value_rows": 64}, "cake_kda_tf32_a6e1e706ef047611b36d"),
        ((), 0, {"unbounded_softplus": True}, "cake_kda_tf32_3a4f2589ba67a3e6401c"),
        (
            (),
            0,
            {"unbounded_softplus": True, "value_rows": 64},
            "cake_kda_tf32_a755dd9542165c5ee3c9",
        ),
        (
            (),
            0,
            {"compact_state": True, "prep_stages": 1, "write_checkpoints": False},
            "cake_kda_tf32_e8117b26c217f6a939a8",
        ),
        (
            (),
            0,
            {"checkpoint_tma": True, "compact_state": True, "prep_stages": 1},
            "cake_kda_tf32_e6ba645edc4387b89652",
        ),
        (
            (),
            0,
            {
                "checkpoint_tma": True,
                "compact_state": True,
                "prep_stages": 1,
                "unbounded_softplus": True,
            },
            "cake_kda_tf32_3e3a1d311719d83b943a",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 1,
                "affine_main_indexed_initial": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_0fd45fbaac397bbacead",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 1,
                "affine_main_indexed_initial": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
            },
            "cake_kda_tf32_4374d24862aa57344e4f",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 1,
                "affine_main_indexed_initial": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "unbounded_softplus": True,
            },
            "cake_kda_tf32_baf76efae0b1113677ea",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 2,
                "pdl_wait_initial_state_f32": True,
                "round_tf32_operands": True,
            },
            "cake_kda_tf32_380f74127da276b25cf1",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 2,
                "pdl_wait_initial_state_f32": True,
                "round_tf32_operands": True,
                "unbounded_softplus": True,
            },
            "cake_kda_tf32_380f74127da276b25cf1",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 2,
                "affine_map_only": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_cc9e6f2d5ed58f1f3084",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 2,
                "affine_map_only": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "unbounded_softplus": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_cc9e6f2d5ed58f1f3084",
        ),
        (
            (),
            0,
            {
                "affine_factor_cache": 2,
                "affine_map_only": True,
                "affine_map_output": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_ff7bf41421e07fc5a457",
        ),
        ((), 1, {"value_rows": 64}, "cake_kda_tf32_fb720910739b9a8d69d1"),
        ((), 1, {"unbounded_softplus": True}, "cake_kda_tf32_29e446f2642575503eea"),
        (
            (),
            0,
            {
                "active_beta_f32": True,
                "affine_factor_cache": 1,
                "affine_main_indexed_initial": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_9a6496ea94cd4302f8cd",
        ),
        (
            (),
            0,
            {
                "active_beta_f32": True,
                "affine_factor_cache": 1,
                "affine_main_indexed_initial": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
            },
            "cake_kda_tf32_c48c48b5db0e008fd946",
        ),
        (
            (),
            0,
            {
                "active_beta_f32": True,
                "affine_factor_cache": 2,
                "pdl_wait_initial_state_f32": True,
                "round_tf32_operands": True,
            },
            "cake_kda_tf32_380f74127da276b25cf1",
        ),
        (
            (),
            0,
            {
                "active_beta_f32": True,
                "affine_factor_cache": 2,
                "affine_map_only": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_cc9e6f2d5ed58f1f3084",
        ),
        (
            (),
            0,
            {
                "active_beta_f32": True,
                "affine_factor_cache": 2,
                "affine_map_only": True,
                "affine_map_output": True,
                "pdl_publish_final_state": True,
                "round_tf32_operands": True,
                "write_checkpoints": False,
            },
            "cake_kda_tf32_ff7bf41421e07fc5a457",
        ),
    ],
}


def _factory_table():
    table = {arch: {} for arch in ARCHES}
    for family, family_rows in _FACTORY_ROWS.items():
        family_defaults = _FACTORY_DEFAULTS[family]
        family_signatures = _FACTORY_SIGNATURES[family]
        for args, signature, overrides, modules in family_rows:
            kwargs = {
                name: family_defaults[name] for name in family_signatures[signature]
            }
            kwargs.update(overrides)
            key = (tuple(args), tuple(sorted(kwargs.items())))
            if isinstance(modules, str):
                modules = {arch: modules for arch in MODULES[modules]["arches"]}
            for arch, name in modules.items():
                table[arch].setdefault(family, {})[key] = name
    return table


FACTORIES = _factory_table()


def device_arch(device=None):
    capability = torch.cuda.get_device_capability(device)
    targets = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
    if capability not in targets:
        raise NotImplementedError("TF32 KDA export requires SM100a or SM103a")
    return targets[capability]


def _key(args, kwargs):
    return (tuple(args), tuple(sorted(kwargs.items())))


def _source_path(relative):
    installed = jit_env.FLASHINFER_CSRC_DIR / Path(relative).relative_to("csrc")
    if installed.is_file():
        return installed
    return Path(__file__).resolve().parents[2] / relative


@cache
def spec(name, arch):
    record = MODULES[name]
    if arch not in record["arches"]:
        raise NotImplementedError(f"KDA module {name} is not exported for {arch}")
    include = jit_env.FLASHINFER_INCLUDE_DIR
    if not include.is_dir():
        include = Path(__file__).resolve().parents[2] / "include"
    sources = [_source_path(path) for path in record["sources"]]
    return gen_jit_spec(
        name=f"{record['cache_name']}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], *record["compile_flags"]],
        extra_include_paths=[
            sources[-1].parent,
            _source_path(_SHIM_HEADER).parent,
            _source_path("csrc/tvm_ffi_utils.h").parent,
            include,
        ],
    )


@cache
def load(name, arch):
    return spec(name, arch).build_and_load()


class PreparedKernel:
    """A native module plus caller-retained pointer-TMA descriptor storage."""

    def __init__(self, name, arch):
        record = MODULES[name]
        self.name = name
        self.arch = arch
        self.schedule = None
        native = load(name, arch)
        entry = record["ffi_entry"]
        self._call = getattr(
            native, entry + "_prepared" if record["tma_workspace_bytes"] else entry
        )
        self._prepare = (
            getattr(native, entry + "_prepare_tma")
            if record["tma_workspace_bytes"]
            else None
        )
        self._ready = self._prepare is None
        self._arg_plan = record["arg_plan"]
        self.descriptor_storage = torch.empty(
            record["tma_workspace_bytes"], dtype=torch.uint8, device="cuda"
        )

    def _arguments(self, grid, bindings):
        # One list build per call: binding names resolve straight from the
        # caller's dict, grid/workspace slots are patched afterwards.
        packers = self._packers
        if packers is None:
            packers = self._packers = self._build_packers()
        names, grid_slots, workspace_slots = packers
        args = [bindings[name] if name is not None else None for name in names]
        if grid_slots:
            grid = tuple(grid) + (1,) * (3 - len(grid))
            for slot, axis in grid_slots:
                args[slot] = grid[axis]
        for slot in workspace_slots:
            args[slot] = self.descriptor_storage
        return args

    _packers = None

    def _build_packers(self):
        names = []
        grid_slots = []
        workspace_slots = []
        for slot, (kind, name) in enumerate(self._arg_plan):
            if kind == "grid":
                names.append(None)
                grid_slots.append((slot, ("grid_x", "grid_y", "grid_z").index(name)))
            elif kind == "workspace":
                names.append(None)
                workspace_slots.append(slot)
            else:
                names.append(name)
        return tuple(names), tuple(grid_slots), tuple(workspace_slots)

    def prepare(self, *, grid, **bindings):
        if self._prepare is not None:
            self._prepare(*self._arguments(grid, bindings))
        self._ready = True

    def launch(self, *, grid, **bindings):
        if not self._ready:
            raise RuntimeError("TMA descriptors must be initialized during preparation")
        return self._call(*self._arguments(grid, bindings))


def prepare_descriptors(prepared):
    """Initialize each caller-owned descriptor workspace without running KDA.

    Prepared tensor addresses and metadata are immutable. Tensor contents may
    change between calls. As with the input tensors, callers must order use on
    another stream after preparation on the current stream.
    """
    import tvm_ffi

    owner = getattr(prepared, "_impl", prepared)
    if hasattr(owner, "_main"):
        from ..cake_kda_tf32_runtime import flush_deferred_rebind

        flush_deferred_rebind(owner)
    with tvm_ffi.use_torch_stream():
        if hasattr(owner, "_main"):
            for child in (owner._main, owner._map, owner._correction):
                if child is not None:
                    prepare_descriptors(child)
            if owner._use_output_projection:
                owner._projection_module.prepare(
                    grid=owner._projection_grid, **owner._projection_args
                )
            if getattr(owner, "_apply_route", False):
                # Kernel round 2: pair-map producer, prefix chain and fused apply.
                owner._pairmap_module.prepare(
                    grid=owner._pairmap_grid, **owner._pairmap_bindings()
                )
                owner._prefix_module.prepare(
                    grid=owner._prefix_grid, **owner._prefix_bindings()
                )
                owner._apply_module.prepare(
                    grid=owner._apply_grid, **owner._apply_bindings()
                )
        else:
            if owner.prepare_module is not None:
                owner.prepare_module.prepare(
                    grid=owner.prepare_grid, **owner.prepare_args
                )
            owner.module.prepare(grid=owner.grid, **owner.args)
            owner._descriptors_stale = False


def _factory(name, *args, **kwargs):
    arch = device_arch()
    variants = FACTORIES.get(arch, {}).get(name, {})
    key = _key(args, kwargs)
    if key not in variants:
        raise NotImplementedError(
            f"Unexported KDA schedule specialization: {(arch, name, *key)}"
        )
    return PreparedKernel(variants[key], arch)
