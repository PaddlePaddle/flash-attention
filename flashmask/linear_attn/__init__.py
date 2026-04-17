# -*- coding: utf-8 -*-
# flash-linear-attention Paddle migration entry point
#
# Migrated from fla-org/flash-linear-attention (MIT License) for PaddlePaddle.
# Provides Triton-based GDN (Gated Delta Networks) and KDA (Kimi Delta
# Attention) operators with chunk-wise and fused-recurrent execution modes.
#
# Known limitations (Phase 1):
#   - Context Parallel (CP) is NOT supported.  The cp_context parameter is
#     accepted for API compatibility but has no effect; passing a non-None
#     value may raise NotImplementedError in backward paths.
#   - fused_recurrent_gdn / fused_recurrent_kda are FORWARD-ONLY.  Calling
#     backward through them will raise NotImplementedError.  Use the
#     chunk-based variants (chunk_gdn / chunk_kda) for training workloads
#     that require gradient computation.

import paddle
from linear_attn.triton_utils import _is_package_installed

# No torch environment: enable triton scope compat globally (zero runtime overhead)
if not _is_package_installed("torch"):
    paddle.enable_compat(scope={"triton"})

from linear_attn.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gdn,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gdn,
)
from linear_attn.ops.kda import (
    chunk_kda,
    fused_recurrent_kda,
)

__all__ = [
    'chunk_gated_delta_rule',
    'chunk_gdn',
    'fused_recurrent_gated_delta_rule',
    'fused_recurrent_gdn',
    'chunk_kda',
    'fused_recurrent_kda',
]
