# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import paddle

from linear_attn.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from linear_attn.ops.gla.chunk import chunk_gla_fwd_o_gk
from linear_attn.ops.kda.chunk_intra import chunk_kda_fwd_intra
from linear_attn.ops.kda.gate import kda_gate_chunk_cumsum
from linear_attn.ops.utils import chunk_local_cumsum
from linear_attn.ops.utils.constant import RCP_LN2


def chunk_kda_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float,
    initial_state: paddle.Tensor,
    output_final_state: bool,
    cu_seqlens: paddle.Tensor | None = None,
    cu_seqlens_cpu: paddle.Tensor | None = None,
    chunk_indices: paddle.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: paddle.Tensor | None = None,
    dt_bias: paddle.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,
    transpose_state_layout: bool = False,
):
    # Apply gate activation
    g_org = None
    if use_gate_in_kernel:
        g_org = g
        g = kda_gate_chunk_cumsum(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices
        )

    # qg = None if disable_recompute is False
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute
    )

    # CP (Context Parallel) is skipped in Paddle migration

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    # CP (Context Parallel) is skipped in Paddle migration

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )
    if disable_recompute is False:
        # Delete to save memory
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            # Only delete h if not requested for inference
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
