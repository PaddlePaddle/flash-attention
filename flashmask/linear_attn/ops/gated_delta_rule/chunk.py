# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import paddle

from linear_attn.modules.l2norm import l2norm_bwd, l2norm_fwd
from linear_attn.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from linear_attn.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from linear_attn.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from linear_attn.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_chunk_cumsum
from linear_attn.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from linear_attn.ops.utils import chunk_local_cumsum
from linear_attn.ops.utils.constant import RCP_LN2
from linear_attn.ops.utils.index import prepare_chunk_indices
from linear_attn.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from linear_attn.triton_utils import activate_paddle_driver, compat_kernel_wrapper_fastpath


def _use_saved_intermediates_no_recompute(
    output_final_state: bool,
    cu_seqlens: paddle.Tensor | None,
    cp_context,
) -> bool:
    return not output_final_state and cu_seqlens is None and cp_context is None


def chunk_gated_delta_rule_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float,
    initial_state: paddle.Tensor,
    output_final_state: bool,
    cu_seqlens: paddle.Tensor | None = None,
    cp_context=None,
    chunk_indices: paddle.Tensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: paddle.Tensor | None = None,
    dt_bias: paddle.Tensor | None = None,
    return_intermediates: bool = False,
):
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=64,
            scale=RCP_LN2 if use_exp2 else None,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        g = chunk_local_cumsum(
            g,
            chunk_size=64,
            scale=RCP_LN2 if use_exp2 else None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    # CP (Context Parallel) is skipped in Phase 1

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    if return_intermediates:
        return g, o, A, final_state, initial_state, g_input, w, u, h, v_new
    return g, o, A, final_state, initial_state, g_input


def chunk_gated_delta_rule_bwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    A: paddle.Tensor,
    scale: float,
    initial_state: paddle.Tensor,
    do: paddle.Tensor,
    dht: paddle.Tensor,
    cu_seqlens: paddle.Tensor | None = None,
    cp_context=None,
    chunk_indices: paddle.Tensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    g_input: paddle.Tensor | None = None,
    A_log: paddle.Tensor | None = None,
    dt_bias: paddle.Tensor | None = None,
    saved_w: paddle.Tensor | None = None,
    saved_u: paddle.Tensor | None = None,
    saved_h: paddle.Tensor | None = None,
    saved_v_new: paddle.Tensor | None = None,
):
    if all(t is not None for t in (saved_w, saved_u, saved_h, saved_v_new)):
        w, u, h, v_new = saved_w, saved_u, saved_h, saved_v_new
    else:
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )

        # CP (Context Parallel) is skipped in Phase 1

        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    # CP (Context Parallel) is skipped in Phase 1

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    dA_log, ddt_bias = None, None
    if use_gate_in_kernel:
        dg, dA_log, ddt_bias = gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return dq, dk, dv, db, dg, dh0, dA_log, ddt_bias


class ChunkGatedDeltaRuleFunction(paddle.autograd.PyLayer):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
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
        use_qk_l2norm_in_kernel: bool = False,
        cp_context=None,
        transpose_state_layout: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: paddle.Tensor | None = None,
        dt_bias: paddle.Tensor | None = None,
    ):
        # Save original input refs before any reassignment, for _tensor_mask/_needs_grad below.
        _orig_forward_args = [
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, cp_context,
            transpose_state_layout, use_gate_in_kernel, A_log, dt_bias,
        ]
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = prepare_chunk_indices(
            cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        use_saved_intermediates = _use_saved_intermediates_no_recompute(
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
        with activate_paddle_driver(), compat_kernel_wrapper_fastpath():
            gdn_outputs = chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=transpose_state_layout,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                return_intermediates=use_saved_intermediates,
            )
        if use_saved_intermediates:
            g, o, A, final_state, initial_state, g_input, w, u, h, v_new = gdn_outputs
        else:
            g, o, A, final_state, initial_state, g_input = gdn_outputs
            w = u = h = v_new = None
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A,
            initial_state, cu_seqlens, chunk_indices,
            g_input, A_log, dt_bias,
            w, u, h, v_new,
        )
        # Store non-tensor params as ctx attributes
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.output_final_state = output_final_state
        ctx.use_saved_intermediates = use_saved_intermediates
        # Paddle PyLayer backward must return exactly as many values as tensor inputs.
        # Record which forward args are tensors so backward can filter its return.
        # Also record which tensor inputs need gradients (stop_gradient=False),
        # because Paddle requires backward to return None for stop_gradient=True tensors.
        # IMPORTANT: Use _orig_forward_args (captured before q/k/g/initial_state were
        # reassigned by l2norm_fwd / chunk_gated_delta_rule_fwd) so that stop_gradient
        # reflects the *caller's* tensors, not the internal intermediates.
        ctx._tensor_mask = tuple(isinstance(a, paddle.Tensor) for a in _orig_forward_args)
        ctx._needs_grad = tuple(
            isinstance(a, paddle.Tensor) and not a.stop_gradient for a in _orig_forward_args
        )
        # Paddle PyLayer forward cannot return None, use dummy tensor as placeholder
        if final_state is None:
            final_state = paddle.zeros([1], dtype=q.dtype)
        return o.cast(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        # When output_final_state=False, forward returned a dummy tensor;
        # restore dht to None so downstream bwd functions handle it correctly
        if not ctx.output_final_state:
            dht = None
        (q, q_rstd, k, k_rstd, v, g, beta, A,
         initial_state, cu_seqlens, chunk_indices,
         g_input, A_log, dt_bias,
         w, u, h, v_new) = ctx.saved_tensor()
        with activate_paddle_driver(), compat_kernel_wrapper_fastpath():
            dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens,
                cp_context=ctx.cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=ctx.transpose_state_layout,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                g_input=g_input,
                A_log=A_log,
                dt_bias=dt_bias,
                saved_w=w if ctx.use_saved_intermediates else None,
                saved_u=u if ctx.use_saved_intermediates else None,
                saved_h=h if ctx.use_saved_intermediates else None,
                saved_v_new=v_new if ctx.use_saved_intermediates else None,
            )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # Build all grads in forward arg order, filter to tensor inputs only.
        # Order: q, k, v, g, beta, scale, initial_state, output_final_state,
        #        cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, cp_context,
        #        transpose_state_layout, use_gate_in_kernel, A_log, dt_bias
        all_grads = [
            dq.cast(q.dtype), dk.cast(k.dtype), dv.cast(v.dtype), dg.cast(g.dtype), db.cast(beta.dtype),
            None, dh0, None, None, None, None, None, None, None, dA_log, ddt_bias,
        ]
        return tuple(
            g if needs_grad else None
            for g, is_tensor, needs_grad in zip(all_grads, ctx._tensor_mask, ctx._needs_grad)
            if is_tensor
        )


def chunk_gated_delta_rule(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float = None,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: paddle.Tensor | None = None,
    cu_seqlens_cpu: paddle.Tensor | None = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    **kwargs,
):
    r"""
    Args:
        q (paddle.Tensor):
            queries of shape `[B, T, H, K]`.
        k (paddle.Tensor):
            keys of shape `[B, T, H, K]`.
        v (paddle.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g (paddle.Tensor):
            (forget) gating tensor of shape `[B, T, HV]`.
            When `use_gate_in_kernel=False` (default), `g` should be in log space (pre-computed decay).
            When `use_gate_in_kernel=True`, `g` is the raw input before gate activation;
            the kernel fuses `-exp(A_log) * softplus(g + dt_bias)` + chunk cumsum internally.
        beta (paddle.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[paddle.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (paddle.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cp_context:
            Context parallel context (skipped in Phase 1). Default: `None`.
        transpose_state_layout (Optional[bool]):
            Whether to use the transposed state layout for the hidden state.
            Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space GDN decay internally.
            When `True`, the passed `g` is the raw input, and `A_log` must be provided.
            The kernel fuses gate activation + chunk cumsum in a single pass.
            Default: `False`.
        A_log (Optional[paddle.Tensor]):
            Decay parameter of shape `[HV]`. Required when `use_gate_in_kernel=True`.
        dt_bias (Optional[paddle.Tensor]):
            Bias added to `g` before activation, of shape `[HV]`.
            Only used when `use_gate_in_kernel=True`.

    Returns:
        o (paddle.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (paddle.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
    # Validate head dimensions
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by "
            f"num_heads (H={H}), but got HV % H = {HV % H}"
        )

    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    # CP (Context Parallel) is skipped in Phase 1
    # cp_context is accepted but ignored

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    use_gate_in_kernel = kwargs.get('use_gate_in_kernel', False)
    A_log = kwargs.get('A_log')
    dt_bias = kwargs.get('dt_bias')
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
    )
    # Convert dummy tensor back to None when output_final_state=False
    if not output_final_state:
        final_state = None
    return o, final_state


chunk_gdn = chunk_gated_delta_rule
