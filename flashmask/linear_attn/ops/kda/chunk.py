# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Related files are modified and supported by the Moonshot AI Team

import paddle

from linear_attn.modules.l2norm import l2norm_bwd, l2norm_fwd
from linear_attn.ops.kda.chunk_bwd import chunk_kda_bwd
from linear_attn.ops.kda.chunk_fwd import chunk_kda_fwd
from linear_attn.triton_utils import activate_paddle_driver, compat_kernel_wrapper_fastpath
from linear_attn.ops.utils.index import prepare_chunk_indices
from linear_attn.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkKDAFunction(paddle.autograd.PyLayer):
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
        A_log: paddle.Tensor,
        dt_bias: paddle.Tensor,
        scale: float,
        initial_state: paddle.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: paddle.Tensor | None = None,
        cu_seqlens_cpu: paddle.Tensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context=None,
        transpose_state_layout: bool = False,
    ):
        chunk_size = 64
        with activate_paddle_driver(), compat_kernel_wrapper_fastpath():
            _orig_forward_args = [
                q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
                output_final_state, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                cu_seqlens, cu_seqlens_cpu, safe_gate, lower_bound,
                disable_recompute, return_intermediate_states, cp_context, transpose_state_layout,
            ]

            q_rstd, k_rstd = None, None
            if use_qk_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)

            chunk_indices = prepare_chunk_indices(
                cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None

            g_input = g

            (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=g_input,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_indices=chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states,
                cp_context=cp_context,
                transpose_state_layout=transpose_state_layout,
            )

            if return_intermediate_states:
                assert not paddle.is_grad_enabled(), "return_intermediate_states is only allowed in inference mode"
                assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
                return o.cast(q.dtype), final_state, h

            saved_tensors = [
                q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta, A_log, dt_bias, Aqk, Akk,
                initial_state, cu_seqlens, chunk_indices,
            ]
            if disable_recompute:
                saved_tensors.extend([w, u, qg, kg, v_new, h])
            ctx.save_for_backward(*saved_tensors)
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        ctx.output_final_state = output_final_state
        # Paddle PyLayer backward must return exactly as many values as tensor inputs.
        # Record which forward args are tensors so backward can filter its return.
        # Also record which tensor inputs need gradients (stop_gradient=False).
        # IMPORTANT: Use _orig_forward_args (captured before q/k were reassigned by
        # l2norm_fwd) so that stop_gradient reflects the *caller's* tensors.
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
        with activate_paddle_driver(), compat_kernel_wrapper_fastpath():
            saved_tensors = ctx.saved_tensor()
            if ctx.disable_recompute:
                (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta, A_log, dt_bias, Aqk, Akk,
                 initial_state, cu_seqlens, chunk_indices,
                 w, u, qg, kg, v_new, h) = saved_tensors
            else:
                (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta, A_log, dt_bias, Aqk, Akk,
                 initial_state, cu_seqlens, chunk_indices) = saved_tensors
                w = u = qg = kg = v_new = h = None

            dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
                q=q,
                k=k,
                v=v,
                g=g_cumsum,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                g_org=g_input if ctx.use_gate_in_kernel else None, lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                A_log=A_log, dt_bias=dt_bias,
                disable_recompute=ctx.disable_recompute,
                w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h,
                cp_context=ctx.cp_context,
                transpose_state_layout=ctx.transpose_state_layout,
            )
            if ctx.use_qk_l2norm_in_kernel:
                dq = l2norm_bwd(q, q_rstd, dq)
                dk = l2norm_bwd(k, k_rstd, dk)

        # Build all grads in forward arg order, filter to tensor inputs only.
        # Order: q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
        #        output_final_state, use_qk_l2norm_in_kernel, use_gate_in_kernel,
        #        cu_seqlens, cu_seqlens_cpu, safe_gate, lower_bound,
        #        disable_recompute, return_intermediate_states, cp_context, transpose_state_layout
        all_grads = [
            dq.cast(q.dtype), dk.cast(k.dtype), dv.cast(v.dtype), dg.cast(g_input.dtype), db.cast(beta.dtype),
            dA, dbias, None, dh0, None, None, None, None, None, None, None, None, None, None, None,
        ]
        return tuple(
            g if needs_grad else None
            for g, is_tensor, needs_grad in zip(all_grads, ctx._tensor_mask, ctx._needs_grad)
            if is_tensor
        )


def chunk_kda(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float | None = None,
    initial_state: paddle.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: paddle.Tensor | None = None,
    cu_seqlens_cpu: paddle.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
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
            values of shape `[B, T, H, V]`.
        g (paddle.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (paddle.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[paddle.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If `True`:
              The passed `g` acts as the raw input for `-exp(A_log).view(H, -1) * softplus(g + dt_bias.view(H, K))`.
              Note that as part of the input arguments,
              `A_log` (shape `[H]`) and the optional `dt_bias` (shape `[H * K]`) should be provided.
            - If `False`, `g` is expected to be the pre-computed decay value.
            Default: `False`.
        cu_seqlens (paddle.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (paddle.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        safe_gate (bool):
            Whether the kernel can assume the gate values (in log space) are in a safe range
            and use M=16 TensorCore acceleration for higher throughput.
            The safe range is ``[lower_bound, 0)``. With the default ``lower_bound=-5``,
            the per-step decay factor ``exp(g)`` is bounded in ``[exp(-5), 1) = [0.0067, 1)``,
            meaning each step retains at least ~0.67% of the state -- a negligible loss that
            has minimal impact on model quality while enabling significant speedup.
            Requires ``lower_bound`` to be set. Default: ``False``.
        lower_bound (Optional[float]):
            Lower bound for the forget gate (in log space) when ``use_gate_in_kernel=True``.
            Changes the gate activation from ``-exp(A_log) * softplus(g + dt_bias)``
            to ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
            which naturally clamps the output to ``[lower_bound, 0)``.
            Recommended value: ``-5`` (i.e., ``exp(-5) = 0.0067``). Default: ``None``.
        disable_recompute (bool):
            Whether to disable gradient recomputation in the kernel. When `True`, the kernel
            will save all intermediate activations for backward pass, which is beneficial
            for training small models at the cost of increased memory usage. Default: `False`.
        return_intermediate_states (bool):
            If True, returns intermediate state `h` for inference scenarios (e.g., vLLM).
            Must be used outside `paddle.is_grad_enabled()` and will return a 3-tuple instead of 2-tuple.
            This is not intended for training as it bypasses autograd. Default: `False`.
        cp_context:
            Context parallel context (skipped in Paddle migration). Default: `None`.
        transpose_state_layout (Optional[bool]):
            Whether to use the transposed state layout for the hidden state.
            Default: `False`.

    Returns:
        - Normal mode (return_intermediate_states=False): A tuple (o, final_state)
            o (paddle.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (paddle.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
        - Inference mode (return_intermediate_states=True): A tuple (o, final_state, h)
            o (paddle.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (paddle.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
            h (paddle.Tensor):
                Intermediate states of shape `[B, NT, H, K, V]` and dtype `bfloat16` for caching or further processing.
                - For equal-length sequences: `NT = #chunks_per_sequence` (typically `ceil(T / chunk_size)`)
                - For variable-length sequences (cu_seqlens): B is always 1 (flattened),
                  NT is the total number of chunks across all sequences,
                  determined by `prepare_chunk_indices(cu_seqlens, chunk_size)`
    """

    # CP (Context Parallel) is skipped in Paddle migration

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
    if initial_state is not None:
        assert initial_state.dtype == paddle.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    result = ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        disable_recompute,
        return_intermediate_states,
        cp_context,
        transpose_state_layout,
    )
    if return_intermediate_states:
        o, final_state, h = result
        if not output_final_state:
            final_state = None
        return o, final_state, h
    o, final_state = result
    # Convert dummy tensor back to None when output_final_state=False
    if not output_final_state:
        final_state = None
    return o, final_state
