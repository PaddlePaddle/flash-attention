# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import paddle
import triton
import triton.language as tl

from linear_attn.ops.utils.op import exp, exp2
from linear_attn.utils import input_guard
from linear_attn.triton_utils import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_EXP2:
                b_h *= exp2(b_g)
            else:
                b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gk[None, :])
                else:
                    b_h *= exp2(b_gk[:, None])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gk[None, :])
                else:
                    b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gv[:, None])
                else:
                    b_h *= exp2(b_gv[None, :])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gv[:, None])
                else:
                    b_h *= exp(b_gv[None, :])

        if TRANSPOSE_STATE:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor | None = None,
    gk: paddle.Tensor | None = None,
    gv: paddle.Tensor | None = None,
    beta: paddle.Tensor | None = None,
    scale: float = None,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: paddle.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = paddle.empty_like(v)
    if output_final_state:
        if transpose_state_layout:
            final_state = paddle.empty([N, HV, V, K], dtype=paddle.float32)
        else:
            final_state = paddle.empty([N, HV, K, V], dtype=paddle.float32)
    else:
        final_state = None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentFunction(paddle.autograd.PyLayer):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        g: paddle.Tensor | None = None,
        gk: paddle.Tensor | None = None,
        gv: paddle.Tensor | None = None,
        beta: paddle.Tensor | None = None,
        scale: float = None,
        initial_state: paddle.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: paddle.Tensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ):
        # Store non-tensor params as ctx attributes
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_exp2 = use_exp2
        ctx.transpose_state_layout = transpose_state_layout

        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )

        # Paddle PyLayer forward cannot return None, use dummy tensor as placeholder
        if final_state is None:
            final_state = paddle.zeros([1], dtype=q.dtype)
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps.",
        )


def fused_recurrent_gated_delta_rule(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor | None = None,
    gk: paddle.Tensor | None = None,
    gv: paddle.Tensor | None = None,
    beta: paddle.Tensor | None = None,
    scale: float = None,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: paddle.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
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
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (paddle.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (paddle.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
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
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (paddle.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        transpose_state_layout (bool):
            Whether to use transposed state layout `[V, K]` instead of `[K, V]`. Default: `False`.

    Returns:
        o (paddle.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (paddle.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
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
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = paddle.ones_like(q[..., 0])

    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        use_exp2,
        transpose_state_layout,
    )
    # Convert dummy tensor back to None when output_final_state=False
    if not output_final_state:
        final_state = None
    return o, final_state


fused_recurrent_gdn = fused_recurrent_gated_delta_rule
