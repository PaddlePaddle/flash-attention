# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import paddle
import paddle.nn.functional as F
import triton
import triton.language as tl

from linear_attn.utils import autotune_cache_kwargs, tensor_cache
from linear_attn.triton_utils import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['B'],
    **autotune_cache_kwargs,
)
@triton.jit
def prepare_position_ids_kernel(
    y,
    cu_seqlens,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(cu_seqlens: paddle.Tensor) -> paddle.Tensor:
    return paddle.diff(cu_seqlens)


@tensor_cache
def prepare_lens_from_mask(mask: paddle.Tensor) -> paddle.Tensor:
    return mask.sum(axis=-1).cast(paddle.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: paddle.Tensor,
    dtype=paddle.int32,
) -> paddle.Tensor:
    return F.pad(lens.cumsum(axis=0).cast(dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: paddle.Tensor,
    dtype=paddle.int32,
) -> paddle.Tensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int,
    seq_len: int,
    split_size: int,
    cu_seqlens: paddle.Tensor | None = None,
    dtype=paddle.int32,
) -> paddle.Tensor:
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return paddle.to_tensor(
        [
            i
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
            for i in range(bos, eos, split_size)
        ] + [cu_seqlens[-1]],
        dtype=dtype,
    )


@tensor_cache
def prepare_position_ids(cu_seqlens: paddle.Tensor, cu_seqlens_cpu: paddle.Tensor | None = None) -> paddle.Tensor:
    if cu_seqlens_cpu is not None:
        return paddle.concat([
            paddle.arange(n, dtype=cu_seqlens.dtype)
            for n in prepare_lens(cu_seqlens_cpu).unbind()
        ])
    return paddle.concat([
        paddle.arange(n, dtype=cu_seqlens.dtype)
        for n in prepare_lens(cu_seqlens).unbind()
    ])


@tensor_cache
def prepare_sequence_ids(cu_seqlens: paddle.Tensor, cu_seqlens_cpu: paddle.Tensor | None = None) -> paddle.Tensor:
    return (prepare_position_ids(cu_seqlens, cu_seqlens_cpu) == 0).cast(paddle.int64).cumsum(axis=0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: paddle.Tensor, cu_seqlens_cpu: paddle.Tensor | None = None) -> paddle.Tensor:
    position_ids = prepare_position_ids(cu_seqlens, cu_seqlens_cpu)
    return paddle.stack([prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu), position_ids], 1).cast(cu_seqlens.dtype)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: paddle.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: paddle.Tensor | None = None,
) -> paddle.Tensor:
    if cu_seqlens_cpu is not None:
        indices = paddle.concat([paddle.arange(n)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return paddle.stack([(indices == 0).cast(paddle.int64).cumsum(axis=0) - 1, indices], 1).cast(cu_seqlens.dtype)
    indices = paddle.concat([paddle.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return paddle.stack([(indices == 0).cast(paddle.int64).cumsum(axis=0) - 1, indices], 1).cast(cu_seqlens.dtype)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: paddle.Tensor,
    chunk_size: int,
) -> paddle.Tensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(axis=-1)


@tensor_cache
def get_max_num_splits(
    cu_seqlens: paddle.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: paddle.Tensor | None = None
) -> int:
    if cu_seqlens_cpu is not None:
        return triton.cdiv(int(max(prepare_lens(cu_seqlens_cpu))), chunk_size)
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
