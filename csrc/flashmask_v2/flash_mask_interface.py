import torch
import flash_mask_2._C as C  # PyTorch FlashMask


flash_mask_2_cuda = torch.ops.flash_mask


def flashmask_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    startend_row_indices: torch.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: torch.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: torch.Tensor | None = None,
):

    if window_size is not None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        sq = query.shape[1]
        bsz = query.shape[0]
        assert startend_row_indices is None, ("can't use window_size with startend_row_indices.")
        if causal:
            startend_row_indices = torch.arange(window_size[0] + 1, sq + window_size[0] + 1, dtype=torch.int32, device=query.device).reshape(1, 1, sq, 1)
            startend_row_indices = torch.clip(startend_row_indices, max=sq).repeat_interleave(bsz, dim=0)
        else:
            startend_row_indices = torch.empty((1, 1, sq, 2), dtype=torch.int32, device=query.device)
            startend_row_indices[0, 0, :, 0] = torch.arange(window_size[0] + 1, sq + window_size[0] + 1, dtype=torch.int32, device=query.device)
            startend_row_indices[0, 0, :, 1] = torch.arange(-window_size[1], sq - window_size[1], dtype=torch.int32, device=query.device)
            startend_row_indices = torch.clip(startend_row_indices, min=0, max=sq).repeat_interleave(bsz, dim=0)

    if block_mask is not None:
        assert startend_row_indices is not None, "must provide startend_row_indices when using block_mask_attn."

    assert startend_row_indices is not None, "Please use flash-attn rather than flash-mask for startend_row_indices not provided."

    assert startend_row_indices.dtype == torch.int32, f"startend_row_indices.dtype must be torch.int32, but got {startend_row_indices.dtype}"
    assert len(startend_row_indices.shape) == 4, f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"
    assert startend_row_indices.shape[0] == key.shape[0], f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
    assert startend_row_indices.shape[2] == key.shape[1], f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[2]}"
    assert startend_row_indices.shape[1] in [1, key.shape[2]], ("startend_row_indices head_num must be equal to 1(broadcast) or head_num_k.")

    if block_mask is not None:
        assert block_mask.dtype == torch.int32, f"block_mask.dtype must be torch.int32, but got {block_mask.dtype}"
        assert block_mask.shape[0] == key.shape[0], f"block_mask.shape[0] must be equal to batch_size, but got {block_mask.shape[0]} and {key.shape[0]}"
        assert block_mask.shape[1] == startend_row_indices.shape[1], f"block_mask.shape[1] must be equal to startend_row_indices.shape[1], but got {block_mask.shape[1]} and {key.shape[2]}"
        assert block_mask.shape[2] == (query.shape[1] + 127) // 128, "block_size must be 128 when using block_mask_attn"
        assert block_mask.shape[3] == (key.shape[1] + 127) // 128, "block_size must be 128 when using block_mask_attn"
        assert key.shape[3] == 128, "headdim must be 128 when using block_mask_attn"

    if causal:
        if startend_row_indices.shape[-1] == 1:
            has_end = False
        elif startend_row_indices.shape[-1] == 2:
            has_end = True
        else:
            raise ValueError(f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}")
    else:
        if startend_row_indices.shape[-1] == 2:
            has_end = False
        elif startend_row_indices.shape[-1] == 4:
            has_end = True
        else:
            raise ValueError(f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}")   

    assert dropout == 0.0, "flashmask_attention_v2 does not support dropout"
    assert not return_seed_offset, "flashmask_attention_v2 does not support return seed_offset"
    assert fixed_seed_offset is None, "flashmask_attention_v2 does not support setting seed_offset"
    assert rng_name == "", "flashmask_attention_v2 does not support setting rng_name"
    assert training, "flashmask_attention_v2 does not support setting training to False"
    assert name is None, "flashmask_attention_v2 does not support setting name"

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** (-0.5)

    out, softmax_lse, out_accum, softmax_lse_accum = torch.ops.flash_mask.fwd(q=query, k=key, v=value, startend_row_indices=startend_row_indices, block_mask=block_mask, softmax_scale=softmax_scale, is_causal=causal)
    torch.cuda.synchronize()
    outputs = [out]
    # print(out)
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
    


# import torch
# import flash_mask_2._C as C  # PyTorch FlashMask

torch.manual_seed(2023)
device = 'cuda'
dtype = torch.bfloat16
assert torch.cuda.is_available()

q = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)
k = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)
v = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)

startend_row_indices = torch.tensor([8]*10 + [5]*10, dtype=torch.int32).reshape([1, 2, 10, 1])

output=flashmask_attention(query=q, key=k, value=v, startend_row_indices=startend_row_indices, causal=True)

# print(output)
