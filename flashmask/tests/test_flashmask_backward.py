# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import pytest
import inspect

from flash_mask.cute import flashmask_utils as fm
from flash_mask.cute.interface import _flash_attn_bwd, flashmask_attention


from tests.generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_none_mask,
    generate_non_causal_mask,
    generate_causal_mask,
    generate_sliding_window_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_share_question_mask,
    generate_global_sliding_window_mask,
    generate_causal_blockwise_mask,
    generate_prefix_lm_document_mask,
    generate_prefix_lm_causal_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask
)

from tests.test_flashmask_util import attention_ref


# bfloat16-64-s12-hqkv0-1-document_mask

@pytest.mark.parametrize("mode", ['causal_mask', 'sliding_window_mask', 'causal_document_mask', 'document_mask', 'share_question_mask', 'causal_blockwise_mask', 'prefix_lm_document_mask', 'prefix_lm_causal_mask', 'qk_sparse_mask', 'random_eviction_mask'])
#@pytest.mark.parametrize("mode", ['causal_mask', 'sliding_window_mask', 'causal_document_mask', 'document_mask', 'share_question_mask'])
#@pytest.mark.parametrize("mode", ['causal_blockwise_mask'])
#@pytest.mark.parametrize("mode", ['document_mask'])
#@pytest.mark.parametrize("mode", ['random_eviction_mask'])
@pytest.mark.parametrize("b", [1, 2, 4])
#@pytest.mark.parametrize("b", [1, 2])
#@pytest.mark.parametrize("b", [1])
#@pytest.mark.parametrize("hqkv", [(1, 1), (4, 1), (4, 4), (32, 4)])
@pytest.mark.parametrize("hqkv", [(1, 1), (4, 1), (4, 4)])
#@pytest.mark.parametrize("hqkv", [(1, 1), (4, 1)])
#@pytest.mark.parametrize("hqkv", [(1, 1)])
#@pytest.mark.parametrize("hqkv", [(1, 1)])
@pytest.mark.parametrize("s", [(31, 31), (32, 32), (61, 61), (64, 64), (96, 96), (128, 128), (128, 256), (128, 799), (132, 132), (132, 256), (256, 128),  (256, 132), (256, 256), (512, 512), (799, 128), (1024, 1024), (4096, 4096), (8191, 8192), (8192, 8191)])
#@pytest.mark.parametrize("s", [4096])
#@pytest.mark.parametrize("s", [512])
#@pytest.mark.parametrize("s", [132])
#@pytest.mark.parametrize("s", [(128, 128), (128, 256), (132, 132), (132, 256), (256, 128),  (256, 132), (256, 256)])
#@pytest.mark.parametrize("s", [(8191, 8192)])
#@pytest.mark.parametrize("s", [(256, 132)])
#@pytest.mark.parametrize("s", [(1024, 1024)])
#@pytest.mark.parametrize("d", [128])
#@pytest.mark.parametrize("d", [64, 80, 128])
@pytest.mark.parametrize("d", [64, 128])
#@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("dtype", ['bfloat16'])
def test_flashmask_backward(mode, b, hqkv, s, d, dtype):

    h, h_kv = hqkv
    sq, sk = s
    paddle.seed(0)
    if mode == 'causal_mask':
        startend_row_indices, causal = generate_causal_mask(b, sq, sk, h_kv)
    elif mode == 'sliding_window_mask':
        startend_row_indices, causal = generate_sliding_window_mask(b, sq, sk, h_kv)
    elif mode == 'causal_document_mask':
        startend_row_indices, causal = generate_causal_document_mask(b, sq, sk, h_kv)
    elif mode == 'document_mask':
        startend_row_indices, causal = generate_document_mask(b, sq, sk, h_kv)
    elif mode == 'share_question_mask':
        startend_row_indices, causal = generate_share_question_mask(b, sq, sk, h_kv)
    #elif mode == 'global_sliding_window_mask':
    #    startend_row_indices, causal = generate_global_sliding_window_mask(b, sq, sk, h_kv)
    elif mode == 'causal_blockwise_mask':
        startend_row_indices, causal = generate_causal_blockwise_mask(b, sq, sk, h_kv)
    elif mode == 'prefix_lm_document_mask':
        startend_row_indices, causal = generate_prefix_lm_document_mask(b, sq, sk, h_kv)
    elif mode == 'prefix_lm_causal_mask':
        startend_row_indices, causal = generate_prefix_lm_causal_mask(b, sq, sk, h_kv)
    elif mode == 'qk_sparse_mask':
        startend_row_indices, causal = generate_qk_sparse_mask(b, sq, sk, h_kv)
    elif mode == 'random_eviction_mask':
        startend_row_indices, causal = generate_random_eviction_mask(b, sq, sk, h_kv)

    query = paddle.randn([b, sq, h, d], dtype=dtype)
    key = paddle.randn([b, sk, h_kv, d], dtype=dtype)
    value = paddle.randn([b, sk, h_kv, d], dtype=dtype)
    output_grad = paddle.randn([b, sq, h, d], dtype=dtype)

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    q_ref, k_ref, v_ref = [x.detach().clone() for x in (query, key, value)]
    q_ref.stop_gradient = False
    k_ref.stop_gradient = False
    v_ref.stop_gradient = False
   
    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, sq, h_kv, dtype, causal)
    out_ref, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias
    )
    out_ref.backward(output_grad)

    out_bf16, _ = attention_ref(
        query,
        key,
        value,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True
    )
    out_bf16.backward(output_grad)

    with paddle.no_grad():
        output, log_sum_exp = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True,
            training=True,
        )

    fm_query_grad, fm_key_grad, fm_value_grad = query.grad, key.grad, value.grad

    flashmask_info = fm.FlashMaskInfoPaddle(
        startend_row_indices=startend_row_indices,
        is_causal=causal,
    )
    fm4_query_grad, fm4_key_grad, fm4_value_grad = _flash_attn_bwd(
        query,
        key,
        value,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info,
        causal=causal,
        #deterministic=True
    )


    softcap = 0.0
    rtol = 2 if softcap == 0.0 else 3
    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (fm4_query_grad - q_ref.grad).abs().max().item() <= rtol * (fm_query_grad - q_ref.grad).abs().max().item() + dq_atol
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (fm4_key_grad - k_ref.grad).abs().max().item() <= rtol * (fm_key_grad - k_ref.grad).abs().max().item() + dk_atol
    dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (fm4_value_grad - v_ref.grad).abs().max().item() <= rtol * (fm_value_grad - v_ref.grad).abs().max().item() + dv_atol
