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

from flash_mask.cute import flashmask_utils as fm

def reference_scan_max_min(tensor, batch, heads, seqlen_k, kBlockN):
    nblocks = ((seqlen_k + kBlockN - 1) // kBlockN + 3 ) & 0xfffffffc
    max_tensor = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
    min_tensor = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
    nblocks = (seqlen_k + kBlockN - 1) // kBlockN
    for b in range(batch):
        for h in range(heads):
            for n in range(nblocks):
                sta = n * kBlockN
                end = min(sta + kBlockN, seqlen_k)
                slice_tensor = tensor[b, h, sta:end]
                max_tensor[b, h, n] = slice_tensor.max()
                min_tensor[b, h, n] = slice_tensor.min()
    return max_tensor, min_tensor

def check_prepare_block_maxmin(flashmask_info, kBlockN: int):
    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape

    if num_vecs == 1:
        LTS_nblock_max, LTS_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 0], batch, heads, seqlen_k, kBlockN)
        np.testing.assert_equal(LTS_nblock_max.numpy(), flashmask_info.LTS_nblock_max.numpy())
        np.testing.assert_equal(LTS_nblock_min.numpy(), flashmask_info.LTS_nblock_min.numpy())
    elif num_vecs == 2 and flashmask_info.is_causal:
        LTS_nblock_max, LTS_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 0], batch, heads, seqlen_k, kBlockN)
        LTE_nblock_max, LTE_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 1], batch, heads, seqlen_k, kBlockN)
        np.testing.assert_equal(LTS_nblock_max.numpy(), flashmask_info.LTS_nblock_max.numpy())
        np.testing.assert_equal(LTS_nblock_min.numpy(), flashmask_info.LTS_nblock_min.numpy())
        np.testing.assert_equal(LTE_nblock_max.numpy(), flashmask_info.LTE_nblock_max.numpy())
        np.testing.assert_equal(LTE_nblock_min.numpy(), flashmask_info.LTE_nblock_min.numpy())

    elif num_vecs == 2 and not flashmask_info.is_causal:
        LTS_nblock_max, LTS_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 0], batch, heads, seqlen_k, kBlockN)
        UTE_nblock_max, UTE_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 1], batch, heads, seqlen_k, kBlockN)
        np.testing.assert_equal(LTS_nblock_max.numpy(), flashmask_info.LTS_nblock_max.numpy())
        np.testing.assert_equal(LTS_nblock_min.numpy(), flashmask_info.LTS_nblock_min.numpy())
        np.testing.assert_equal(UTE_nblock_max.numpy(), flashmask_info.UTE_nblock_max.numpy())
        np.testing.assert_equal(UTE_nblock_min.numpy(), flashmask_info.UTE_nblock_min.numpy())
    elif num_vecs == 4:
        LTS_nblock_max, LTS_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 0], batch, heads, seqlen_k, kBlockN)
        LTE_nblock_max, LTE_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 1], batch, heads, seqlen_k, kBlockN)
        UTS_nblock_max, UTS_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 2], batch, heads, seqlen_k, kBlockN)
        UTE_nblock_max, UTE_nblock_min = reference_scan_max_min(flashmask_info.startend_row_indices[..., 3], batch, heads, seqlen_k, kBlockN)
        np.testing.assert_equal(LTS_nblock_max.numpy(), flashmask_info.LTS_nblock_max.numpy())
        np.testing.assert_equal(LTS_nblock_min.numpy(), flashmask_info.LTS_nblock_min.numpy())
        np.testing.assert_equal(LTE_nblock_max.numpy(), flashmask_info.LTE_nblock_max.numpy())
        np.testing.assert_equal(LTE_nblock_min.numpy(), flashmask_info.LTE_nblock_min.numpy())
        np.testing.assert_equal(UTS_nblock_max.numpy(), flashmask_info.UTS_nblock_max.numpy())
        np.testing.assert_equal(UTS_nblock_min.numpy(), flashmask_info.UTS_nblock_min.numpy())
        np.testing.assert_equal(UTE_nblock_max.numpy(), flashmask_info.UTE_nblock_max.numpy())
        np.testing.assert_equal(UTE_nblock_min.numpy(), flashmask_info.UTE_nblock_min.numpy())
    else:
        raise ValueError(f"Unsupported num_vecs={num_vecs} in flashmask_info")

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("h", [1, 2, 4])
@pytest.mark.parametrize("s", [31, 32, 61, 64, 96, 128, 132, 256, 512, 1024])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("kBlockN", [64, 128, 256])
#@pytest.mark.parametrize("b", [1, 2, 4])
#@pytest.mark.parametrize("h", [1, 2, 4])
#@pytest.mark.parametrize("s", [31])
#@pytest.mark.parametrize("n", [1, 2, 4])
#@pytest.mark.parametrize("is_causal", [False, True])
#@pytest.mark.parametrize("kBlockN", [128])
def test_prepare_block_maxmin(b, h, s, n, is_causal, kBlockN):
    paddle.seed(0)
    inp = paddle.randint(low=0, high=s, shape=(b, n, s, n), dtype=paddle.int32)

    flashmask_info = fm.FlashMaskInfoPaddle(
        startend_row_indices=inp,
        is_causal=is_causal,
    )
    fm.prepare_block_maxmin(flashmask_info, kBlockN)

    check_prepare_block_maxmin(flashmask_info, kBlockN)

