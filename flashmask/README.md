# FlashMask

This repository provides the official implementation of FlashMask, FlashMask V3 and FlashMask V4 from the following papers.

FlashMask: Efficient and Rich Mask Extension of FlashAttention.
+ Paper: https://arxiv.org/abs/2410.01359
+ Blog: https://zhuanlan.zhihu.com/p/4539730179

The core equation utilized in FlashMask is as follows:

$$
\text{result} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}} + M\right) \cdot V
$$

In this equation:
+ $Q$, $K$, and $V$ are the input tensors to the attention module.
+ All these tensors share the same dimensions.
+ $d$ denotes the size of the last dimension of these tensors.
+ $M$ represents the column-wise sparse mask introduced by FlashMask.

## FlashMask Feature Comparison

<table>
  <thead>
    <tr>
      <th>Training/Inference</th>
      <th>Feature</th>
      <th>FlashMask</th>
      <th>FlashMask V3</th>
      <th>FlashMask V4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8" align="center" style="vertical-align: middle;"><strong>Training</strong></td>
      <td>Custom Mask</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Context Parallel</td>
      <td>✅</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Block Mask</td>
      <td>❌</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>MLA</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Linear Attention</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Support Head Dim</td>
      <td>up to 256</td>
      <td>up to 256</td>
      <td>64, 128</td>
    </tr>
    <tr>
      <td>Supported DataType</td>
      <td>FP16, BF16</td>
      <td>BF16</td>
      <td>BF16</td>
    </tr>
    <tr>
      <td>Deterministic</td>
      <td>✅</td>
      <td>✅(Support Head Dim <= 128 Only)</td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="5" align="center" style="vertical-align: middle;"><strong>Inference</strong></td>
      <td>Custom Mask</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>PagedAttn</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Split KV</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>PackGQA</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>FP8</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td rowspan="2" align="center" style="vertical-align: middle;"><strong>Supported Framework</strong></td>
      <td>Paddle</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td>❌</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>


## Performance
FlashMask V3 is optimized for Hopper GPUs (e.g. H800).

### Flash Mask

FlashMask V3 shows substantial speedups across different head dimensions. The following benchmarks represent the performance improvement range across various sequence lengths.

<img width="5400" height="3600" alt="2761a8aaec6e13c7f7c5882d2962dad5" src="https://github.com/user-attachments/assets/cc8c0913-d3a8-4d0d-b6c3-83a299886225" />

Head Dimension 128
+ vs. FlashMask: 40.2% ~ 141.1% Increase
+ vs. FlexAttention: 7.3% ~ 67.5% Increase

Head Dimension 256
+ vs. FlashMask: 11.1% ~ 106.2% Increase
+ vs. FlexAttention: 66.9% ~ 212.2% Increase

### Block Mask

FlashMask V3 demonstrates a substantial performance advantage over Block Attention, as shown in the benchmark. Across various sequence lengths (8K, 32K, 128K) and configurations, it achieves a 75.7% to 197.3%​ speedup in forward computation and 48.0% to 94.4% speedup in backward computation.

<img width="5400" height="1800" alt="image" src="https://github.com/user-attachments/assets/bc79b760-9fbe-49d6-a79c-25047904b977" />

<img width="5400" height="1800" alt="image" src="https://github.com/user-attachments/assets/613421ad-cdb5-4ad4-b90b-776d6de9f8fc" />

Head Dimension 128, Fwd
+ vs. BlockAttention: 75.7% ~ 197.3% Increase

Head Dimension 128, Bwd
+ vs. BlockAttention: 48.0% ~ 94.4% Increase


### Distributed
TODO


## Installation
### Paddle
#### 1. FlashMask & FlashMask V3
Install Latest Stable Release or Nightly Release

For detailed information about installation, please view [Quick Install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html).

#### 2. FlashMask V4
```
cd flash-attention/flashmask
python3 setup.py install
```

### PyTorch
FlashMask V3
```
flash-attention/csrc/flashmask_v2
python setup.py install
```

FlashMask V4
```
TODO
```


## How to us FlashMask

```python
from flash_mask.cute.interface import flashmask_attention
```

```python
def flashmask_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: Tensor | None = None,
):
    """

    Args:
        query (Tensor):  The query tensor in the attention module.
            A 4-D tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        key (Tensor): The key tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        value (Tensor): The value tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        startend_row_indices(Tensor):
            A column-wise sparse attention mask row indices tensor.
            A 4-D tensor with shape [batch_size, k_num_heads, k_seq_len, {1, 2, 4}].
            The dtype must be int32. k_num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
            Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings.

            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 1],
              indicating unidirectional attention. The value represents the starting row index of the left
              lower triangular mask in the dense mask. The value startend_row_indices[..., 0] indicates that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked.
            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating unidirectional attention. The values represent the starting and ending row indices of
              the left lower triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1]-th row (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating bidirectional attention. The values represent the starting row index of the left
              lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 1]-th row upwards (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 4] ,
              indicating bidirectional attention. The values represent the start and end row indices of the
              left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:4] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1] row (exclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 2]-th row downwards (inclusive) but above the startend_row_indices[..., 3] row (exclusive) will be masked.

        dropout (float): The dropout ratio. Default is 0.0.
        causal (bool): Whether to enable causal mode. Default is False.
        window_size (int|tuple, optional): Indicates the window size of sliding window local attention.
            If causal mode is enabled, Query at position i will only attend to keys between [i - window_size, i] or [i - window_size[0], i].
            If causal mode is disabled, Query at position i will only attend to keys between [i - window_size, i + window_size] or [i - window_size[0], i + window_size[1]].
        return_softmax_lse (bool): Whether to return the log-sum-exp of the softmax. Default is False.
        return_seed_offset (bool): Whether to return the random seed offset. Default is False.
        fixed_seed_offset(Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name (str): The name to select Generator.
        training (bool): Whether the module is in training mode. Default is True.
        name (str, optional): Name of the operation. Default is None. Normally, users do not need to set this property.
            For more information, refer to :ref:`api_guide_Name` .
        block_mask (tensor, optional):
            A 4-D integer mask tensor indicating whether each block in the attention matrix should be kept or masked. Must be used together with flashmask.
            The shape should be [batch_size, num_heads, blocklen_q, blocklen_k], where:

            blocklen_q = ceil(seqlen_q / 128), i.e., block_mask.shape[2] must be (seqlen_q + 127) // 128
            blocklen_k = ceil(seqlen_k / 128), i.e., block_mask.shape[3] must be (seqlen_k + 127) // 128
            block_mask.shape[1] (number of heads) must match the num_heads dimension of the flashmask
            Both seqlen_q and seqlen_k must be less than or equal to 128 * 1024
            The dtype should be int32, and each element should be either 0 or 1.
            A value of 1 indicates that the corresponding block is kept (not masked), while 0 means the block is masked.

            Usage Notes:

            Only supported when blockdim_q = blockdim_k = 128 now.
            Only supported when headdim = 128 now.
            This argument must be provided together with flashmask.
            The mask will be applied at the block level: each [i, j] position in block_mask controls whether the corresponding [128 x 128] block in the attention matrix is masked.
            Any mismatch in expected shape or head dimension will raise an error.


    Returns
        Tensor. The computed attention result with the same shape as the input `query`.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Hint:
        This API supports GQA.
    """
```

For a example

```python
//// TODO


```




## Copyright and License
PaddlePaddle/flash-attention is provided under the Apache-2.0 license.
