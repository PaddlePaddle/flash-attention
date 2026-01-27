# FlashMask

This repository provides the official implementation of FlashMask and FlashMask V3 from the following papers.

FlashMask: Efficient and Rich Mask Extension of FlashAttention.
+ Paper: https://arxiv.org/abs/2410.01359
+ Blog: https://zhuanlan.zhihu.com/p/4539730179

FlashMask V3 is optimized for Hopper GPUs (e.g. H800).

## Performance
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
TODO(baoqiwen)


## How to us FlashMask

```python
from flash_mask.cute.interface import flashmask_attention
```

```python
def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
):
"""
  query / key / value: Core input tensors for the attention mechanism.
  startend_row_indices: Defines start and end positions for variable-length sequences.
  dropout: Fraction of units to drop during training to prevent overfitting.
  causal: Whether to apply a causal mask.
  window_size: The size of the sliding window for local attention.
  return_softmax_lse: Whether to return the Log-Sum-Exp value for gradients.
  return_seed_offset: Whether to return the random seed offset for reproducibility.
  fixed_seed_offset: Manually specify the seed offset to fix Dropout behavior.
  rng_name: Name of the random number generator to be used.
  training: Flag indicating whether the model is in training or inference.
  name: A custom name for this operation.
  softmax_scale: Scaling factor to adjust the dot-product attention magnitude.
  block_mask: Block-based mask tensor for complex sparse attention patterns.
"""
```

For a example

```python
TODO
```



## Copyright and License
PaddlePaddle/flash-attention is provided under the Apache-2.0 license.
