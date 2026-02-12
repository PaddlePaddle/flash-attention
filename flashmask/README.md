<div align="center">

# ⚡ FlashMask

**Efficient and Rich Mask Extension of FlashAttention**


[![arXiv](https://img.shields.io/badge/arXiv-2410.01359-b31b1b.svg)](https://arxiv.org/abs/2410.01359)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PaddlePaddle/flash-attention/blob/main/LICENSE)
[![Paddle](https://img.shields.io/badge/Paddle-3.3.0+-orange.svg)](https://www.paddlepaddle.org.cn/)

<p align="center">
  <img src="https://github.com/user-attachments/assets/f2a9cef4-9ad1-49a5-a791-711550ae5957" width="85%" alt="FlashMask Overview"/>
</p>
<p><i>Figure 1: (a) Supported Mask Types, (b) ColumnWise Sparse Representation, (c) Efficient Implementation</i></p>

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Citation](#-citation)

---


## 🎯 Overview
We propose FlashMask, an extension of FlashAttention that introduces a **column-wise sparse representation** of attention masks. 

This approach efficiently represents a wide range of mask types and facilitates the development of optimized kernel implementations. By adopting this novel representation, FlashMask achieves **linear memory complexity O(N)**, suitable for modeling long-context sequences. Moreover, this representation enables kernel optimizations that eliminate unnecessary computations by leveraging sparsity in the attention mask, without sacrificing computational accuracy, resulting in higher computational efficiency. 

### Core Equation

The core equation utilized in FlashMask is as follows:

$$
\text{result} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}} + M\right) \cdot V
$$

In this equation:
+ $Q$, $K$, and $V$ are the input tensors to the attention module.
+ All these tensors share the same dimensions.
+ $d$ denotes the size of the last dimension of these tensors.
+ $M$ represents the column-wise sparse mask introduced by FlashMask.

---

## ✨ Key Features

<details open>
<summary><b>🔢 Version Comparison Matrix</b></summary>

<table align="center">
<thead>
<tr>
<th>Category</th>
<th>Feature</th>
<th align="center">FlashMask</th>
<th align="center">FlashMask V3</th>
<th align="center">FlashMask V4</th>
</tr>
</thead>
<tbody>

<tr>
<td rowspan="7" align="center"><b>Training</b></td>
<td>Custom Mask</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td>Context Parallel</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td>Block Mask</td>
<td align="center">❌</td>
<td align="center">✅</td>
<td align="center">❌</td>
</tr>
<tr>
<td>Support Head Dim</td>
<td align="center">up to 256</td>
<td align="center">up to 256</td>
<td align="center">64, 128</td>
</tr>
<tr>
<td>Data Type</td>
<td align="center">FP16, BF16</td>
<td align="center">BF16</td>
<td align="center">BF16</td>
</tr>
<tr>
<td>Deterministic</td>
<td align="center">✅</td>
<td align="center">✅*</td>
<td align="center">✅</td>
</tr>
<tr>
<td>FP8</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>

<tr style="border-top: 2px solid #ddd;">
<td rowspan="5" align="center"><b>Inference</b></td>
<td>Custom Mask</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td>PagedAttn</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td>Split KV</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td>PackGQA</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>
<tr>
<td>FP8</td>
<td align="center">❌</td>
<td align="center">❌</td>
<td align="center">❌</td>
</tr>

<tr style="border-top: 2px solid #ddd;">
<td rowspan="2" align="center"><b>Framework</b></td>
<td>Paddle</td>
<td align="center">✅</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>
<tr>
<td>PyTorch</td>
<td align="center">❌</td>
<td align="center">✅</td>
<td align="center">✅</td>
</tr>

</tbody>
</table>

<small>*V3 deterministic only for head dim ≤128</small>

</details>


---

## 🚀 Performance

### FlashMask
#### End-to-End Training Speedup
FlashMask achieves **1.65× to 3.22×** end-to-end speedup compared to dense FlashAttention methods across SFT, LoRA, DPO, and RM tasks.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ca490b4-b57b-4e6a-a913-57517754a47e" width="80%" alt="Training Throughput"/>
</p>

Note: Benchmark data is sourced from the associated research paper.

#### Kernel-Level Benchmarks (A100-SXM 80GB)
FlashMask surpasses **FlexAttention (PyTorch 2.6.0)** by **12.1% to 60.7%** in kernel TFLOPs/s, achieving **37.8% to 62.3%** of theoretical peak FLOPs/s.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2239af39-1787-4338-bfa3-cfc623e88151" width="80%" alt="Kernel Speed Comparison"/>
</p>

Note: Benchmark data is sourced from the associated research paper.

### FlashMask V3 (Hopper Optimized)

| Head Dim | vs FlashMask | vs FlexAttention |
|:--------:|:------------:|:----------------:|
| 128 | +40.2% ~ +141.1% | +7.3% ~ +67.5% |
| 256 | +11.1% ~ +106.2% | +66.9% ~ +212.2% |

Attention-Gym: commit f6ff20

PyTorch: 2.9.0.dev20250901+cu129

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc8c0913-d3a8-4d0d-b6c3-83a299886225" width="85%" alt="V3 Performance"/>
</p>


### Block Mask

FlashMask V3 demonstrates a substantial performance advantage over [Block Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), as shown in the benchmark. Across various sequence lengths (8K, 32K, 128K) and configurations, it achieves a 75.7% to 197.3%​ speedup in forward computation and 48.0% to 94.4% speedup in backward computation.

| Direction | Speedup Range |
|-----------|---------------|
| Forward | **+75.7% ~ +197.3%** |
| Backward | **+48.0% ~ +94.4%** |

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc79b760-9fbe-49d6-a79c-25047904b977" width="48%" alt="Block Mask Fwd"/>
  <img src="https://github.com/user-attachments/assets/613421ad-cdb5-4ad4-b90b-776d6de9f8fc" width="48%" alt="Block Mask Bwd"/>
</p>

---


### MARCO
🔧 Performance optimizations have been implemented and are currently being pushed upstream.

**MARCO: Mask-Aware Responsive Communication Overlap** eliminates Context Parallel bottlenecks through:

1. **Dynamic Load Balancing** — On-the-fly workload estimation for even distribution
2. **Communication Overlapping** — Hides KV all-gather latency via computation overlap

#### MARCO vs Magi-Attention (32K & 128K Sequences)

<p align="center">
  <img src="https://github.com/user-attachments/assets/a1dbf4df-6e51-47e8-9c5a-c883dd094166" width="32%" alt="MARCO Bench 1"/>
  <img src="https://github.com/user-attachments/assets/604583f8-c149-4888-ab2a-db8c9ad12985" width="32%" alt="MARCO Bench 2"/>
  <img src="https://github.com/user-attachments/assets/7753cb34-5d1d-4432-89f6-e587b86ffda6" width="32%" alt="MARCO Bench 3"/>
</p>

<details>
<summary><b>📊 Detailed Runtime Comparison (ms)</b></summary>
<table>
  <thead>
    <tr>
      <th>Seqlen</th>
      <th>Shape</th>
      <th>Operation</th>
      <th>magi-fwd (ms)</th>
      <th>magi-bwd (ms)</th>
      <th>magi-total (ms)</th>
      <th>FlashMask-fwd (ms)</th>
      <th>FlashMask-bwd (ms)</th>
      <th>FlashMask-total (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9" align="center" style="vertical-align: middle;"><strong>32K</strong></td>
      <td rowspan="3">b=2, hq=8<br>hkv=1, d=128</td>
      <td>Causal Document Mask</td>
      <td>3.34</td>
      <td>5.22</td>
      <td>8.56</td>
      <td>0.63</td>
      <td>2.07</td>
      <td>2.70</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>3.54</td>
      <td>5.33</td>
      <td>8.88</td>
      <td>0.92</td>
      <td>2.44</td>
      <td>3.37</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>3.39</td>
      <td>5.16</td>
      <td>8.55</td>
      <td>0.63</td>
      <td>2.07</td>
      <td>2.70</td>
    </tr>
    <tr>
      <td rowspan="3">b=2, hq=32<br>hkv=4, d=128</td>
      <td>Causal Document Mask</td>
      <td>3.88</td>
      <td>7.62</td>
      <td>11.50</td>
      <td>1.64</td>
      <td>7.26</td>
      <td>8.90</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>4.88</td>
      <td>11.95</td>
      <td>16.83</td>
      <td>2.70</td>
      <td>8.74</td>
      <td>11.44</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>3.98</td>
      <td>8.05</td>
      <td>12.03</td>
      <td>1.63</td>
      <td>7.26</td>
      <td>8.89</td>
    </tr>
    <tr>
      <td rowspan="3">b=2, hq=64<br>hkv=8, d=128</td>
      <td>Causal Document Mask</td>
      <td>5.07</td>
      <td>13.71</td>
      <td>18.78</td>
      <td>3.00</td>
      <td>14.19</td>
      <td>17.18</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>7.37</td>
      <td>22.30</td>
      <td>29.67</td>
      <td>5.08</td>
      <td>17.15</td>
      <td>22.24</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>5.27</td>
      <td>14.03</td>
      <td>19.30</td>
      <td>2.98</td>
      <td>14.20</td>
      <td>17.18</td>
    </tr>
    <tr>
      <td rowspan="9" align="center" style="vertical-align: middle;"><strong>128K</strong></td>
      <td rowspan="3">b=1, hq=8<br>hkv=1, d=128</td>
      <td>Causal Document Mask</td>
      <td>3.53</td>
      <td>5.65</td>
      <td>9.17</td>
      <td>0.98</td>
      <td>3.57</td>
      <td>4.55</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>3.71</td>
      <td>6.77</td>
      <td>10.48</td>
      <td>1.64</td>
      <td>4.28</td>
      <td>5.92</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>3.55</td>
      <td>5.68</td>
      <td>9.23</td>
      <td>1.08</td>
      <td>3.59</td>
      <td>4.67</td>
    </tr>
    <tr>
      <td rowspan="3">b=1, hq=32<br>hkv=4, d=128</td>
      <td>Causal Document Mask</td>
      <td>4.42</td>
      <td>10.41</td>
      <td>14.83</td>
      <td>2.53</td>
      <td>9.65</td>
      <td>12.18</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>5.50</td>
      <td>15.44</td>
      <td>20.94</td>
      <td>4.69</td>
      <td>13.58</td>
      <td>18.28</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>4.44</td>
      <td>10.53</td>
      <td>14.97</td>
      <td>2.81</td>
      <td>9.85</td>
      <td>12.67</td>
    </tr>
    <tr>
      <td rowspan="3">b=1, hq=64<br>hkv=8, d=128</td>
      <td>Causal Document Mask</td>
      <td>6.18</td>
      <td>19.10</td>
      <td>25.27</td>
      <td>4.66</td>
      <td>18.71</td>
      <td>23.37</td>
    </tr>
    <tr>
      <td>Document Mask</td>
      <td>8.81</td>
      <td>29.39</td>
      <td>38.20</td>
      <td>8.78</td>
      <td>27.18</td>
      <td>35.96</td>
    </tr>
    <tr>
      <td>Prefix LM Document Mask</td>
      <td>6.35</td>
      <td>19.70</td>
      <td>26.05</td>
      <td>5.14</td>
      <td>19.12</td>
      <td>24.26</td>
    </tr>
  </tbody>
</table>
</details>

### FlashMask V4
🔧 Performance optimizations have been implemented and are currently being pushed upstream.

Baseline: FA4 with mask_mod

Config: Head dim = 128

| Sequence Length | Speedup vs FA4 |
|-----------|---------------|
| 8k | **-0.4% ~ +57.2%** |
| 32k | **-0.7% ~ +39.5%** |
| 128k | **-3.4% ~ +20.9%** |

<p align="center">
  <img src="https://github.com/user-attachments/assets/27085b7b-6e86-44c0-9ec1-6922ea8c2325" width="85%" alt="V3 Performance"/>
</p>

---

## 📦 Installation
### 🥇 Paddle (Recommended)
#### FlashMask & FlashMask V3
FlashMask and FlashMask V3 are included in the standard PaddlePaddle distribution. No additional plugins are required.
For detailed information about installation, please view [Quick Install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html). 
```bash
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```


#### FlashMask V4
```bash
git clone https://github.com/PaddlePaddle/flash-attention.git
cd flash-attention/flashmask
python3 setup.py install
```

#### MARCO
```bash
git clone https://github.com/PaddlePaddle/flash-attention.git
cd flash-attention/csrc/utils/cp_balance/csrc/
python3 setup.py install
```

### PyTorch
#### FlashMask V3
```bash
git clone https://github.com/PaddlePaddle/flash-attention.git
cd flash-attention/csrc/flashmask_v2
python setup.py install
```

#### FlashMask V4
To use the FlashMask V4 features, you need to pull the specific implementation from the PaddlePaddle repository's [Pull Request #103](https://github.com/PaddlePaddle/flash-attention/pull/103). Follow the steps below:
```bash
git clone https://github.com/PaddlePaddle/flash-attention.git
cd flash-attention/flashmask
# Fetch and checkout the specific PR (Pull Request 103)
gh pr checkout 103
python setup.py install
```

## 🚀 Quick Start
### How to use FlashMask
#### Installation & Import
```python
from flash_mask.cute.interface import flashmask_attention
```

#### API Reference
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

#### Implementation Example
Here is a Python example demonstrating how to use the flashmask_attention interface with a block mask implementation.

```python
import pytest
import paddle
from functools import partial
from paddle.nn.functional.flash_attention import flashmask_attention


def generate_sliding_window_mask(batch_size, seqlen_q, seqlen_k, h, window_size = None):
    if window_size == None:
        window_size = 1024
        if seqlen_k != 8192:
            window_size = int(window_size * (seqlen_k / 8192))
            print(f"{seqlen_k=}, auto setting window_size to {window_size}")

    startend_row_indices = paddle.arange(
        window_size, seqlen_k + window_size, dtype="int32"
    ).reshape((1, 1, seqlen_k, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=seqlen_q
    ).repeat_interleave(batch_size, 0)

    causal=True
    return startend_row_indices, causal


# blockmask utils
def random_blockmask(shape, dtype='int32', is_causal=False, ref_q = None):
    mask = paddle.randint(0, 2, shape, dtype=paddle.int32)
    B, S, Q, K = shape
    return mask


def test_flashmask():
    paddle.seed(2024)

    batch_size = 28
    seqlen_q = 128
    seqlen_k = 128
    nheads = 16
    nheads_kv = 4
    nheads_startend_row_indices = 1

    d = 128
    dv = 128

    dtype = paddle.bfloat16

    fa_version = 3

    assert nheads % nheads_kv == 0
    q_ref = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)

    k_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)
    v_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)

    q_ref.stop_gradient = False
    k_ref.stop_gradient = False
    v_ref.stop_gradient = False

    q_bf16, k_bf16, v_bf16 = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q_bf16.stop_gradient = False
    k_bf16.stop_gradient = False
    v_bf16.stop_gradient = False

    q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]

    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    startend_row_indices, causal = generate_sliding_window_mask(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    blockmask = random_blockmask(
        shape=[
            startend_row_indices.shape[0],
            startend_row_indices.shape[1],
            (seqlen_q + 127) // 128,
            (seqlen_k + 127) // 128
        ],
        dtype=paddle.int32,
        is_causal=causal,
        ref_q = q_ref
    )

    if fa_version == 2:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fa_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    else:
        raise ValueError(
            f"Invalid flash attention version: {fa_version}"
        )

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
        block_mask=blockmask
    )
```


### How to use MARCO
#### FlashMask V3 overlapped layer
The following code presents a usable PyLayer object that calls overlapped all-gather automatically under the hood. The users themselves do not need to configure communication overlapping.

```python
import paddle
import paddle.nn.functional as F
from paddle import _C_ops
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.nn.functional.flash_attention import flashmask_attention
from paddle.autograd.py_layer import PyLayer

def rearrange_blocks(input_tensor, cp_size):
    # running time of the function is not recorded, since with workload balancer
    # we don't need this re-arange
    B, _, S, _ = input_tensor.shape
    n_blocks = cp_size * 2
    block_size = S // n_blocks
    
    blocks = input_tensor.reshape([B, 1, n_blocks, block_size, 2])
    
    indices = []
    for i in range(cp_size):
        indices.append(i)
        indices.append(n_blocks - 1 - i)
    
    reordered_blocks = blocks.index_select(index=paddle.to_tensor(indices), axis=2)
    output = reordered_blocks.reshape([B, 1, S, 2])
    return output

def reduce_scatter_any_axis_simpler(input_tensor, axis, group=None):
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor
    rank = group.rank

    assert input_tensor.shape[axis] % parallelism == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be ",
        f"divided exactly by context parallelism '{parallelism}'",
    )

    # split into CP-size chunks, for example, rank 1 holds:
    # [(1, 6), (2, 5), (3, 4), (0, 7)] (dual-chunk splits the KV into CP-size * 2 = 8 smaller chunks)
    chunks = paddle.split(input_tensor, parallelism, axis=axis)

    # re-arange chunks: the rank 1 example will be [(0, 7), (1, 6), (2, 5), (3, 4)]
    ordered_chunks = chunks[-rank:]
    ordered_chunks.extend(chunks[:-rank])

    # Perform alltoall communication, for example, rank 2 gathers all of the (2, 5) chunk from all other ranks
    # while it sends the local (0, 7), (1, 6), (3, 4) to all other ranks
    output_buffers = [paddle.empty(chunks[0].shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.alltoall(output_buffers, ordered_chunks, group=group, use_calc_stream=True)

    # Sum the received chunks
    result = paddle.stack(output_buffers, axis=0).sum(axis=0)
    return result


def get_group_unique_id(cg_group, rank_to_generate: int = 0):
    if cg_group.rank == rank_to_generate:
        unique_id = paddle.nn.functional.flashmask_get_unique_id()
    else:
        unique_id = paddle.zeros([128], dtype='uint8', device='cpu')
    result_list = []
    # must use all_gather_object, since all_gather (dist env is GPU 
    # yet tensor is CPU) will throw exception
    dist.all_gather_object(result_list, unique_id, group=cg_group)
    return result_list[rank_to_generate]

# If we are using the latest load-balancing strategy developed by haoyang:
# since the startend_row_indices are re-ordered [rank0][rank1][rank2][rank3]
# and in each rank, the startend_row_indinces are of the load-balanced state
# KV are also sorted, so we do not need to perform dual_chunk and rearrange 
# anymore
def cp_flashmask_overlap_forward(query, key, value, startend_row_indices, group, causal, is_training, unique_id: paddle.Tensor = None, disable_dual_chunk = False):
    rank = group.rank
    cp_size = group.world_size

    # All-gather key tensors across context parallel ranks

    # key_gathered = all_gather_balance(key, axis=1, group=group)
    # value_gathered = all_gather_balance(value, axis=1, group=group)
    seq_blocksize = query.shape[1] // 2

    # Preprocess indices for dual-chunk strategy
    if not disable_dual_chunk:
        startend_row_indices = preprocess_index_dual_chunks(
            startend_row_indices,
            chunk_id_first=rank,
            chunk_id_second=2 * cp_size - rank - 1,
            seq_blocksize=seq_blocksize,
            max_seqlen_q=seq_blocksize,
        )
        startend_row_indices = rearrange_blocks(startend_row_indices, cp_size)
    # cyclic left move on seqlen dim, so that the mask indices are shifted correctly
    processed_mask = paddle._C_ops.roll(startend_row_indices, shifts=-query.shape[1] * (rank + 1), axis=2)
    
    # using local key/value chunks, and overlap the sparse gather 
    # print(f"Start overlap flashmask attention: {rank} / {cp_size}")
    output, log_sum_exp = flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=processed_mask,
        causal=causal,
        return_softmax_lse=True,
        training=is_training,
        unique_id=unique_id,
        rank=rank,
        nranks=cp_size,
    )
    return output, log_sum_exp, startend_row_indices


def cp_flashmask_overlap_backward(
    query, key, value, startend_row_indices, output, log_sum_exp, output_grad, group, causal
):
    """
    Backward pass of context parallel flashmask attention with distributed overlap.

    This function implements the backward pass of flashmask attention with context parallelism,
    computing gradients for query, key, and value tensors.

    Args:
        query (paddle.Tensor): Query tensor
        key (paddle.Tensor): Key tensor
        value (paddle.Tensor): Value tensor
        startend_row_indices (paddle.Tensor): Processed startend_row_indices
        output (paddle.Tensor): Forward pass output
        log_sum_exp (paddle.Tensor): Log-sum-exp from forward pass
        output_grad (paddle.Tensor): Gradient of output
        group (paddle.distributed.Group): Communication group
        causal (bool): Whether causal attention was used

    Returns:
        tuple: (query_grad, key_grad, value_grad)
    """
    rank = group.rank
    cp_size = group.world_size
    # roll the local chunk to the first chunk position (while fwd roll the local chunk to the back)
    startend_row_indices = paddle._C_ops.roll(startend_row_indices, shifts=-query.shape[1] * rank, axis=2)

    query_grad, key_grad_gathered, value_grad_gathered = paddle._C_ops.flashmask_attention_v2_grad(
        query,
        key,
        value,
        output,
        log_sum_exp,
        startend_row_indices,
        None,  # block_mask
        output_grad,
        query.shape[-1] ** (-0.5),
        False,
        rank,
        cp_size,            # nranks
    )

    key_grad = reduce_scatter_any_axis_simpler(key_grad_gathered, axis=1, group=group)
    value_grad = reduce_scatter_any_axis_simpler(value_grad_gathered, axis=1, group=group)
    return query_grad, key_grad, value_grad

class OverlappedFlashMask(PyLayer):
    """
    CP FlashMask attention with NVSHMEM-based overlapping
    The fetch time of required KV chunks are overlapped (hidden), and the kernel
    will treat the attention calculation as using the entire KV tensor (local size * cp_size)
    so that we don't have throughput loss
    """

    is_first_call = True

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        group=None,
        mode="overlap",
    ):
        """
        Forward pass of FlashMask attention with context parallelism.

        Args:
            ctx: Context object for saving information for backward pass
            query (paddle.Tensor): Query tensor, pre-divided by CP size
            key (paddle.Tensor): Key tensor, pre-divided by CP size
            value (paddle.Tensor): Value tensor, pre-divided by CP size
            startend_row_indices (paddle.Tensor): Row indices for attention mask
            fixed_seed_offset (paddle.Tensor, optional): Fixed seed offset for dropout
            dropout (float): Dropout probability
            causal (bool): Whether to use causal attention
            training (bool): Whether in training mode
            mode (str): Attention mode, currently supports "allgather_kv"

        Returns:
            paddle.Tensor: Attention output

        Raises:
            NotImplementedError: If dropout > 0.0 or causal=True
            AssertionError: If query sequence length is not divisible by 2
        """
        # Validate input parameters
        if dropout > 0.0:
            raise NotImplementedError("Dropout is not supported in FlashMask context parallel yet.")

        if causal:
            raise NotImplementedError("FlashMaskContextParallel does not support causal=True yet.")

        if fixed_seed_offset is not None:
            raise NotImplementedError("Fixed seed offset is not supported yet.")

        if group is None:
            group = dist.fleet.get_hybrid_communicate_group().get_context_parallel_group()

        # Validate query sequence length for DualChunkSwap strategy
        assert query.shape[1] % 2 == 0, (
            f"Query sequence length must be divisible by 2. "
            f"FlashMaskContextParallel uses DualChunkSwap strategy for load balancing. "
            f"Current query sequence length: {query.shape[1]}"
        )
        assert key.shape[2] <= 4, (
            "KV head expected to be <= 4, since large KV head increases communication load,"
            f"Which might not be suitable for overlapping. Current KV num head: {key.shape[2]}"
        )
        assert startend_row_indices.shape[1] == 1, (
            f"Currently, we only support mask num head = 1, but got: {startend_row_indices.shape[1]}"
        )

        unique_id_tensor = None
        if OverlappedFlashMask.is_first_call:
            OverlappedFlashMask.is_first_call = False
            unique_id_tensor = get_group_unique_id(group)

        # Perform forward pass
        output, log_sum_exp, startend_row_indices = cp_flashmask_overlap_forward(
            query, key, value, startend_row_indices, group, causal, training, unique_id_tensor,
            disable_dual_chunk = mode == "balance_overlap"
        )

        # Save tensors for backward pass
        ctx.save_for_backward(query, key, value, output, log_sum_exp, startend_row_indices)
        ctx.group = group
        ctx.causal = causal

        return output

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass of FlashMask attention with context parallelism.

        Args:
            ctx: Context object with saved information
            output_grad (paddle.Tensor): Gradient of output

        Returns:
            tuple: Gradients for all input arguments
        """
        # Retrieve saved tensors
        query, key, value, output, log_sum_exp, startend_row_indices = ctx.saved_tensor()
        group = ctx.group
        causal = ctx.causal

        # Compute gradients
        query_grad, key_grad, value_grad = cp_flashmask_overlap_backward(
            query, key, value, startend_row_indices, output, log_sum_exp, output_grad, group, causal
        )

        return query_grad, key_grad, value_grad
```

#### FlashMask V3 using load balancing
Since load-balancing module works on sharding Q, K, V and rearrange mask tensors, the module can be used independently with overlap attention layer. Therefore, users are free to choose whether some of the functionalities should be switched off. The following code presents an example for using load balancing:

```python
import paddle
from paddle.distributed import fleet
from cp_balance.cp_balance import balance_flashmask_input

fleet.init(is_collective=True, strategy={
    "your_strategy": "this is a dummy one"
})
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

cp_size = cp_group.world_size
cp_rank = cp_group.rank
local_mask, buckets = balance_flashmask_input(startend_row_indices.clone(), cp_size, cp_rank)
# get local sharded QKV

full_q, full_k, full_v = get_full_qkv_from_some_source()

def scatter_balance(input_tensor, group=None, axis=0, buckets=None):
    """
    Evenly split input tensor along the specified axis across model parallel ranks.

    This function implements balanced scattering by taking chunks from both ends
    of the tensor to ensure load balancing across ranks.

    Args:
        input_tensor (paddle.Tensor): Input tensor to be scattered
        group (paddle.distributed.Group, optional): Communication group.
            If None, uses model parallel group from fleet
        axis (int, optional): Axis along which to scatter. Defaults to 0

    Returns:
        paddle.Tensor: Scattered tensor chunk for current rank

    Note:
        This API is different from distributed.scatter - it performs balanced
        splitting by taking chunks from both ends of the sequence.
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    rank = group.rank
    seq_len = input_tensor.shape[axis]

    assert len(buckets) == parallelism, "buckets should have same size as parallelism"
    assert seq_len % (parallelism * len(buckets[rank])) == 0, "seq_len must be divisible by parallelism * len(buckets[rank])"
    local_chunks = []
    balance_chunksize = seq_len // (parallelism * len(buckets[rank]))
    for(_, idx) in buckets[rank]:
        chunk_start = idx * balance_chunksize
        chunk_end = (idx + 1) * balance_chunksize
        chunk = paddle.slice(input_tensor, axes=[axis], starts=chunk_start, ends=chunk_end)
        local_chunks.append(chunk)
    return paddle.concat(local_chunks, axis=axis)

q = scatter_balance(full_q, group = cp_group, axis=1, buckets = buckets).contiguous()
k = scatter_balance(full_k, group = cp_group, axis=1, buckets = buckets).contiguous()
v = scatter_balance(full_v, group = cp_group, axis=1, buckets = buckets).contiguous()
```


## 📝 Citation
If you use FlashMask in your research or project, we appreciate that you use the following citations:

```bibtex
@article{wang2024flashmask,
  title={Flashmask: Efficient and rich mask extension of flashattention},
  author={Wang, Guoxia and Zeng, Jinle and Xiao, Xiyuan and Wu, Siming and Yang, Jiabin and Zheng, Lujing and Chen, Zeyu and Bian, Jiang and Yu, Dianhai and Wang, Haifeng},
  journal={arXiv preprint arXiv:2410.01359},
  year={2024}
}
```

