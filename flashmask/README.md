# FlashMask
Efficient and Rich Mask Extension of FlashAttention

Paper: https://arxiv.org/abs/2410.01359

## FlashMask V3
FlashMask V3 is optimized for Hopper GPUs (e.g. H100).
<img width="5400" height="3600" alt="2761a8aaec6e13c7f7c5882d2962dad5" src="https://github.com/user-attachments/assets/cc8c0913-d3a8-4d0d-b6c3-83a299886225" />

#### Performance
FlashMask V3 shows substantial speedups across different head dimensions. The following benchmarks represent the performance improvement range across various sequence lengths.

Head Dimension 128
+ vs. FlashMask: 40.2% ~ 141.1% Increase
+ vs. FlexAttention: 7.3% ~ 67.5% Increase

Head Dimension 256
+ vs. FlashMask: 11.1% ~ 106.2% Increase
+ vs. FlexAttention: 66.9% ~ 212.2% Increase

