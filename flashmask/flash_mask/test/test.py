import sys
import os

# 添加当前目录到路径
# sys.path.append('/workspace/paddle/flash-attention/flashmask/flash_mask')

print("="*60)
print("Testing FlashMask Import")
print("="*60)

from flash_mask.flashmask_attention_v3.interface import flashmask_attention

try:
    import paddle
    
    # 小批量测试数据
    query = paddle.randn([1, 16, 2, 32], dtype='float16')
    key = paddle.randn([1, 16, 2, 32], dtype='float16')
    value = paddle.randn([1, 16, 2, 32], dtype='float16')
    mask = paddle.to_tensor([8]*16, dtype='int32').reshape([1, 1, 16, 1])
    
    result = flashmask_attention(
        query=query,
        key=key,
        value=value,
        startend_row_indices=mask,
        causal=True
    )
    
    print(f"✓ Function executed successfully!")
    print(f"  Result shape: {result.shape}")
    print(f"  Result dtype: {result.dtype}")
    
except Exception as e:
    print(f"✗ Function execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete!")
print("="*60)