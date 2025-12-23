import torch
import flash_mask_2._C as C  # PyTorch FlashMask

torch.manual_seed(2023)
device = 'cuda'
dtype = torch.bfloat16
assert torch.cuda.is_available()

q = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)
k = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)
v = torch.rand((1, 10, 2, 32 ), dtype=dtype, device=device, requires_grad=True)

startend_row_indices = torch.tensor([8]*10 + [5]*10, dtype=torch.int32).reshape([1, 2, 10, 1,])

output=torch.ops.flash_mask.fwd(q=q, k=k, v=v, startend_row_indices=startend_row_indices, is_causal=True)
