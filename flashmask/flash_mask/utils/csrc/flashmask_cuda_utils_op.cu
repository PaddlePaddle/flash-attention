#include "paddle/extension.h"
#include <cuda_runtime.h>
#include <vector>

namespace {

int64_t ElemSize(paddle::DataType dtype) {
    switch (dtype) {
        case paddle::DataType::BFLOAT16:
        case paddle::DataType::FLOAT16:
            return 2;
        case paddle::DataType::FLOAT32:
            return 4;
        default:
            PD_THROW("Unsupported dtype; expected bf16/fp16/fp32.");
    }
}

void CheckBshd(const paddle::Tensor& x, const char* name) {
    PD_CHECK(x.is_gpu(), name, " must be a GPU tensor.");
    PD_CHECK(x.shape().size() == 4, name, " must be a 4D BSHD tensor.");
    PD_CHECK(x.is_contiguous(), name, " must be contiguous.");
    ElemSize(x.type());
}

void CheckSameShapeTypePlace(const paddle::Tensor& a, const paddle::Tensor& b, const char* an, const char* bn) {
    PD_CHECK(a.shape() == b.shape(), an, " and ", bn, " must have the same shape.");
    PD_CHECK(a.type() == b.type(), an, " and ", bn, " must have the same dtype.");
    PD_CHECK(a.place() == b.place(), an, " and ", bn, " must be on the same place.");
}

void CheckSliceBounds(int64_t total_seqlen, int64_t start, int64_t length) {
    PD_CHECK(start >= 0, "start must be non-negative, got ", start);
    PD_CHECK(length >= 0, "length must be non-negative, got ", length);
    PD_CHECK(start + length <= total_seqlen,
             "slice [", start, ":", start + length, ") exceeds seqlen ", total_seqlen);
}

std::vector<int64_t> BshdSliceShape(const paddle::Tensor& src, int64_t length) {
    const auto shape = src.shape();
    return {shape[0], length, shape[2], shape[3]};
}

void CopyBshdAxis1Slice(
    const paddle::Tensor& src,
    paddle::Tensor& dst,
    int64_t start,
    int64_t length) {
    const auto src_shape = src.shape();
    const int64_t batch = src_shape[0];
    const int64_t total_seqlen = src_shape[1];
    const int64_t inner = src_shape[2] * src_shape[3];
    const int64_t elem_size = ElemSize(src.type());

    const size_t src_pitch = static_cast<size_t>(total_seqlen * inner * elem_size);
    const size_t dst_pitch = static_cast<size_t>(length * inner * elem_size);
    const auto* src_ptr = reinterpret_cast<const char*>(src.data()) + start * inner * elem_size;
    auto* dst_ptr = reinterpret_cast<char*>(dst.data());

    cudaError_t err = cudaMemcpy2DAsync(
        dst_ptr,
        dst_pitch,
        src_ptr,
        src_pitch,
        dst_pitch,
        static_cast<size_t>(batch),
        cudaMemcpyDeviceToDevice,
        src.stream());
    PD_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync BSHD slice failed: ", cudaGetErrorString(err));
}

void CheckAccum(const paddle::Tensor& x, const char* name) {
    PD_CHECK(x.is_gpu(), name, " must be a GPU tensor.");
    PD_CHECK(x.shape().size() == 3, name, " must be a 3D [B, H, S*D] tensor.");
    PD_CHECK(x.is_contiguous(), name, " must be contiguous.");
    PD_CHECK(x.type() == paddle::DataType::FLOAT32, name, " must be float32.");
}

void ZeroAccumAxis1(
    paddle::Tensor& x,
    int64_t start,
    int64_t length,
    int64_t hdim,
    bool split,
    const char* name) {
    CheckAccum(x, name);
    PD_CHECK(hdim > 0, name, " hdim must be positive, got ", hdim);

    const auto shape = x.shape();
    const int64_t rows = shape[0] * shape[1];
    const int64_t flat = shape[2];
    const int64_t elem_size = 4;
    PD_CHECK(!split || flat % 2 == 0, name, " split layout requires even flat dim, got ", flat);

    const int64_t plane = split ? flat / 2 : flat;
    PD_CHECK(plane % hdim == 0, name, " plane dim ", plane, " must be divisible by hdim ", hdim);
    CheckSliceBounds(plane / hdim, start, length);

    auto* base = reinterpret_cast<char*>(x.data());
    const size_t pitch = static_cast<size_t>(flat * elem_size);
    const size_t width = static_cast<size_t>(length * hdim * elem_size);
    const size_t height = static_cast<size_t>(rows);
    const size_t offset = static_cast<size_t>(start * hdim * elem_size);

    cudaError_t err = cudaMemset2DAsync(base + offset, pitch, 0, width, height, x.stream());
    PD_CHECK(err == cudaSuccess, "cudaMemset2DAsync ", name, " low failed: ", cudaGetErrorString(err));
    if (split) {
        const size_t split_offset = static_cast<size_t>(plane * elem_size);
        err = cudaMemset2DAsync(base + split_offset + offset, pitch, 0, width, height, x.stream());
        PD_CHECK(err == cudaSuccess, "cudaMemset2DAsync ", name, " high failed: ", cudaGetErrorString(err));
    }
}

}  // namespace

std::vector<paddle::Tensor> BshdSliceContiguousKv(
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    int64_t start,
    int64_t length) {
    CheckBshd(key, "key");
    CheckBshd(value, "value");
    CheckSameShapeTypePlace(key, value, "key", "value");
    CheckSliceBounds(key.shape()[1], start, length);

    auto key_out = paddle::empty(BshdSliceShape(key, length), key.type(), key.place());
    auto value_out = paddle::empty(BshdSliceShape(value, length), value.type(), value.place());
    CopyBshdAxis1Slice(key, key_out, start, length);
    CopyBshdAxis1Slice(value, value_out, start, length);
    return {key_out, value_out};
}

std::vector<paddle::Tensor> AccumZeroAxis1KvInplace(
    paddle::Tensor& dk_accum,
    paddle::Tensor& dv_accum,
    int64_t start,
    int64_t length,
    int64_t dk_hdim,
    int64_t dv_hdim,
    bool dk_split,
    bool dv_split) {
    PD_CHECK(dk_accum.place() == dv_accum.place(), "dk_accum and dv_accum must be on the same place.");
    ZeroAccumAxis1(dk_accum, start, length, dk_hdim, dk_split, "dk_accum");
    ZeroAccumAxis1(dv_accum, start, length, dv_hdim, dv_split, "dv_accum");
    // Output aliases are established solely via SetInplaceMap above;
    // an empty return avoids triggering the [CustomOp] in-place size-match hint.
    return {};
}

PD_BUILD_OP(bshd_slice_contiguous_kv)
    .Inputs({"Key", "Value"})
    .Outputs({"KeyOut", "ValueOut"})
    .Attrs({"start: int64_t", "length: int64_t"})
    .SetKernelFn(PD_KERNEL(BshdSliceContiguousKv));

PD_BUILD_OP(accum_zero_axis1_kv)
    .Inputs({"DkAccum", "DvAccum"})
    .Outputs({"DkOut", "DvOut"})
    .Attrs({
        "start: int64_t",
        "length: int64_t",
        "dk_hdim: int64_t",
        "dv_hdim: int64_t",
        "dk_split: bool",
        "dv_split: bool"
    })
    .SetInplaceMap({{"DkAccum", "DkOut"}, {"DvAccum", "DvOut"}})
    .SetKernelFn(PD_KERNEL(AccumZeroAxis1KvInplace));
