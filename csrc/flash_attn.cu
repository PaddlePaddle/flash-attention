#pragma once  // NOLINT

#include <cuda.h>          // NOLINT
#include <cuda_runtime.h>  // NOLINT

#include <cassert>
#include <vector>

#include "paddle/extension.h"
#include "capi/flash_attn.h"
static std::pair<uint64_t, uint64_t> GenerateRNGState(
    const phi::GPUContext& ctx,
    const paddle::optional<paddle::Tensor>& fixed_seed_offset,
    const std::string& rng_name,
    const int64_t batch_size,
    const int64_t num_heads) {
  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    uint64_t seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    uint64_t offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
    return std::make_pair(seed, offset);
  } else {
    uint64_t inc = batch_size * num_heads * 32;
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    // Error phi::Generator * gen = ctx.GetGenerator();
    // Error seed_offset_pair = gen->IncrementOffset(inc);
    return seed_offset_pair;
  }
}

static std::vector<int64_t> GetAttnMaskDims(const paddle::Tensor* attn_mask) {
  std::vector<int64_t> mask_dim_4d;
  if (attn_mask) {
    const auto& origin_dims = attn_mask->shape();
    auto rank = origin_dims.size();
    //#PADDLE_ENFORCE_GE(
    //#    rank,
    //#    4,
    //#    phi::errors::InvalidArgument(
    //#        "The number of dimensions of attn_mask is expected to be greater "
    //#        "or equal to 4, but recieved %d. The shape of attn_mask is {%s}",
    //#        rank,
    //#        origin_dims));

    int64_t first_dim = 1;
    for (int i = 0; i < rank - 3; i++) {
      first_dim *= origin_dims[i];
    }
    mask_dim_4d = {first_dim,
                   origin_dims[rank - 3],
                   origin_dims[rank - 2],
                   origin_dims[rank - 1]};
  }
  return mask_dim_4d;
}

struct FlashAttnParamsBase {
  int batch_size;
  // for padded kernel, max_seqlen_q and seqlen_q is the same.
  int64_t max_seqlen_q;
  // for padded kernel, max_seqlen_k and seqlen_k is the same.
  int64_t max_seqlen_k;
  int num_heads;
  int num_heads_k;
  int head_size;

  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int head_size_rounded;

  bool is_bf16;
  float softmax_scale;
  std::vector<int64_t> softmax_lse_dims;

  bool causal;
  std::vector<int64_t> mask_dims;
  const paddle::Tensor* attn_mask_tensor;

  FlashAttnParamsBase(const int _batch_size,
                      const int64_t _max_seqlen_q,
                      const int64_t _max_seqlen_k,
                      const int _num_heads,
                      const int _num_heads_k,
                      const int _head_size,
                      const float _scale,
                      const bool _causal,
                      const paddle::DataType q_dtype,
                      const paddle::optional<paddle::Tensor>& attn_mask)
      : batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads),
        head_size(_head_size),
        softmax_scale(_scale),
        causal(_causal),
        attn_mask_tensor(attn_mask.get_ptr()) {
    is_bf16 = q_dtype == paddle::DataType::BFLOAT16;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    head_size_rounded = round_multiple(head_size, 32);
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    softmax_lse_dims = {batch_size, num_heads, seqlen_q_rounded};

    if (attn_mask_tensor) {
      //PADDLE_ENFORCE_NE(causal,
      //                  true,
      //                  phi::errors::InvalidArgument(
      //                      "When attn_mask is set, causal can not be true."));

      //PADDLE_ENFORCE_EQ(
      //    attn_mask->dtype(),
      //    q_dtype,
      //    phi::errors::InvalidArgument(
      //        "attn_mask is expected to have the same data type with q."));

      mask_dims = GetAttnMaskDims(attn_mask_tensor);
    }
  }
};

template <typename T>
struct FlashAttnFwdParamsV2 : public FlashAttnParamsBase {
  float dropout;
  bool return_softmax;
  uint64_t seed;
  uint64_t offset;
  paddle::Tensor rng_state;
  paddle::Tensor* softmax;
  paddle::Tensor* softmax_lse;
  paddle::Tensor* seed_offset;

  FlashAttnFwdParamsV2(const phi::GPUContext& ctx,
                       const int _batch_size,
                       const int64_t _max_seqlen_q,
                       const int64_t _max_seqlen_k,
                       const int _num_heads,
                       const int _num_heads_k,
                       const int _head_size,
                       const float _dropout,
                       const float _scale,
                       const bool _causal,
                       const bool _return_softmax,
                       const paddle::DataType q_dtype,
                       const bool is_test,
                       const std::string& rng_name,
                       const paddle::optional<paddle::Tensor>& fixed_seed_offset,
                       const paddle::optional<paddle::Tensor>& attn_mask,
                       paddle::Tensor* _softmax,
                       paddle::Tensor* _softmax_lse,
                       paddle::Tensor* _seed_offset)
      : FlashAttnParamsBase(_batch_size,
                            _max_seqlen_q,
                            _max_seqlen_k,
                            _num_heads,
                            _num_heads_k,
                            _head_size,
                            _scale,
                            _causal,
                            q_dtype,
                            attn_mask),
        dropout(_dropout),
        return_softmax(_return_softmax),
        softmax(_softmax),
        softmax_lse(_softmax_lse),
        seed_offset(_seed_offset) {
    dropout = is_test ? 0.0f : _dropout;

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = paddle::empty({2}, phi::CppTypeToDataType<int64_t>::Type());

    auto seed_offset_pair = GenerateRNGState(
        ctx, fixed_seed_offset, rng_name, batch_size, num_heads);
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;

    seed_offset->reshape({2});
    int64_t seed_offset_data[2];
    seed_offset_data[0] = static_cast<int64_t>(seed);
    seed_offset_data[1] = static_cast<int64_t>(offset);
    //tensor.cc
    softmax_lse->reshape(softmax_lse_dims);
    // Error paddle::Tensor tp = paddle::empty(softmax_lse_dims, phi::CppTypeToDataType<float>::Type());

    if (return_softmax) {
      //PADDLE_ENFORCE_EQ(
      //    dropout > 0.0f,
      //    true,
      //    phi::errors::InvalidArgument(
      //        "return_softmax is only supported when dropout > 0.0"));

      softmax->reshape(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      // Error ctx.template Alloc<T>(softmax);
    }
  }
};

struct FlashAttnBwdParamsV2 : public FlashAttnParamsBase {
  float dropout;
  uint64_t seed;
  uint64_t offset;
  paddle::Tensor softmax_d;
  paddle::Tensor dq_accum;
  paddle::Tensor rng_state;

  FlashAttnBwdParamsV2(const phi::GPUContext& ctx,
                       const int _batch_size,
                       const int64_t _max_seqlen_q,
                       const int64_t _max_seqlen_k,
                       const int _num_heads,
                       const int _num_heads_k,
                       const int _head_size,
                       const float _dropout,
                       const float _scale,
                       const bool _causal,
                       const paddle::DataType q_dtype,
                       const paddle::optional<paddle::Tensor>& attn_mask,
                       const int64_t* seed_offset_data)
      : FlashAttnParamsBase(_batch_size,
                            _max_seqlen_q,
                            _max_seqlen_k,
                            _num_heads,
                            _num_heads_k,
                            _head_size,
                            _scale,
                            _causal,
                            q_dtype,
                            attn_mask),
        dropout(_dropout) {
    seed = static_cast<uint64_t>(seed_offset_data[0]);
    offset = static_cast<uint64_t>(seed_offset_data[1]);

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = paddle::empty({2}, phi::CppTypeToDataType<int64_t>::Type()); 

    // gradient of softmax_lse
    softmax_d = paddle::empty(softmax_lse_dims,phi::CppTypeToDataType<float>::Type());

    // an internal gradient of q, which will be further accumulated.
    dq_accum = paddle::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded},phi::CppTypeToDataType<float>::Type());
  }
};

static void CheckFlashAttnStatus(const bool status) {
  // Error PADDLE_ENFORCE_EQ(status,
  // Error                   true,
  // Error                   phi::errors::External(
  // Error                       "Error in Flash-Attention, detail information is: %s",
  // Error                       phi::dynload::flash_attn_error()));
}

static void RaiseNotSupportedError() {
  // ErrorPADDLE_THROW(
  // Error    phi::errors::Unimplemented("FlashAttention is unsupported, please check "
  // Error                               "the GPU compability and CUDA Version."));
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const paddle::Tensor& q,
                     const paddle::Tensor& k,
                     const paddle::Tensor& v,
                     const paddle::optional<paddle::Tensor>& fixed_seed_offset,
                     const paddle::optional<paddle::Tensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     paddle::Tensor* out,
                     paddle::Tensor* softmax,
                     paddle::Tensor* softmax_lse,
                     paddle::Tensor* seed_offset) {
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.shape();
//Error  PADDLE_ENFORCE_EQ(dims.size(),
//Error                    4,
//Error                    phi::errors::InvalidArgument(
//Error                        "flash_attn receive input with dim "
//Error                        "[batch_size, seq_len, num_heads, head_dim]"));
//Error
  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.shape()[1];
  const int64_t num_heads_k = k.shape()[2];

  // TODO(umiswing): Add check shape

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  FlashAttnFwdParamsV2<T> params = FlashAttnFwdParamsV2<T>(ctx,
                                                           batch_size,
                                                           seqlen_q,
                                                           seqlen_k,
                                                           num_heads,
                                                           num_heads_k,
                                                           head_size,
                                                           dropout,
                                                           softmax_scale,
                                                           causal,
                                                           return_softmax,
                                                           q.dtype(),
                                                           is_test,
                                                           rng_name,
                                                           fixed_seed_offset,
                                                           attn_mask,
                                                           softmax,
                                                           softmax_lse,
                                                           seed_offset);

  //VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.shape() << "], k.shape=["
  //         << k.shape() << "], v.shape=[" << v.shape() << "]";
  //VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
  //         << ", seed=" << params.seed << ", offset=" << params.offset;
  //VLOG(10) << "[FlashAttn Forward] softmax_scale=" << softmax_scale
  //         << ", softmax_unscale=" << softmax_unscale;
  //if (attn_mask.get_ptr()) {
  //  VLOG(10) << "[FlashAttn Forward] attn_mask.shape=["
  //           << (attn_mask.get_ptr())->shape() << "]";
  //}

  //Error  ctx.template Alloc<T>(out);

  cudaStream_t stream = q.stream();

  bool succ = flash_attn_fwd(
      q.data(),
      k.data(),
      v.data(),
      params.rng_state.data(),
      out->data(),
      params.return_softmax ? params.softmax->data() : nullptr,
      params.softmax_lse->data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      softmax_unscale,
      params.causal,
      params.return_softmax,
      params.is_bf16,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.mask_dims.data());
  CheckFlashAttnStatus(succ);
}
std::vector<paddle::Tensor> FaFwd(
                                   const paddle::Tensor& q,
                                   const paddle::Tensor& k,
                                   const paddle::Tensor& v,
                                   const paddle::optional<paddle::Tensor>& fixed_seed_offset,
                                   const paddle::optional<paddle::Tensor>& attn_mask,
                                   float dropout,
                                   bool causal,
                                   bool return_softmax,
                                   bool is_test,
                                   const std::string& rng_name) {
  return_softmax = false;
  paddle::Tensor out = paddle::empty(q.shape(), q.type());
  // out.set_layout(q.layout());
  paddle::Tensor softmax = paddle::empty({1}, q.type());
  paddle::Tensor softmax_lse = paddle::empty({1}, q.type());
  paddle::Tensor seed_offset = paddle::empty({1}, q.type());
  auto place = q.place(); 
  const phi::GPUContext *ctx{nullptr};
  //Error auto ctx = phi::GPUContext();
 //Error auto ctx = new phi::GPUContext(place);
  switch(q.type()){
    case paddle::DataType::FLOAT16:
      FlashAttnKernel<phi::dtype::float16,phi::GPUContext>(*ctx,q,k,v, fixed_seed_offset, attn_mask, dropout, causal, return_softmax, is_test, rng_name, &out, &softmax, &softmax_lse, &seed_offset);
     break;
    case paddle::DataType::BFLOAT16:
      FlashAttnKernel<phi::dtype::bfloat16,phi::GPUContext>(*ctx,q,k,v, fixed_seed_offset, attn_mask, dropout, causal, return_softmax, is_test, rng_name, &out, &softmax, &softmax_lse, &seed_offset);
     break;
    default:
     break;
      // Error 
    }
  return {out, softmax, softmax_lse, seed_offset};
}

std::vector<std::vector<int64_t>> FaFwdInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape,
    std::vector<int64_t> fixed_seed_offset,
    std::vector<int64_t> mask_shape,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name) {
  return {q_shape, k_shape, v_shape, mask_shape};
}

PD_BUILD_OP(flash_attn_with_mask)
    .Inputs({"q", "k", "v", "fixed_seed_offset","attn_mask"})
    .Outputs({"out", "softmax", "softmax_lse","seed_offset"})
    .Attrs({"dropout: float","causal:bool", "return_softmax:bool","is_test:bool","rng_name:std::string"})
    .SetKernelFn(PD_KERNEL(FaFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(FaFwdInferShape));

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const paddle::Tensor& q,
                         const paddle::Tensor& k,
                         const paddle::Tensor& v,
                         const paddle::Tensor& out,
                         const paddle::Tensor& softmax_lse,
                         const paddle::Tensor& seed_offset,
                         const paddle::optional<paddle::Tensor>& attn_mask,
                         const paddle::Tensor& dout,
                         float dropout,
                         bool causal,
                         paddle::Tensor* dq,
                         paddle::Tensor* dk,
                         paddle::Tensor* dv) {
  void* dq_ptr = nullptr;
  void* dk_ptr = nullptr;
  void* dv_ptr = nullptr;

  // Error ctx.template Alloc<T>(dq);
  dq_ptr = dq->data();

  paddle::Tensor dk_tmp;
  dk_tmp = paddle::empty_like(k, q.type());
  dk_ptr = dk_tmp.data();

  paddle::Tensor dv_tmp;
  dv_tmp = paddle::empty_like(v, q.type());
  dv_ptr = dv_tmp.data();

  const cudaStream_t stream = q.stream();

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.shape();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.shape()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.shape()[1];
  const int64_t num_heads_k = k.shape()[2];

  // TODO(umiswing): add shape check
  // Error PADDLE_ENFORCE_EQ(
  // Error     head_size_og,
  // Error     head_size,
  // Error     phi::errors::InvalidArgument(
  // Error         "flash_attn_bwd receive input with head_size_og == head_size"));

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           batch_size,
                           seqlen_q,
                           seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           softmax_scale,
                           causal,
                           q.dtype(),
                           attn_mask,
                           seed_offset.data<int64_t>());

  // Error VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.shape() << "], k.shape=["
  // Error          << k.shape() << "], v.shape=[" << v.shape() << "]";
  // Error VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
  // Error          << ", seed=" << params.seed << ", offset=" << params.offset;
  // Error VLOG(10) << "[FlashAttn Forward] softmax_scale=" << softmax_scale
  // Error          << ", softmax_unscale=" << softmax_unscale;
  // Error if (attn_mask.get_ptr()) {
  // Error   VLOG(10) << "[FlashAttn Backward] attn_mask.shape=["
  // Error            << (attn_mask.get_ptr())->shape() << "]";
  // Error }
#ifdef PADDLE_WITH_ADVANCED
  int num_splits = 1; // Error get_num_split();
#else
  int num_splits = 0; // Error get_num_split();
#endif

  bool succ = flash_attn_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      params.rng_state.data(),
      dq_ptr,
      dk_ptr,
      dv_ptr,
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      softmax_unscale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
  CheckFlashAttnStatus(succ);
}

std::vector<paddle::Tensor> FaBwd(
                         const paddle::Tensor& q,
                         const paddle::Tensor& k,
                         const paddle::Tensor& v,
                         const paddle::Tensor& out,
                         const paddle::Tensor& softmax_lse,
                         const paddle::Tensor& seed_offset,
                         const paddle::optional<paddle::Tensor>& attn_mask,
                         const paddle::Tensor& dout,
                         float dropout,
                         bool causal) {

  paddle::Tensor dq = paddle::empty(q.shape(), q.type());
  paddle::Tensor dk = paddle::empty(q.shape(), q.type());
  paddle::Tensor dv = paddle::empty(q.shape(), q.type());
  const phi::GPUContext *ctx{nullptr};
  //Error auto ctx = phi::GPUContext();
  switch(q.type()){
    case paddle::DataType::FLOAT16:
      FlashAttnGradKernel<phi::dtype::float16,phi::GPUContext>(*ctx,q,k,v,out, softmax_lse, seed_offset, attn_mask, dout, dropout, causal,&dq, &dk, &dv);
     break;
    case paddle::DataType::BFLOAT16:
      FlashAttnGradKernel<phi::dtype::bfloat16,phi::GPUContext>(*ctx,q,k,v,out, softmax_lse, seed_offset, attn_mask, dout, dropout, causal, &dq, &dk, &dv);
     break;
    default:
     break;
      // Error 
    }
  return {dq, dk, dv};
}

std::vector<std::vector<int64_t>> FaBwdInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> softmax_lse,
    std::vector<int64_t> seed_offset,
    std::vector<int64_t> mask_shape,
    std::vector<int64_t> dout,
    float dropout,
    bool causal) {
  return {q_shape, k_shape, v_shape, out_shape, softmax_lse,seed_offset, mask_shape,dout};
}

PD_BUILD_OP(flash_attn_with_mask_grad)
    .Inputs({"q", "k", "v", "out", "softmax_lse","seed_offset", "attn_mask","dout"})
    .Outputs({"dq", "dk", "dv"})
    .Attrs({"dropout: float","causal:bool"})
    .SetKernelFn(PD_KERNEL(FaBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(FaBwdInferShape));
