// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <memory>
#include <vector>

#include "utils.hpp"

// TODO: Use shared_ptr to use the same memory of BaseParams when calling
// forward/backward parameters
// TODO: Fix input constness
// Common argements used by both batched & grouped gemms
struct BaseParams {
  explicit BaseParams(const Index b, const Index max_seqlen_q,
                      const Index max_seqlen_kv, const Index h_q,
                      const Index h_kv, const Index d, const Index qseq_stride,
                      const Index kvseq_stride, const Index outseq_stride, const Index qhead_stride,
                      const Index kvhead_stride, const Index outhead_stride, const Index softmaxlse_batch_stride,
                      const float p_dropout, const float softmax_scale,
                      const bool is_causal, const bool is_bf16)
      : b(b), max_seqlen_q(max_seqlen_q), max_seqlen_kv(max_seqlen_kv),
        h_q(h_q), h_kv(h_kv), d(d), p_dropout(p_dropout),
        softmax_scale(softmax_scale), is_bf16(is_bf16),
        is_dropout(p_dropout > 0.0f), is_mnko_padding(false),
        is_causal(is_causal), q_seq_stride(qseq_stride),
        kv_seq_stride(kvseq_stride), out_seq_stride(outseq_stride),
        q_head_stride(qhead_stride), kv_head_stride(kvhead_stride),
        out_head_stride(outhead_stride),
        softmax_lse_batch_stride(softmaxlse_batch_stride) {

    is_mnko_padding = ((d % 32) != 0) || (d == 96);
    if (d > 128) {
      std::cout << "Unsupported head dimension" << std::endl;
    }
  }
  // The dimensions.
  Index b, max_seqlen_q, max_seqlen_kv, d;

  // The number of heads.
  Index h_q, h_kv;

  // The scaling factors for the kernel.
  float softmax_scale;
  // float softmax_scale_log2;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint8_t p_dropout_in_uint8_t;

  // seeds
  std::tuple<uint64_t, uint64_t> seeds;

  bool is_bf16;
  bool is_dropout;
  bool is_mnko_padding;
  bool is_causal;

  Index q_seq_stride;
  Index kv_seq_stride;
  Index out_seq_stride;

  Index q_head_stride;
  Index kv_head_stride;
  Index out_head_stride;

  Index softmax_lse_batch_stride;

  static inline const bool kIsUnitTestMode =
      get_env_("FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE");
  static inline const bool kIsDeterministic =
      get_env_("FLASH_ATTENTION_INTERNAL_DETERMINISTIC");
};

// Common Batched Arguments
struct BatchedParams : public BaseParams {
  explicit BatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out,
      void * softmax_lse, // TODO: forward reference, backward const reference
      const Index qbatch_stride, const Index kvbatch_stride, const Index outbatch_stride, 
      const Index qseq_stride, const Index kvseq_stride, const Index outseq_stride, 
      const Index qhead_stride, const Index kvhead_stride, const Index outhead_stride, 
      const Index softmaxlse_batch_stride,
      const float p_dropout, const float softmax_scale, const bool is_causal, const bool is_bf16)
      : BaseParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, qseq_stride, kvseq_stride, outseq_stride, qhead_stride, 
                  kvhead_stride, outhead_stride, softmaxlse_batch_stride, p_dropout, softmax_scale, is_causal, is_bf16),
        q_ptr(q), k_ptr(k), v_ptr(v),
        out_ptr(out), softmax_lse_ptr(softmax_lse),
        q_batch_stride(qbatch_stride), kv_batch_stride(kvbatch_stride),
        out_batch_stride(outbatch_stride) {
    if (!is_mnko_padding && d <= 32) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    } else if (!is_mnko_padding && d <= 64) {
      if (is_dropout) {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                     : true);
      } else {
        is_mnko_padding =
            ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 256) == 0 ? false
                                                                     : true);
      }
    } else if (!is_mnko_padding && d <= 128) {
      is_mnko_padding =
          ((max_seqlen_q % 128) == 0 && (max_seqlen_kv % 128) == 0 ? false
                                                                   : true);
    }

    // TODO: Change to tensor.shape()
    // Q layout [b, max_seqlen_q, h_q, d]
    q_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    q_strides =
        std::vector<Index>{q_batch_stride, q_head_stride, q_seq_stride, 1};

    // K layout [b, max_seqlen_kv, h_kv, d]
    k_lengths = std::vector<Index>{b, h_kv, max_seqlen_kv, d};
    k_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, kv_seq_stride, 1};

    // V layout [b, max_seqlen_kv, h_kv, d]
    v_lengths = std::vector<Index>{b, h_kv, d, max_seqlen_kv};
    v_strides =
        std::vector<Index>{kv_batch_stride, kv_head_stride, 1, kv_seq_stride};

    // Y layout [b, max_seqlen_q, h_q, d]
    out_lengths = std::vector<Index>{b, h_q, max_seqlen_q, d};
    out_strides = std::vector<Index>{out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

    // LSE layout [b, h_q, max_seqlen_q]
    lse_lengths = std::vector<Index>{b, h_q, max_seqlen_q};
    // std::vector<Index> lse_strides{h_q*max_seqlen_q, max_seqlen_q, 1};
  }

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  void *__restrict__ z_ptr;
  void *__restrict__ out_ptr;
  void *__restrict__ softmax_lse_ptr;

  Index q_batch_stride;
  Index kv_batch_stride;
  Index out_batch_stride;
  Index softmax_lse_batch_stride;

  std::vector<Index> q_lengths;
  std::vector<Index> q_strides;
  std::vector<Index> k_lengths;
  std::vector<Index> k_strides;
  std::vector<Index> v_lengths;
  std::vector<Index> v_strides;
  std::vector<Index> z_lengths;
  std::vector<Index> z_strides;
  std::vector<Index> out_lengths;
  std::vector<Index> out_strides;
  std::vector<Index> lse_lengths;
  // std::vector<Index> lse_strides;
};

// Forward Batched Arguments
struct FlashFwdBatchedParams : public BatchedParams {
  explicit FlashFwdBatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out,
      void * z,
      void * softmax_lse, // TODO: forward reference, backward const reference
      const float p_dropout, const float softmax_scale, const bool is_causal,
      const bool return_softmax, const bool is_bf16)
      : BatchedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, softmax_lse, max_seqlen_q * h_q * d, max_seqlen_kv * h_kv * d, max_seqlen_q * h_q * d,
                      h_q * d, h_kv * d, h_q * d, d, d,
                      d, max_seqlen_q * h_q, p_dropout, softmax_scale, is_causal, is_bf16) {
    z_ptr = return_softmax ? z : nullptr;

    // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
    z_lengths = std::vector<Index>{b, h_q, max_seqlen_q, max_seqlen_kv};
    z_strides =
        std::vector<Index>{h_q * max_seqlen_q * max_seqlen_kv,
                           max_seqlen_q * max_seqlen_kv, max_seqlen_kv, 1};
  }

  bool return_softmax;
};

// Backward Batched Arguments
struct FlashBwdBatchedParams : public BatchedParams {
  explicit FlashBwdBatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out, // TODO: Fix constness
      void * dout, void * dq, void * dk,
      void * dv, void * dsoftmax,
      void * softmax_lse, // TODO: Fix constness
      const float p_dropout, const float softmax_scale, const bool is_causal, const bool is_bf16)
      : BatchedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, softmax_lse, max_seqlen_q * h_q * d, max_seqlen_kv * h_kv * d, max_seqlen_q * h_q * d,
                      h_q * d, h_kv * d, h_q * d, d, d,
                      d, max_seqlen_q * h_q, p_dropout, softmax_scale, is_causal, is_bf16),
        dq_ptr(dq), dk_ptr(dk), dv_ptr(dv),
        dout_ptr(dout), dsoftmax_ptr(dsoftmax) {
    z_ptr = nullptr;

    Index dkv_batch_stride = max_seqlen_kv * h_q * d; // GQA
    Index dkv_seq_stride = h_q * d;
    Index dq_seq_stride = h_q * d;
    Index dout_seq_stride = h_q * d;
    Index dkv_head_stride = d;

    // TORCH_CHECK(dq_seq_stride == q_seq_stride);
    // TORCH_CHECK(dout_seq_stride == out_seq_stride);

    // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
    z_lengths = std::vector<Index>{b, h_q, max_seqlen_q, max_seqlen_kv};
    z_strides =
        std::vector<Index>{h_q * max_seqlen_q * max_seqlen_kv,
                           max_seqlen_q * max_seqlen_kv, max_seqlen_kv, 1};

    // MQA / GQA readiness
    // KGrad layout [b, max_seqlen_kv, h_q, d]
    dk_lengths = std::vector<Index>{b, h_q, max_seqlen_kv, d};
    dk_strides = std::vector<Index>{dkv_batch_stride, dkv_head_stride,
                                    dkv_seq_stride, 1};

    // VGrad layout [b, max_seqlen_kv, h_q, d]
    dv_lengths = std::vector<Index>{b, h_q, d, max_seqlen_kv};
    dv_strides = std::vector<Index>{dkv_batch_stride, dkv_head_stride, 1,
                                    dkv_seq_stride};
  }

  void *__restrict__ dq_ptr;
  void *__restrict__ dk_ptr;
  void *__restrict__ dv_ptr;

  void *__restrict__ dout_ptr;
  void *__restrict__ dsoftmax_ptr;

  // KGrad layout [b, max_seqlen_kv, h_q, d]
  std::vector<Index> dk_lengths;
  std::vector<Index> dk_strides;

  // VGrad layout [b, max_seqlen_kv, h_q, d]
  std::vector<Index> dv_lengths;
  std::vector<Index> dv_strides;
};

struct FlashInferBatchedParams : public BatchedParams {
  explicit FlashInferBatchedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out, 
      const float softmax_scale, const bool is_causal, const bool is_bf16, 
      void * softmax_lse)
      : BatchedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, softmax_lse, max_seqlen_q * h_q * d, max_seqlen_kv * h_kv * d, max_seqlen_q * h_q * d,
                      d, d, d, max_seqlen_q * d, max_seqlen_kv * d,
                      max_seqlen_q * d, max_seqlen_q * h_q, 0.0, softmax_scale, is_causal, is_bf16) {
  }
};

// Common Grouped Arguments
struct GroupedParams : public BaseParams {
  explicit GroupedParams(const Index b, const Index max_seqlen_q,
                         const Index max_seqlen_kv, const Index h_q,
                         const Index h_kv, const Index d,
                         void * q, void * k,
                         void * v, void * out,
                         const void *cu_seqlens_q_d,
                         const void *cu_seqlens_kv_d,
                         void * softmax_lse, 
                         const Index qbatch_stride, const Index kvbatch_stride, const Index outbatch_stride, 
                         const Index qseq_stride, const Index kvseq_stride, const Index outseq_stride, 
                         const Index qhead_stride, const Index kvhead_stride, const Index outhead_stride, 
                         const Index softmaxlse_batch_stride,
                         const float p_dropout,
                         const float softmax_scale, const bool is_causal, const bool is_bf16)
      : BaseParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, qseq_stride, kvseq_stride, outseq_stride, qhead_stride, 
                  kvhead_stride, outhead_stride, softmaxlse_batch_stride, p_dropout, softmax_scale, is_causal, is_bf16),
        seqlens_q(
            get_host_seqlens(static_cast<const int *>(cu_seqlens_q_d), b)),
        seqlens_kv(
            get_host_seqlens(static_cast<const int *>(cu_seqlens_kv_d), b)) {
    char *q_ptr = reinterpret_cast<char *>(q);
    char *k_ptr = reinterpret_cast<char *>(k);
    char *v_ptr = reinterpret_cast<char *>(v);
    char *out_ptr = reinterpret_cast<char *>(out);
    char *softmax_lse_ptr = reinterpret_cast<char *>(softmax_lse);

    // TODO: move to GPU
    for (int i = 0; i < b; ++i) {
      int curr_q_batch_stride = seqlens_q[i] * q_seq_stride;
      int curr_kv_batch_stride = seqlens_kv[i] * kv_seq_stride;
      int curr_out_batch_stride = seqlens_q[i] * out_seq_stride;

      if (!is_mnko_padding) {
        if (!is_dropout && d > 32 && d <= 64) {
          is_mnko_padding =
              !((seqlens_q[i] % 128) == 0 && (seqlens_kv[i] % 256) == 0);
        } else {
          is_mnko_padding =
              !((seqlens_q[i] % 128) == 0 && (seqlens_kv[i] % 128) == 0);
        }
      }

      q_ptrs.push_back(reinterpret_cast<void *>(q_ptr));
      q_ptr += get_size_in_bytes_half(curr_q_batch_stride);

      k_ptrs.push_back(reinterpret_cast<void *>(k_ptr));
      k_ptr += get_size_in_bytes_half(curr_kv_batch_stride);

      v_ptrs.push_back(reinterpret_cast<void *>(v_ptr));
      v_ptr += get_size_in_bytes_half(curr_kv_batch_stride);

      out_ptrs.push_back(reinterpret_cast<void *>(out_ptr));
      out_ptr += get_size_in_bytes_half(curr_out_batch_stride);

      softmax_lse_ptrs.push_back(reinterpret_cast<void *>(softmax_lse_ptr));
      softmax_lse_ptr +=
          get_size_in_bytes_float(softmax_lse_batch_stride);

      // Q layout [b, max_seqlen_q, h_q, d]
      std::vector<Index> q_lengths{1, h_q, seqlens_q[i], d};
      std::vector<Index> q_strides{curr_q_batch_stride, q_head_stride,
                                   q_seq_stride, 1};

      // K layout [b, max_seqlen_kv, h_kv, d]
      std::vector<Index> k_lengths{1, h_kv, seqlens_kv[i], d};
      std::vector<Index> k_strides{curr_kv_batch_stride, kv_head_stride,
                                   kv_seq_stride, 1};

      // V layout [b, max_seqlen_kv, h_kv, d]
      std::vector<Index> v_lengths{1, h_kv, d, seqlens_kv[i]};
      std::vector<Index> v_strides{curr_kv_batch_stride, kv_head_stride, 1,
                                   kv_seq_stride};

      // Y layout [b, max_seqlen_q, h_q, d]
      std::vector<Index> out_lengths{1, h_q, seqlens_q[i], d};
      std::vector<Index> out_strides{curr_out_batch_stride, out_head_stride,
                                     out_seq_stride, 1};

      // LSE layout [b, h_q, max_seqlen_q]
      std::vector<Index> lse_lengths{1, h_q, seqlens_q[i]};
      std::vector<Index> lse_strides{h_q * seqlens_q[i], seqlens_q[i], 1};

      q_lengths_vec.push_back(q_lengths);
      q_strides_vec.push_back(q_strides);
      k_lengths_vec.push_back(k_lengths);
      k_strides_vec.push_back(k_strides);
      v_lengths_vec.push_back(v_lengths);
      v_strides_vec.push_back(v_strides);
      out_lengths_vec.push_back(out_lengths);
      out_strides_vec.push_back(out_strides);
      lse_lengths_vec.push_back(lse_lengths);
      lse_strides_vec.push_back(lse_strides);
    }
  }

  std::vector<int> seqlens_q;
  std::vector<int> seqlens_kv;

  std::vector<const void *> q_ptrs;
  std::vector<const void *> k_ptrs;
  std::vector<const void *> v_ptrs;
  std::vector<void *> z_ptrs;
  std::vector<void *> out_ptrs;
  std::vector<void *> softmax_lse_ptrs;

  std::vector<std::vector<Index>> q_lengths_vec;
  std::vector<std::vector<Index>> q_strides_vec;
  std::vector<std::vector<Index>> k_lengths_vec;
  std::vector<std::vector<Index>> k_strides_vec;
  std::vector<std::vector<Index>> v_lengths_vec;
  std::vector<std::vector<Index>> v_strides_vec;
  std::vector<std::vector<Index>> z_lengths_vec;
  std::vector<std::vector<Index>> z_strides_vec;
  std::vector<std::vector<Index>> out_lengths_vec;
  std::vector<std::vector<Index>> out_strides_vec;
  std::vector<std::vector<Index>> lse_lengths_vec;
  std::vector<std::vector<Index>> lse_strides_vec;
};

// Forward Grouped Arguments
struct FlashFwdGroupedParams : public GroupedParams {
  explicit FlashFwdGroupedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out,
      const void *cu_seqlens_q_d, const void *cu_seqlens_kv_d,
      void * z_vec, void * softmax_lse,
      const float p_dropout, const float softmax_scale, const bool is_causal,
      const bool return_softmax, const bool is_bf16)
      : GroupedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, cu_seqlens_q_d, cu_seqlens_kv_d, softmax_lse,
                      max_seqlen_q * h_q * d, max_seqlen_kv * h_kv * d, max_seqlen_q * h_q * d,
                      h_q * d, h_kv * d, h_q * d, d, d,
                      d, max_seqlen_q * h_q,
                      p_dropout, softmax_scale, is_causal, is_bf16) {
    for (int i = 0; i < b; ++i) {
      if (return_softmax) {
        if (is_bf16) {
          z_ptrs.push_back(nullptr);
        } else {
          z_ptrs.push_back(reinterpret_cast<void *>(reinterpret_cast<half *>(z_vec) + i * h_q * max_seqlen_q * max_seqlen_kv));
        }
        // z_vec.push_back(torch::empty({1, h_q, seqlens_q[i], seqlens_kv[i]},
        //                              opts.dtype(torch::kUInt8)));
        // z_ptrs.push_back(nullptr);
      } else {
        z_ptrs.push_back(nullptr);
      }

      // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
      std::vector<Index> z_lengths{1, h_q, seqlens_q[i], seqlens_kv[i]};
      std::vector<Index> z_strides{h_q * seqlens_q[i] * seqlens_kv[i],
                                   seqlens_q[i] * seqlens_kv[i], seqlens_kv[i],
                                   1};

      z_lengths_vec.push_back(z_lengths);
      z_strides_vec.push_back(z_strides);
    }
  }
};

// Backward Grouped Arguments
struct FlashBwdGroupedParams : public GroupedParams {
  explicit FlashBwdGroupedParams(
      const Index b, const Index max_seqlen_q, const Index max_seqlen_kv,
      const Index h_q, const Index h_kv, const Index d, void * q,
      void * k, void * v, void * out,
      void * dout, void * dq, void * dk,
      void * dv, const void *cu_seqlens_q_d,
      const void *cu_seqlens_kv_d, void * dsoftmax_vec,
      void * softmax_lse, const float p_dropout,
      const float softmax_scale, const bool is_causal, const bool is_bf16)
      : GroupedParams(b, max_seqlen_q, max_seqlen_kv, h_q, h_kv, d, q, k, v,
                      out, cu_seqlens_q_d, cu_seqlens_kv_d, softmax_lse,
                      max_seqlen_q * h_q * d, max_seqlen_kv * h_kv * d, max_seqlen_q * h_q * d,
                      h_q * d, h_kv * d, h_q * d, d, d,
                      d, max_seqlen_q * h_q,
                      p_dropout, softmax_scale, is_causal, is_bf16),
        bwd_out_ptrs(
            std::vector<const void *>(out_ptrs.begin(), out_ptrs.end())),
        bwd_softmax_lse_ptrs(std::vector<const void *>(
            softmax_lse_ptrs.begin(), softmax_lse_ptrs.end())) {
    // TODO: Use the actual type
    char *dq_ptr = reinterpret_cast<char *>(dq);
    char *dk_ptr = reinterpret_cast<char *>(dk);
    char *dv_ptr = reinterpret_cast<char *>(dv);
    char *dout_ptr = reinterpret_cast<char *>(dout);

    Index dq_seq_stride = h_q * d;
    Index dkv_seq_stride = h_q * d; // GQA
    Index dout_seq_stride = h_q * d;
    Index dq_head_stride = d;
    Index dkv_head_stride = d;
    Index dout_head_stride = d;

    // TORCH_CHECK(dq_seq_stride == q_seq_stride);
    // TORCH_CHECK(dout_seq_stride == out_seq_stride);

    // auto opts = q.options();
    for (int i = 0; i < b; ++i) {
      // TODO: reuse it in the foward on GPU
      int curr_dq_batch_stride = seqlens_q[i] * dq_seq_stride;
      int curr_dkv_batch_stride = seqlens_kv[i] * dkv_seq_stride;
      int curr_dout_batch_stride = seqlens_q[i] * dout_seq_stride;

      z_ptrs.push_back(nullptr);

      dq_ptrs.push_back(reinterpret_cast<void *>(dq_ptr));
      dq_ptr += get_size_in_bytes_half(curr_dq_batch_stride);

      dk_ptrs.push_back(reinterpret_cast<void *>(dk_ptr));
      dk_ptr += get_size_in_bytes_half(curr_dkv_batch_stride);

      dv_ptrs.push_back(reinterpret_cast<void *>(dv_ptr));
      dv_ptr += get_size_in_bytes_half(curr_dkv_batch_stride);

      dout_ptrs.push_back(reinterpret_cast<const void *>(dout_ptr));
      dout_ptr += get_size_in_bytes_half(curr_dout_batch_stride);

      // dsoftmax_vec.push_back(
      //     torch::empty({1, h_q, seqlens_q[i]}, opts.dtype(torch::kFloat32)));
      dsoftmax_ptrs.push_back(reinterpret_cast<void *>(reinterpret_cast<float *>(dsoftmax_vec) + i * h_q * max_seqlen_q));

      // Z layout [b, h_q, max_seqlen_q, max_seqlen_kv]
      std::vector<Index> z_lengths =
          std::vector<Index>{b, h_q, seqlens_q[i], seqlens_kv[i]};
      std::vector<Index> z_strides =
          std::vector<Index>{h_q * seqlens_q[i] * seqlens_kv[i],
                             seqlens_q[i] * seqlens_kv[i], seqlens_kv[i], 1};

      // MQA / GQA readiness
      // KGrad layout [b, max_seqlen_kv, h_q, d]
      std::vector<Index> dk_lengths{1, h_q, seqlens_kv[i], d};
      std::vector<Index> dk_strides{curr_dkv_batch_stride,
                                    dkv_head_stride, dkv_seq_stride, 1};

      // VGrad layout [b, max_seqlen_kv, h_q, d]
      std::vector<Index> dv_lengths{1, h_q, d, seqlens_kv[i]};
      std::vector<Index> dv_strides{curr_dkv_batch_stride,
                                    dkv_head_stride, 1, dkv_seq_stride};

      z_lengths_vec.push_back(z_lengths);
      z_strides_vec.push_back(z_strides);
      dk_lengths_vec.push_back(dk_lengths);
      dk_strides_vec.push_back(dk_strides);
      dv_lengths_vec.push_back(dv_lengths);
      dv_strides_vec.push_back(dv_strides);
    }
  }

  std::vector<void *> dq_ptrs;
  std::vector<void *> dk_ptrs;
  std::vector<void *> dv_ptrs;

  std::vector<const void *> bwd_out_ptrs;
  std::vector<const void *> bwd_softmax_lse_ptrs;

  std::vector<const void *> dout_ptrs;
  std::vector<void *> dsoftmax_ptrs;

  // MQA / GQA readiness
  std::vector<std::vector<Index>> dk_lengths_vec;
  std::vector<std::vector<Index>> dk_strides_vec;
  std::vector<std::vector<Index>> dv_lengths_vec;
  std::vector<std::vector<Index>> dv_strides_vec;

  bool return_softmax;
};
