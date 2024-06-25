// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/philox_rand.hpp"

namespace ck {

template <typename DataType, typename ThreadSliceDesc_M_K, ck::index_t DropoutSizePerThread=4> 
struct BlockwiseDropout
{
    static constexpr auto I0         = Number<0>{};
    static constexpr auto I1         = Number<1>{};
    static constexpr index_t MRepeat = ThreadSliceDesc_M_K{}.GetLength(I0);
    static constexpr index_t KRepeat = ThreadSliceDesc_M_K{}.GetLength(I1);

    template <typename CThreadBuffer, typename Offset, bool using_sign_bit = false>
    __host__ __device__ void ApplyDropoutAttnBwd(CThreadBuffer& in_thread_buf,
                                                 ck::philox& ph,
                                                 index_t element_global_1d_id,
                                                 index_t MRaw)
    {
        auto execute_dropout = [&](bool keep, DataType val) {
            if constexpr(using_sign_bit)
                return keep ? val : -val;
            else
                return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat;
        int philox_calls = tmp_size / DropoutSizePerThread;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            if constexpr(DropoutSizePerThread == 4) {
                ph.get_random_4x8((tmp + i * 4), element_global_1d_id + i * Offset{} * MRaw);
            }
            else if constexpr(DropoutSizePerThread == 16)
            {
                ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{} * MRaw);
            }
        }
        int tmp_index = 0;
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            in_thread_buf(iM) =
                execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(iM));
            tmp_index = tmp_index + 1;
        });
    }

    template <typename CThreadBuffer, typename Offset, bool using_sign_bit = false>
    __host__ __device__ void ApplyDropoutAttnBwdQ(CThreadBuffer& in_thread_buf,
                                                 ck::philox& ph,
                                                 index_t element_global_1d_id,
                                                 index_t MRaw)
    {
        auto execute_dropout = [&](bool keep, DataType val) {
            if constexpr(using_sign_bit)
                return keep ? val : -val;
            else
                return keep ? val * p_dropout_rescale : float(0);
        };
        int rowid=threadIdx.x%16/4;
        constexpr int tmp_size = MRepeat * KRepeat;
        int philox_calls = tmp_size / DropoutSizePerThread;
        int batchoff=element_global_1d_id%(MRaw*MRaw);
        int m=batchoff%MRaw;
        int n=batchoff/MRaw;
        element_global_1d_id = element_global_1d_id-batchoff+m * MRaw + n-rowid*4 ;
        float *p_thread_buf = &in_thread_buf(I0);
        for(int k=0;k<4;k++){
            for(int i = 0; i < 4; i++){
                uint8_t tmp2[4];
                ph.get_random_4x8(tmp2, element_global_1d_id+i*4*MRaw+k*16);
                p_thread_buf[k*4+i]=execute_dropout(tmp2[rowid]<= p_dropout_uint8_t,p_thread_buf[k*4+i]);
            }
        }
    }

    template <typename CThreadBuffer,
              typename ZThreadBuffer,
              typename Offset,
              bool using_sign_bit = false>
    __host__ __device__ void ApplyDropoutAttnBwdSaveZ(CThreadBuffer& in_thread_buf,
                                                      ck::philox& ph,
                                                      index_t element_global_1d_id,
                                                      ZThreadBuffer& z_thread_buf,
                                                      index_t MRaw)
    {

        auto execute_dropout = [&](bool keep, DataType val) {
            if constexpr(using_sign_bit)
                return keep ? val : -val;
            else
                return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat;

        int philox_calls = tmp_size / DropoutSizePerThread;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            if constexpr(DropoutSizePerThread == 4) {
                ph.get_random_4x8((tmp + i * 4), element_global_1d_id + i * Offset{} * MRaw);
            }
            else if constexpr(DropoutSizePerThread == 16)
            {
                ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{} * MRaw);
            }
        }

        block_sync_lds();

        int tmp_index = 0;
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadSliceDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) =
                    execute_dropout(tmp[tmp_index] <= p_dropout_uint8_t, in_thread_buf(offset));
                z_thread_buf(offset) = tmp[tmp_index];
                tmp_index            = tmp_index + 1;
            });
        });
    }

    template <typename CThreadBuffer, typename ZThreadBuffer, typename Step, typename Offset>
    __host__ __device__ void ApplyDropoutWithZ(CThreadBuffer& in_thread_buf,
                                               ZThreadBuffer& z_thread_buf)
    {

        auto execute_dropout = [&](bool keep, DataType val) {
            return keep ? val * p_dropout_rescale : float(0);
        };

        constexpr int tmp_size = MRepeat * KRepeat / Step{}.value;
        static_for<0, tmp_size, 1>{}([&](auto i) {
            in_thread_buf(i + Offset{}) =
                execute_dropout(z_thread_buf(i) <= p_dropout_uint8_t, in_thread_buf(i + Offset{}));
        });
    }

    // get raw z matrix with random number for shuffle
    template <typename ZThreadBuffer, typename Step, typename Offset>
    __host__ __device__ void GenerateZMatrixAttnFwd(ck::philox& ph,
                                                    index_t element_global_1d_id,
                                                    ZThreadBuffer& z_thread_buf)
    {
        constexpr int tmp_size = MRepeat * KRepeat / Step{}.value;

        int philox_calls = tmp_size / DropoutSizePerThread;

        uint8_t tmp[tmp_size];
        for(int i = 0; i < philox_calls; i++)
        {
            if constexpr(DropoutSizePerThread == 4) {
                ph.get_random_4x8((tmp + i * 4), element_global_1d_id + i * Offset{});
            }
            else if constexpr(DropoutSizePerThread == 16)
            {
                ph.get_random_16x8((tmp + i * 16), element_global_1d_id + i * Offset{});
            }
        }

        static_for<0, tmp_size, 1>{}([&](auto i) { z_thread_buf(i) = tmp[i.value]; });
    }

    uint8_t p_dropout_uint8_t;
    DataType p_dropout_rescale;
};

} // namespace ck
