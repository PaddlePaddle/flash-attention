/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 *
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// cutlass::arch::NamedBarrier::sync/arrive are only enabled Sm90 even though they work
// for Sm80 as well. We reimplement them here, enabled for both Sm90 and Sm80.

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class FwdNamedBarriers {
    QueryEmpty = 0,
    WarpSchedulerWG1 = 1,
    WarpSchedulerWG2 = 2,
    WarpSchedulerWG3 = 3,
    FlashMaskNBlock = 4,        // AppendKV is not used in FlashMask V3
    QueryRotated = 5,
    PFull = 6,
    TileCountSmemEmpty = 7,
    TileCountSmemFull = 8,
    ProducerWG = 9,             // ProducerWG is only used in Transpose_V, so it is currently not used
    TileCountSmemEmptyDual = 9, // HACK: ProducerWG is useless in FlashMask currently, reuse
    // HACK: 10 + 6 --> we will simplicity use barrier 0 (syncthreads bar)
    // This is not safe to use! Since we do not know for sure whether bar.sync 16 is equivalent to bar.sync 0
    // Even if it is, bar.sync 0 might conflicts with __syncthreads, but if there is no __syncthreads
    // in the kernel, we will be fine (by far, nearly 20k tests passed)
    TileCountSmemFullDual = 10,
    NBlockProducer = 4,         // HACK: NBlockProducer is only used in PPTScheduler
    PEmpty = 3,                 // HACK: PEmpty is only used when we don't have 3 WGs
};

enum class BwdNamedBarriers {
    KVEmpty = 0,
    PdS = 1,
    // This needs to match FwdNamedBarriers::TileCountSmemEmpty since TileScheduler uses it
    FlashmaskSmemEmpty = 2,
    FlashmaskSmemFull = 3,
    dQEmptyWG1 = 4,
    dQEmptyWG2 = 5,
    dQEmptyWG3 = 6,
    dQFullWG1 = 7,
    dQFullWG2 = 8,
    dQFullWG3 = 9,
    FlashmaskProducer = 0,  // HACK: KVEmpty sync is unused
};

} // flash
