/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    // num_head is num_head_q if not PackGQA, else num_head_k
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;  // Only used if Varlen and cu_seqlens == nullptr and seqused == nullptr
    int const seqlen_k, headdim, headdim_v, element_size;  // Used to calculate L2 swizzling
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    // int const* const num_m_blocks_ptr = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
};

// method / static vars needed for every scheduler
// overwrite some of the methods / vars in the derived class, if needed
class TileSchedulerBase {
public:
    static constexpr bool pipelining = false;
    static constexpr int stride = 1;
public:

    CUTLASS_DEVICE
    TileSchedulerBase() {}

    CUTLASS_DEVICE void producer_notify() const {}
    CUTLASS_DEVICE void consumer_notify() const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    constexpr uint32_t stage() const noexcept { return 0; }

    CUTLASS_DEVICE
    void
    init_consumer() const {}
};

///////////////////////////////////////////////////////////////////////////////

template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler: public TileSchedulerBase {
public:
    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        int const num_blocks, num_head, num_batch, num_splits;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod nsplits_divmod;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr = nullptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // We use the top 16 bits to store num_splits
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
    }


    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int split_idx = 0;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        if constexpr (Split) {
            int split_idx;
            work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
            work_info.split_idx = split_idx;
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[work_info.bidb]
                : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
            if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            is_valid_tile = work_info.block_idx * kBlock < seqlen;
        }
        if constexpr (Varlen && Split) {
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
            // Use the top 16 bits to store num_splits
            work_info.split_idx |= (num_splits_dynamic << 16);
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }
};

///////////////////////////////////////////////////////////////////////////////

template<bool Split=false>
class StaticPersistentTileScheduler: public TileSchedulerBase {

public:
    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }


    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            int split_idx = 0;
            if constexpr (Split) {
                bidh = params.nsplits_divmod.divmod(split_idx, bidh);
            }
            return {block, bidh, bidb, split_idx};
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }
};

template<int NumConsumerThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=96, int Stride=1>
class PreemptivePersistentTileScheduler: public TileSchedulerBase {
    // **PPT** scheduler: performs correct synchronization for producer (generate_n_block) and consumer (KV load and computation pipeline)
    // This scheduler has the same coordinate computation logic as StaticPersistentTileSch, the difference is that
    // we employ a preemptive scheduling strategy based on a rough estimation of the workload for the consumer
    // In PPT, NumConsumerThreads is the total number of threads for (KV load and computation pipeline), and for FlashMask V2
    // it will be the #threads for (wg_id = 0, wp_id = 0) + (wg_id > 0, wp_id = *). The NumProducerThreads is simply 96 (hard-coded).
    static_assert(NumProducerThreads == 96, "PreemptivePersistentTileScheduler has incorrect producer thread num.");
    static constexpr int NumThreads = NumConsumerThreads + NumProducerThreads;
public:
    using SharedStorage = int;
    static constexpr int stride = Stride;
protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params

    struct Params {
        const int total_blocks;
        const cutlass::FastDivmod m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), 
                cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            return {block, bidh, bidb, 0};
        }

    };


    CUTLASS_DEVICE
    PreemptivePersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        // when all the blocks (SMs) done initializing and no SM has done the first task, tile_count_semaphore will be
        // at least `gridDim.x`, then, we just let prefetch_next_work and non-deterministic schedule (workload-related) take over 

        // For FlashMask V2, only generate_n_block pipeline is the big brother producer to be preemptively scheduled!
        // since the initial work is assigned deterministically via blockIdx.x, we need to ensure that the initial state of
        // tile_count_semaphore is gridDim.x. Can't use atomicAdd here, since if we do, for example, SM1 is really fast, it performs
        // prefetch_next_work even before SM2 calls get_initial_work, then SM1 will risk computing the same block as SM2.

        // for the initial work: assign deterministically
        return {int(blockIdx.x) * stride};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // this is a kick-off for the whole producer (producer waits for TileCountSmemEmpty), otherwise we will have a dead-lock, also
        // this init_consumer can only be called in consumer warps, otherwise we will have more arriving threads than needed
        // NumConsumerThreads: including (wg_id = 0, warp_id = 0: KV load) and (wg_id > 0, warp_id = *: computation) 
        flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        // only producer will call this method
        if (threadIdx.x == 96) {    // hard-coded, since n_block producer threads are in [32, 128)
            // the next job we are going to process: number of blocks currently done
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, stride);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // only threadIdx.x == 96 has the correct `current_work.tile_idx` (see prefetch next_work)
            // so there is no need to use shfl_sync to broadcast. Also shfl cannot broadcast across warps
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            if (threadIdx.x == 96) {    // hard-coded, since n_block producer threads are in [32, 128)
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            // Sync all the producers in case some of the producers return before the smem is updated
            flash::named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::NBlockProducer) /*id*/);
            return {*tile_count_smem};
        } else {
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            return {tile_idx};
        }
    }
};


template<int NumConsumerThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=128>
class BwdPreemptivePersistentTileScheduler: public TileSchedulerBase {
    static constexpr int NumThreads = NumConsumerThreads + NumProducerThreads;
public:
    using SharedStorage = int;
protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params

    struct Params {
        const int total_blocks;
        const cutlass::FastDivmod m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            return {block, bidh, bidb};
        }

    };

    CUTLASS_DEVICE
    BwdPreemptivePersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (!IsProducerWarp) {
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskSmemFull) /*id*/);
        }
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    producer_notify() const {     // notify the consumer that we've written data into the buffer
        flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskSmemFull) /*id*/);
    }

    CUTLASS_DEVICE
    void
    consumer_notify() const {
        // sync to make sure (*tile_count_smem) modification is visible to consumers
        flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskSmemEmpty) /*id*/);
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskSmemEmpty) /*id*/);
            // TODO(heqianyue): atomicAdd here?
            if (threadIdx.x == 0) {    // hard-coded, since n_block producer threads are in [32, 128)
                // the next job we are going to process: number of currently blocks done
                *tile_count_smem = atomicAdd(params.tile_count_semaphore, 1);
            }
            flash::named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskProducer) /*id*/);
        } else {
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(BwdNamedBarriers::FlashmaskSmemFull) /*id*/);
        }
        // how to make sure consumers can actually get this?
        return {*tile_count_smem};
    }
};


template<int NumConsumerThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=96>
class DualPreemptivePersistentTileExecutionScheduler: public TileSchedulerBase {
    // **PPT** scheduler: performs correct synchronization for producer (generate_n_block) and consumer (KV load and computation pipeline)
    // This scheduler has the same coordinate computation logic as StaticPersistentTileSch, the difference is that
    // we employ a preemptive scheduling strategy based on a rough estimation of the workload for the consumer
    // In PPT, NumConsumerThreads is the total number of threads for (KV load and computation pipeline), and for FlashMask V2
    // it will be the #threads for (wg_id = 0, wp_id = 0) + (wg_id > 0, wp_id = *). The NumProducerThreads is simply 96 (hard-coded).

    // The following static_assert is NOT compulsory, it's just that we found that 64 producer threads performs worse
    static_assert(NumProducerThreads == 96, "DualPPTX Scheduler has incorrect producer thread num.");
    static constexpr int NumThreads = NumConsumerThreads + NumProducerThreads;
public:
    using SharedStorage = int;
    static constexpr bool pipelining = true;        // DualPPTX has coarse-grained pipelining
protected:
    SharedStorage* const tile_count_smem;
    uint32_t sch_stage_;
public:
    // Device side kernel params

    struct Params {
        const int total_blocks;
        const cutlass::FastDivmod m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), 
                cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            return {block, bidh, bidb, 0};
        }

    };

    CUTLASS_DEVICE
    DualPreemptivePersistentTileExecutionScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) {
        // when all the blocks (SMs) done initializing and no SM has done the first task, tile_count_semaphore will be
        // at least `gridDim.x`, then, we just let prefetch_next_work and non-deterministic schedule (workload-related) take over 

        // For FlashMask V2, only generate_n_block pipeline is the big brother producer to be preemptively scheduled!
        // since the initial work is assigned deterministically via blockIdx.x, we need to ensure that the initial state of
        // tile_count_semaphore is gridDim.x. Can't use atomicAdd here, since if we do, for example, SM1 is really fast, it performs
        // prefetch_next_work even before SM2 calls get_initial_work, then SM1 will risk computing the same block as SM2.

        // for the initial work: assign deterministically
        if constexpr (IsProducerWarp) {
            sch_stage_ = 0;  // producer initial state is 0, since the first get_next, producer should sync full-1 (dual)
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
        } else {
            sch_stage_ = 1;  // consumer initial state is 1, since the first get_next, producer should sync empty-0 (non-dual)
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFullDual) /*id*/);
        }
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        // PPTX prefetch is moved to consumer for more exact delay scheduling
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) {
        // change state immediately, since we are to get next work
        // Note that for the return value: except from the initial work, PPT always dynamic schedules
        // Dual PPTX will have static schedule for only twice: get initial work and the first time get_next_work
        // This is intentional, since in the first get_next_work, smem is not fully ready.
        if constexpr (IsProducerWarp) {
            sch_stage_ = 0x1 ^ sch_stage_;
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) + (sch_stage_ << 1) /*id*/);
            int tile_idx = tile_count_smem[sch_stage_];
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) + (sch_stage_ << 1) /*id*/);
            // Sync all the producers in case some of the producers return before the smem is updated
            return {tile_idx >= 0 ? tile_idx : int(blockIdx.x + gridDim.x)};
        } else {
            // for example: 
            // the 1st get_next_work of consumer: load from 1, and atomicAdd store to 0 
            //      load from 1 not initialized, use blockIdx.x + gridDim.x (static scheduling)
            // the 2nd get_next_work of consumer: load from 0, and atomicAdd store to 1
            //      load from 0 initialized: the 3rd consumer work ID is correctly set 
            int tile_idx = tile_count_smem[sch_stage_];
            sch_stage_ = 0x1 ^ sch_stage_;
            if (threadIdx.x == NumConsumerThreads) {    // thread 288 hard-coded, since n_block consumer threads are in [128, 384)
                tile_count_smem[sch_stage_] = atomicAdd(params.tile_count_semaphore, 1);
            }
            flash::named_barrier_sync(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) + (sch_stage_ << 1) /*id*/);
            flash::named_barrier_arrive(NumThreads, static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) + (sch_stage_ << 1) /*id*/);
            return {tile_idx >= 0 ? tile_idx : int(blockIdx.x + gridDim.x)};
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    uint32_t stage() const noexcept {
        // Returns stage offset: sch_stage_ * 2. Producer always returns the current stage, 
        // while consumer returns 1 - current stage, so that consumer can always have valid input
        if constexpr (IsProducerWarp)
            return sch_stage_ << 1;
        else
            return (0x1 ^ sch_stage_) << 1;
    }
};

} // flash
