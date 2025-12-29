#pragma once
#include <memory>
#include "sr_buffer.cuh"
#include "cutlass/bfloat16.h"

namespace flashmask {

/**
 * SM-level overlapping communicator
 * 
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 * 
 * Call the constructor of this communicator, and call `run_overlap_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel
 * Then you should be able to get async remote get, costing only 4 SMs.
 * 
 * TODO(heqianyue): Can we create this class only once? So that the resource management can be amortized
 * currently, we choose to use `static` local variable due to its simplicity, but this could be risky and resource-consuming
 * 
 * TODO(heqianyue): Code-style --- make this class move-only explicitly. Currently this class is only implicitly move-only.
*/
template <typename KVType>
class OverlapCommunicator {
public:
    OverlapCommunicator(
        const KVType* const k_data,
        const KVType* const v_data,
        int b_kv,
        int s_kv,
        int h_kv,
        int d_kv,
        int rank,
        int nranks,
        int cp_size,
        const uint8_t* unique_id_ptr = nullptr
    );

    ~OverlapCommunicator();

    /**
     * run the overlap kernel asynchronously
     * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
     *      After the execution of this function, S will be set to `S * cp_size`
     * TODO(heqianyue): extend to more mask types!
    */
    void run_overlap_kernel(
        const int* const lt_start_ptr,
        const int* const ut_end_ptr,
        int* const write_ptr,
        int& S,
        const bool fwd = true
    );

    void wait_wptr_init();

    void* k_data() const { return kv_buffer->k_data(); }
    void* v_data() const { return kv_buffer->v_data(); }

    void update_kv_buffer(
        const KVType* const new_k_data,
        const KVType* const new_v_data,
        const bool fwd = true
    );

    // in `USE_SEMAPHORES` mode, call this function before calling `updayte_kv_buffer`
    // to make sure other PEs have finished reading the local KV data in our SR buffer
    // also, barriers the remote_get `comm_kernel`
    void wait_sr_buffer_empty();

    int cp_size() const {
        return _cp_size;
    }

    cudaEvent_t wptr_init;
private:
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;
    cudaStream_t comm_stream;
    const int B;
    const int S_local;
    const int H;
    const int D;
    const int _cp_size;
    int _my_pe;
    int _total_n_pes;
    int _cp_stride;
    size_t _cp_chunk_size;
    size_t _total_numel;

    int* block_work_ids;

    static constexpr int num_warps = 8;
    static constexpr int num_blocks = 32;   // 32 reg, 256 thread, one SM of H800 can hold 8 blocks, we use 4 SM

    // returns the team and stride between teams
    static nvshmem_team_t simple_collective_topology_setter(int my_global_pe, int stride, int n_pes);
};

namespace comm {

// OverlapCommunicator instance is managed via this singleton, therefore
// the instance is accessible to both fwd and bwd passes

// init and get instance (mutable ref), used in fwd
void init_singleton_instance(
    const cutlass::bfloat16_t* const k_data,
    const cutlass::bfloat16_t* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    int cp_size,
    const uint8_t* unique_id_ptr
);

// get instance (mutable ref), make sure the instance is initialized, used in both fwd and bwd
OverlapCommunicator<cutlass::bfloat16_t>& singleton();

// check whether the singleton unique_ptr is nullptr
bool is_singleton_null();

}   // namespace comm

}   // namespace flashmask
