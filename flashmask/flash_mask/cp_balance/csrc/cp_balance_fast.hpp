#pragma once
// CpBalanceSolver — Production-ready CP load balancing solver
//
// Solves: N items with integer weights -> M workers (K = N/M items each),
//         minimize the maximum worker load.
//
// Constraints: M divides N, N <= 512, M <= 32, K <= 16.
//
// Usage:
//   // C++ with vector
//   auto result = CpBalanceSolver::solve({100, 80, 60, 40, 30, 20, 15, 10}, 2);
//   // result.max_load, result.assign[worker] = {item indices...}
//
//   // Raw pointer (for Paddle/Python C extension, zero-copy to tensor)
//   int out[M * K];
//   int max_load = CpBalanceSolver::solve_to(weights.data(), N, M, out);
//
// Paddle custom op example:
//
//   #include "paddle/extension.h"
//   #include "cp_balance_fast.hpp"
//
//   std::vector<paddle::Tensor> CpBalanceOp(const paddle::Tensor& weights, int64_t M) {
//       int N = weights.shape()[0], K = N / M;
//       auto assign = paddle::empty({M, K}, paddle::DataType::INT32, weights.place());
//       int max_load = CpBalanceSolver::solve_to(
//           weights.data<int>(), N, M, assign.data<int>());
//       return {assign, paddle::full({1}, max_load, paddle::DataType::INT32)};
//   }
//
//   PD_BUILD_OP(cp_balance)
//       .Inputs({"Weights"}).Attrs({"M: int"})
//       .Outputs({"Assign", "MaxLoad"})
//       .SetKernelFn(PD_KERNEL(CpBalanceOp));

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

class CpBalanceSolver {
public:
    // ===== Limits =====
    static constexpr int kMaxN  = 512;
    static constexpr int kMaxM  = 32;
    static constexpr int kMaxK  = kMaxN / 2;
    static constexpr int kMax2K = 34;   // 2-way subproblem max items
    static constexpr int kMax3K = 50;   // 3-way subproblem max items

    // ===== Result (general C++ API) =====
    struct Result {
        int max_load = 0;
        std::vector<std::vector<int>> assign;  // assign[worker] = item indices
    };

    // Solve and return Result with vector-based assignment.
    [[nodiscard]] static Result solve(const std::vector<int>& weights, int M) {
        int N = static_cast<int>(weights.size());
        Result r;
        if (N <= 0 || M <= 0 || N % M != 0) return r;
        int K = N / M;

        // Run core solver into stack arrays
        std::array<std::array<int, kMaxK>, kMaxM> assign{};
        std::array<int, kMaxM> count{}, load{};
        int max_load = solve_core(weights.data(), N, M, K, assign, count, load);

        // Build result
        r.max_load = max_load;
        r.assign.resize(M);
        for (int j = 0; j < M; j++) {
            r.assign[j].assign(assign[j].begin(), assign[j].begin() + count[j]);
        }
        return r;
    }

    // Solve and write assignment directly to a flat buffer.
    // out_assign: pre-allocated buffer of size [M * K], row-major.
    //   out_assign[j * K + t] = item index for worker j, slot t.
    // Returns max_load (0 if input invalid).
    [[nodiscard]] static int solve_to(const int* weights, int N, int M, int* out_assign) {
        if (N <= 0 || M <= 0 || N % M != 0) return 0;
        int K = N / M;

        std::array<std::array<int, kMaxK>, kMaxM> assign{};
        std::array<int, kMaxM> count{}, load{};
        int max_load = solve_core(weights, N, M, K, assign, count, load);

        // Copy to flat output
        for (int j = 0; j < M; j++) {
            std::copy_n(assign[j].begin(), K, out_assign + j * K);
        }
        return max_load;
    }

private:
    // ===== Internal sub-solver result types =====
    struct Part2 {
        int max_load;
        std::array<bool, kMax2K> in_group0{};
    };

    struct Part3 {
        int max_load;
        std::array<int, kMax3K> group{};
    };

    // ===== Deterministic sort comparator: weight desc, index asc =====
    template <typename W>
    static auto desc_weight_asc_index(const W& w) {
        return [&](int a, int b) {
            return w[a] > w[b] || (w[a] == w[b] && a < b);
        };
    }

    // ===== Meet-in-the-Middle 2-way solver (K <= 10) =====
    static Part2 mitm_2way(const int* items, int n_items) {
        int K = n_items / 2;
        int total = 0;
        for (int i = 0; i < n_items; i++) total += items[i];

        int size_a = K, size_b = n_items - K;

        // Enumerate B-half subsets, bucket by popcount, sort by sum
        struct SubsetInfo { int sum; uint16_t mask; };
        static thread_local std::vector<SubsetInfo> buckets[17];
        for (int i = 0; i <= size_b; i++) buckets[i].clear();

        for (int mask = 0; mask < (1 << size_b); mask++) {
            int sum = 0;
            for (int i = 0; i < size_b; i++) {
                if (mask & (1 << i)) sum += items[K + i];
            }
            buckets[__builtin_popcount(mask)].push_back({sum, static_cast<uint16_t>(mask)});
        }
        for (int i = 0; i <= size_b; i++) {
            std::sort(buckets[i].begin(), buckets[i].end(),
                      [](const auto& a, const auto& b) { return a.sum < b.sum; });
        }

        // Enumerate A-half subsets, binary-search B for best match
        int best_max = total;
        uint32_t best_mask_a = 0;
        uint16_t best_mask_b = 0;

        for (int mask_a = 0; mask_a < (1 << size_a); mask_a++) {
            int count_a = __builtin_popcount(mask_a);
            int sum_a = 0;
            for (int i = 0; i < size_a; i++) {
                if (mask_a & (1 << i)) sum_a += items[i];
            }

            int need = K - count_a;
            if (need < 0 || need > size_b) continue;
            auto& bucket = buckets[need];
            if (bucket.empty()) continue;

            int want = (total + 1) / 2 - sum_a;
            int lo = 0, hi = static_cast<int>(bucket.size());
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (bucket[mid].sum < want) lo = mid + 1;
                else hi = mid;
            }

            for (int delta : {-1, 0}) {
                int idx = lo + delta;
                if (idx < 0 || idx >= static_cast<int>(bucket.size())) continue;
                int g0_sum = sum_a + bucket[idx].sum;
                int cur_max = std::max(g0_sum, total - g0_sum);
                if (cur_max < best_max) {
                    best_max = cur_max;
                    best_mask_a = mask_a;
                    best_mask_b = bucket[idx].mask;
                }
            }
        }

        Part2 result;
        result.max_load = best_max;
        for (int i = 0; i < size_a; i++) {
            if (best_mask_a & (1 << i)) result.in_group0[i] = true;
        }
        for (int i = 0; i < size_b; i++) {
            if (best_mask_b & (1 << i)) result.in_group0[K + i] = true;
        }
        return result;
    }

    // ===== BnB 2-way solver (K > 10): binary search + DFS =====
    static Part2 bnb_2way(const int* items, int n_items) {
        int K = n_items / 2;
        int total = 0;
        for (int i = 0; i < n_items; i++) total += items[i];
        int lower_bound = std::max((total + 1) / 2, items[0]);

        // Sort items descending (large first → prune earlier)
        int ord[kMax2K], sorted_w[kMax2K], suffix_sum[kMax2K + 1];
        std::iota(ord, ord + n_items, 0);
        std::sort(ord, ord + n_items, desc_weight_asc_index(items));
        for (int i = 0; i < n_items; i++) sorted_w[i] = items[ord[i]];
        suffix_sum[n_items] = 0;
        for (int i = n_items - 1; i >= 0; i--) {
            suffix_sum[i] = suffix_sum[i + 1] + sorted_w[i];
        }

        // LPT upper bound
        int lpt_load[2] = {}, lpt_count[2] = {}, lpt_assign[kMax2K];
        for (int i = 0; i < n_items; i++) {
            int g = (lpt_count[0] >= K) ? 1
                  : (lpt_count[1] >= K) ? 0
                  : (lpt_load[0] <= lpt_load[1]) ? 0 : 1;
            lpt_load[g] += sorted_w[i];
            lpt_count[g]++;
            lpt_assign[i] = g;
        }
        int upper_bound = std::max(lpt_load[0], lpt_load[1]);

        Part2 best;
        best.max_load = upper_bound;
        for (int i = 0; i < n_items; i++) {
            if (lpt_assign[i] == 0) best.in_group0[ord[i]] = true;
        }
        if (lower_bound == upper_bound) return best;

        // BnB state
        int load[2], count[2], assign[kMax2K];
        int64_t node_count;
        bool timed_out;
        static constexpr int64_t kNodeLimit = 500000;
        int target;

        auto search = [&](auto& self, int pos) -> bool {
            if (++node_count > kNodeLimit) { timed_out = true; return false; }
            if (pos == n_items) return true;

            int cur_w = sorted_w[pos];
            int total_cap = 0, n_open = 0, last_open = -1;
            for (int j = 0; j < 2; j++) {
                if (count[j] >= K) continue;
                total_cap += target - load[j];
                n_open++;
                last_open = j;
            }

            // Suffix-sum pruning
            if (suffix_sum[pos] > total_cap) return false;

            // Single-group forced
            if (n_open == 1) {
                if (n_items - pos != K - count[last_open]) return false;
                if (suffix_sum[pos] + load[last_open] > target) return false;
                for (int p = pos; p < n_items; p++) assign[p] = last_open;
                return true;
            }

            // Equal-value pruning
            int start_group = 0;
            if (pos > 0 && sorted_w[pos] == sorted_w[pos - 1]) {
                start_group = assign[pos - 1];
            }

            for (int j = start_group; j < 2; j++) {
                if (count[j] >= K || load[j] + cur_w > target) continue;
                // Symmetric pruning
                if (j == 1 && load[0] == load[1] && count[0] == count[1]) continue;

                load[j] += cur_w; count[j]++; assign[pos] = j;
                if (self(self, pos + 1)) return true;
                count[j]--; load[j] -= cur_w;
            }
            return false;
        };

        // Binary search for minimum feasible target
        for (int lo = lower_bound, hi = upper_bound - 1; lo <= hi;) {
            int mid = (lo + hi) / 2;
            target = mid;
            std::memset(load, 0, sizeof(load));
            std::memset(count, 0, sizeof(count));
            node_count = 0; timed_out = false;

            if (search(search, 0)) {
                best.max_load = mid;
                best.in_group0 = {};
                for (int i = 0; i < n_items; i++) {
                    if (assign[i] == 0) best.in_group0[ord[i]] = true;
                }
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return best;
    }

    // Dispatch: MITM for K <= 10, BnB for K > 10
    static Part2 partition_2way(const int* items, int n_items) {
        return (n_items / 2 <= 10) ? mitm_2way(items, n_items) : bnb_2way(items, n_items);
    }

    // ===== Insertion sort for small arrays (faster than std::sort for N <= 24) =====
    static void isort_desc(int* ord, int n, const int* w) {
        for (int i = 1; i < n; i++) {
            int key = ord[i], key_w = w[key];
            int j = i - 1;
            while (j >= 0 && (w[ord[j]] < key_w || (w[ord[j]] == key_w && ord[j] > key))) {
                ord[j + 1] = ord[j]; j--;
            }
            ord[j + 1] = key;
        }
    }

    // ===== BnB 3-way target solver: "can 3K items → 3 groups of K, each ≤ target?" =====
    static Part3 bnb_3way_target(const int* weights, int n_items, int target) {
        int K = n_items / 3;
        int total = 0;
        for (int i = 0; i < n_items; i++) total += weights[i];

        int ord[kMax3K], sorted_w[kMax3K], suffix_sum[kMax3K + 1];
        std::iota(ord, ord + n_items, 0);
        isort_desc(ord, n_items, weights);
        for (int i = 0; i < n_items; i++) sorted_w[i] = weights[ord[i]];
        suffix_sum[n_items] = 0;
        for (int i = n_items - 1; i >= 0; i--) {
            suffix_sum[i] = suffix_sum[i + 1] + sorted_w[i];
        }

        int lower_bound = std::max({(total + 2) / 3, sorted_w[0]});
        Part3 result;
        result.max_load = target + 1;  // infeasible by default
        if (target < lower_bound) return result;

        int load[3] = {}, count[3] = {}, assign[kMax3K];
        int64_t node_count = 0;
        bool timed_out = false;
        static constexpr int64_t kNodeLimit = 2000000;

        auto search = [&](auto& self, int pos) -> bool {
            if (++node_count > kNodeLimit) { timed_out = true; return false; }
            if (pos == n_items) return true;

            int remaining = n_items - pos;
            int cur_w = sorted_w[pos];
            int total_cap = 0, n_open = 0, last_open = -1;

            for (int j = 0; j < 3; j++) {
                if (count[j] >= K) continue;
                int need = K - count[j];
                int tail_start = n_items - need;
                // Tail pruning
                if (tail_start < pos) {
                    if (remaining < need) return false;
                    if (suffix_sum[pos] + load[j] > target) return false;
                } else {
                    if (suffix_sum[tail_start] + load[j] > target) return false;
                }
                total_cap += target - load[j];
                n_open++;
                last_open = j;
            }

            // Suffix-sum pruning
            if (suffix_sum[pos] > total_cap) return false;

            // Single-group forced
            if (n_open == 1) {
                if (remaining != K - count[last_open]) return false;
                if (suffix_sum[pos] + load[last_open] > target) return false;
                for (int p = pos; p < n_items; p++) assign[p] = last_open;
                return true;
            }

            // Equal-value pruning
            int start_group = 0;
            if (pos > 0 && sorted_w[pos] == sorted_w[pos - 1]) {
                start_group = assign[pos - 1];
            }

            // Symmetric pruning: skip duplicate (load, count) states
            int seen_load[3], seen_count[3], n_seen = 0;
            for (int j = start_group; j < 3; j++) {
                if (count[j] >= K || load[j] + cur_w > target) continue;

                bool dup = false;
                for (int s = 0; s < n_seen; s++) {
                    if (seen_load[s] == load[j] && seen_count[s] == count[j]) { dup = true; break; }
                }
                if (dup) continue;
                seen_load[n_seen] = load[j];
                seen_count[n_seen] = count[j];
                n_seen++;

                load[j] += cur_w; count[j]++; assign[pos] = j;
                if (self(self, pos + 1)) return true;
                count[j]--; load[j] -= cur_w;
            }
            return false;
        };

        if (search(search, 0)) {
            int gl[3] = {};
            for (int p = 0; p < n_items; p++) gl[assign[p]] += sorted_w[p];
            result.max_load = std::max({gl[0], gl[1], gl[2]});
            for (int i = 0; i < n_items; i++) result.group[ord[i]] = assign[i];
        }
        return result;
    }

    // ===== Core solver: writes into caller-provided stack arrays =====
    static int solve_core(
        const int* w, int N, int M, int K,
        std::array<std::array<int, kMaxK>, kMaxM>& assign,
        std::array<int, kMaxM>& count,
        std::array<int, kMaxM>& load)
    {
        // LPT initialization: sort descending, assign to lightest worker
        int ord[kMaxN];
        std::iota(ord, ord + N, 0);
        std::sort(ord, ord + N, desc_weight_asc_index(w));

        for (int i = 0; i < N; i++) {
            int idx = ord[i];
            int lightest = -1;
            for (int j = 0; j < M; j++) {
                if (count[j] < K && (lightest < 0 || load[j] < load[lightest])) {
                    lightest = j;
                }
            }
            assign[lightest][count[lightest]++] = idx;
            load[lightest] += w[idx];
        }

        // Skip optimization if K > 16 (subproblems too expensive)
        if (K > 16) {
            return *std::max_element(load.begin(), load.begin() + M);
        }

        // Lower bound: no solution can beat this
        int total = 0;
        for (int i = 0; i < N; i++) total += w[i];
        int lower_bound = std::max((total + M - 1) / M, N > 0 ? w[ord[0]] : 0);

        int max_rounds = (K <= 10) ? 200 : ((K <= 14) ? 50 : 20);

        for (int round = 0; round < max_rounds; round++) {
            bool any_improved = false;

            // ---- Phase 2: Pairwise exact repartition ----
            for (bool improved = true; improved;) {
                improved = false;

                // Find heaviest worker
                int cur_max = 0, heavy = 0;
                for (int j = 0; j < M; j++) {
                    if (load[j] > cur_max) { cur_max = load[j]; heavy = j; }
                }
                if (cur_max <= lower_bound) break;

                // Precompute top-1/top-2 load among non-heavy workers
                int omax1 = 0, omax1_idx = -1, omax2 = 0;
                for (int j = 0; j < M; j++) {
                    if (j == heavy) continue;
                    if (load[j] > omax1) { omax2 = omax1; omax1 = load[j]; omax1_idx = j; }
                    else if (load[j] > omax2) omax2 = load[j];
                }

                // Try each partner, pick the best
                int best_partner = -1, best_new_max = cur_max;
                Part2 best_split;
                for (int j = 0; j < M; j++) {
                    if (j == heavy) continue;

                    int pw[kMax2K];
                    int p = 0;
                    for (int t = 0; t < K; t++) pw[p++] = w[assign[heavy][t]];
                    for (int t = 0; t < K; t++) pw[p++] = w[assign[j][t]];

                    auto split = partition_2way(pw, 2 * K);
                    int omax = (j == omax1_idx) ? omax2 : omax1;
                    int new_max = std::max(omax, split.max_load);
                    if (new_max < best_new_max) {
                        best_new_max = new_max;
                        best_partner = j;
                        best_split = split;
                    }
                }

                // Apply best split
                if (best_partner >= 0) {
                    int indices[kMax2K];
                    int p = 0;
                    for (int t = 0; t < K; t++) indices[p++] = assign[heavy][t];
                    for (int t = 0; t < K; t++) indices[p++] = assign[best_partner][t];

                    count[heavy] = count[best_partner] = 0;
                    load[heavy] = load[best_partner] = 0;
                    for (int t = 0; t < 2 * K; t++) {
                        int wk = best_split.in_group0[t] ? heavy : best_partner;
                        assign[wk][count[wk]++] = indices[t];
                        load[wk] += w[indices[t]];
                    }
                    improved = true;
                    any_improved = true;
                }
            }

            // ---- Phase 3: Triple exact repartition (K <= 8 only) ----
            if (M >= 3 && K <= 8) {
                int cur_max = 0, heavy = 0;
                for (int j = 0; j < M; j++) {
                    if (load[j] > cur_max) { cur_max = load[j]; heavy = j; }
                }
                if (cur_max <= lower_bound) break;

                // Sort partners by load ascending (try lightest first)
                int partners[kMaxM], np = 0;
                for (int j = 0; j < M; j++) {
                    if (j != heavy) partners[np++] = j;
                }
                std::sort(partners, partners + np, [&](int a, int b) {
                    return load[a] < load[b] || (load[a] == load[b] && a < b);
                });

                bool improved = false;
                for (int pi = 0; pi < np && !improved; pi++) {
                    int p1 = partners[pi];
                    for (int pk = pi + 1; pk < np && !improved; pk++) {
                        int p2 = partners[pk];

                        // Bystander pruning
                        int omax = 0;
                        for (int q = np - 1; q >= 0; q--) {
                            if (q != pi && q != pk) { omax = load[partners[q]]; break; }
                        }
                        if (omax >= cur_max) continue;

                        // Lower-bound pruning
                        int tri_sum = load[heavy] + load[p1] + load[p2];
                        int lw_max = 0;
                        for (int t = 0; t < K; t++) {
                            lw_max = std::max(lw_max, w[assign[heavy][t]]);
                            lw_max = std::max(lw_max, w[assign[p1][t]]);
                            lw_max = std::max(lw_max, w[assign[p2][t]]);
                        }
                        if (std::max(omax, std::max((tri_sum + 2) / 3, lw_max)) >= cur_max) continue;

                        // Solve 3-way feasibility
                        int tw[kMax3K], ti[kMax3K];
                        int p = 0;
                        for (int t = 0; t < K; t++) { ti[p] = assign[heavy][t]; tw[p] = w[ti[p]]; p++; }
                        for (int t = 0; t < K; t++) { ti[p] = assign[p1][t];    tw[p] = w[ti[p]]; p++; }
                        for (int t = 0; t < K; t++) { ti[p] = assign[p2][t];    tw[p] = w[ti[p]]; p++; }

                        auto s3 = bnb_3way_target(tw, 3 * K, cur_max - 1);
                        if (std::max(omax, s3.max_load) < cur_max) {
                            int wmap[3] = {heavy, p1, p2};
                            count[heavy] = count[p1] = count[p2] = 0;
                            load[heavy] = load[p1] = load[p2] = 0;
                            for (int t = 0; t < 3 * K; t++) {
                                int wk = wmap[s3.group[t]];
                                assign[wk][count[wk]++] = ti[t];
                                load[wk] += w[ti[t]];
                            }
                            improved = true;
                            any_improved = true;
                        }
                    }
                }
            }

            if (!any_improved) break;
        }

        return *std::max_element(load.begin(), load.begin() + M);
    }
};
