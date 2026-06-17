#pragma once
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>

namespace flashmask {

/**
 * Centralized runtime feature switches for the overlap communicator.
 *
 * Every switch is read once, at OverlapCommunicator construction time, from a
 * FLASHMASK_*-prefixed environment variable. The effective (post-fallback)
 * state is logged once via print().
 *
 * The whole struct — fields, parsing and logging — lives here as inline
 * definitions, so overlap_comm.cu just *uses* a flags instance and never has to
 * define switch-management code. overlap_comm.cuh only #includes this header to
 * embed the by-value member; the heavy attention launch templates that pull in
 * overlap_comm.cuh do not depend on anything here beyond the trivial layout.
 *
 * How to add a new switch
 *   1. Add a `bool` field below, with its default as the in-struct initializer.
 *   2. Parse it in from_env()  (one line).
 *   3. Log it  in print()       (one field in the format string).
 */
struct OverlapFeatureFlags {
    // FLASHMASK_USE_HIERARCHICAL: multi-node hierarchical AG/RS rank mapping.
    // Hierarchical communication defaults to false. Since using hierarchical communication
    // requires the code modification on Python end (context_parallel_utils.py)
    bool use_hierarchical = false;
    // FLASHMASK_USE_BHSD_LAYOUT: SR buffer uses (B,H,S,D) instead of (B,S,H,D).
    bool use_bhsd_layout = false;
    // FLASHMASK_PER_STAGE_BUFFER: one RS buffer slot per segment (vs. a single shared slot).
    bool per_stage_buffer = true;

    // Parse a boolean FLASHMASK_* env var. Unset/empty → `default_value`. Accepts
    // (case-insensitive) "1"/"true"/"on"/"yes" as true and "0"/"false"/"off"/"no"
    // as false; any other value falls back to `default_value` with a warning.
    static bool parse_bool_env(const char* name, bool default_value) {
        const char* raw = std::getenv(name);
        if (raw == nullptr || raw[0] == '\0') return default_value;

        std::string val(raw);
        for (char& c : val) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (val == "1" || val == "true" || val == "on" || val == "yes") return true;
        if (val == "0" || val == "false" || val == "off" || val == "no") return false;

        printf("[FlashMask Overlap] %s='%s' not recognized, using default %d.\n",
               name, raw, int(default_value));
        return default_value;
    }

    // Parse every switch from the environment, applying the defaults above when unset.
    static OverlapFeatureFlags from_env() {
        OverlapFeatureFlags f;   // in-struct initializers carry the defaults
        f.use_hierarchical = parse_bool_env("FLASHMASK_USE_HIERARCHICAL", f.use_hierarchical);
        f.use_bhsd_layout  = parse_bool_env("FLASHMASK_USE_BHSD_LAYOUT",  f.use_bhsd_layout);
        f.per_stage_buffer = parse_bool_env("FLASHMASK_PER_STAGE_BUFFER", f.per_stage_buffer);
        return f;
    }

    // Log the effective (post-fallback) switch state. Prints on rank 0 only.
    void print(int rank) const {
        if (rank != 0) return;
        printf("[FlashMask Overlap] Feature switches: "
               "FLASHMASK_USE_HIERARCHICAL=%d, FLASHMASK_USE_BHSD_LAYOUT=%d, FLASHMASK_PER_STAGE_BUFFER=%d\n",
               int(use_hierarchical), int(use_bhsd_layout), int(per_stage_buffer));
    }
};

}   // namespace flashmask
