#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace flashmask {

class UniqueIdFileSync {
public:
    // Used in Rank 0, write the NVSHMEM unique ID
    static std::vector<uint8_t> generate_and_write_unique_id(int rank, const std::string& file_path = "./nvshmem_unique_id.bin") {
        if (rank != 0) {
            throw std::runtime_error("generateAndWriteByRank0 should only be called by rank 0");
        }
        
        nvshmemx_uniqueid_t unique_id;
        nvshmemx_get_uniqueid(&unique_id);
        
        std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
        std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
        
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + file_path);
        }
        
        file.write(reinterpret_cast<const char*>(&unique_id), sizeof(unique_id));
        file.close();
        
        if (!file.good()) {
            throw std::runtime_error("Failed to write unique_id to file");
        }
        
        return result;
    }
    
    // Rank > 0: read from the local file
    static std::vector<uint8_t> wait_and_read_unique_id(int rank, const std::string& file_path = "./nvshmem_unique_id.bin") {
        if (rank == 0) {
            throw std::runtime_error("readAndWaitByOtherRanks should not be called by rank 0");
        }
        
        // max time span for NVSHMEM initialization (10s)
        const int max_attempts = 200;
        const int wait_ms = 50;
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            // check for existence
            if (std::filesystem::exists(file_path)) {
                // check whether the file has the correct size
                auto file_size = std::filesystem::file_size(file_path);
                if (file_size >= sizeof(nvshmemx_uniqueid_t)) {
                    std::ifstream file(file_path, std::ios::binary);
                    if (file) {
                        nvshmemx_uniqueid_t unique_id;
                        file.read(reinterpret_cast<char*>(&unique_id), sizeof(unique_id));
                        
                        if (file.gcount() == sizeof(unique_id)) {
                            std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
                            std::memcpy(result.data(), &unique_id, sizeof(unique_id));
                            return result;
                        }
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        }
        
        throw std::runtime_error("Timeout waiting for unique_id file from rank 0");
    }
    
    static void clean_up_file(const std::string& file_path = "./nvshmem_unique_id.bin") {
        if (std::filesystem::exists(file_path)) {
            try {
                std::filesystem::remove(file_path);
            } catch (...) {
                // ignore failure
            }
        }
    }
};

}   // namespace flashmask
