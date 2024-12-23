// kernel_rabin_karp_optimized.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include "kseq/kseq.h"
#include "common.h"

// Define a large prime for hashing
#define PRIME 65536
#define BASE 256

// Adjust MAX_SIGNATURES based on your data (up to 1000)
#define MAX_SIGNATURES 1000

// Constant memory for signature hashes and highest powers
__constant__ unsigned long d_sig_hashes_const[MAX_SIGNATURES];
__constant__ unsigned long d_highest_powers_const[MAX_SIGNATURES];

// CUDA Kernel for precomputing signature hashes and highest powers
__global__ void precomputeSignatureHashes(
    const char* __restrict__ d_signatures,
    const int* __restrict__ d_sig_lengths,
    const size_t* __restrict__ d_sig_offsets,
    unsigned long* d_sig_hashes,
    unsigned long* d_highest_powers,
    int num_signatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_signatures) return;

    const char* signature = &d_signatures[d_sig_offsets[idx]];
    int len = d_sig_lengths[idx];

    unsigned long hash = 0;
    for (int j = 0; j < len; ++j) {
        // hash = (hash * BASE + signature[j]) % PRIME;
        hash = ((hash << 8) + signature[j]) & (PRIME - 1);
    }

    unsigned long highest_power = 1;
    for (int j = 0; j < len - 1; ++j) {
        highest_power = (highest_power * BASE) % PRIME;
    }

    d_sig_hashes[idx] = hash;
    d_highest_powers[idx] = highest_power;
}

// CUDA Kernel for Rabin-Karp matching signatures within samples using 2D thread blocks
__global__ void rabinKarpKernelOptimized(
    const char* __restrict__ d_samples,
    const int* __restrict__ d_sample_lengths,
    const size_t* __restrict__ d_sample_offsets,
    const char* __restrict__ d_signatures,
    const int* __restrict__ d_sig_lengths,
    const size_t* __restrict__ d_sig_offsets,
    const char* __restrict__ d_quals, // Phred+33 encoded quality scores
    float* d_match_scores,
    const unsigned long* __restrict__ d_sig_hashes,
    const unsigned long* __restrict__ d_highest_powers,
    int num_samples,
    int num_signatures
) {
    // Calculate the global thread index based on 2D block and grid dimensions
    int threads_per_block = blockDim.x * blockDim.y;
    int global_thread_id = blockIdx.x * threads_per_block + threadIdx.y * blockDim.x + threadIdx.x;

    if (global_thread_id >= (num_samples * num_signatures)) return;

    // Determine the corresponding sample and signature indices
    int sample_idx = global_thread_id / num_signatures;
    int signature_idx = global_thread_id % num_signatures;

    int sample_length = d_sample_lengths[sample_idx];
    int sig_length = d_sig_lengths[signature_idx];

    // Early exit if signature is longer than the sample
    if (sig_length > sample_length) {
        d_match_scores[global_thread_id] = 0.0f;
        return;
    }

    const char* sample = &d_samples[d_sample_offsets[sample_idx]];
    const char* signature = &d_signatures[d_sig_offsets[signature_idx]];
    const char* qual = &d_quals[d_sample_offsets[sample_idx]];

    unsigned long sig_hash = d_sig_hashes[signature_idx];
    unsigned long highest_power = d_highest_powers[signature_idx];

    // Calculate the initial hash for the first window of the sample
    unsigned long window_hash = 0;
    #pragma unroll
    for (int i = 0; i < sig_length; ++i) {
        window_hash = ((window_hash << 8) + __ldg(&sample[i])) & (PRIME-1);
    }

    bool match_found = false;
    float confidence_sum = 0.0f;
    int match_count = 0;

    // Slide the window over the sample
    for (int i = 0; i <= sample_length - sig_length; ++i) {
        // If the hash matches, verify the actual substring
        if (window_hash == sig_hash) {
            bool match = true;
            float current_confidence = 0.0f;
            #pragma unroll
            for (int j = 0; j < sig_length; j++) {
                char s = __ldg(&sample[i + j]);
                char sig_char = __ldg(&signature[j]);
                if (s != 'N' && sig_char != 'N' && s != sig_char) {
                    match = false;
                    break;
                }
                int phred = static_cast<int>(__ldg(&qual[i + j])) - 33;
                current_confidence += static_cast<float>(phred);
            }
            if (match) {
                match_found = true;
                confidence_sum += current_confidence;
                match_count += sig_length;
                break; // Only the first match is considered
            }
        }

        // Compute hash for the next window using rolling hash
        if (i < sample_length - sig_length) {
            window_hash = (window_hash + PRIME - (__ldg(&sample[i]) * highest_power) & (PRIME - 1)) & (PRIME - 1);
            window_hash = ((window_hash << 8) + __ldg(&sample[i + sig_length])) & (PRIME - 1);
        }
    }

    // Calculate average confidence score
    if (match_found && match_count > 0) {
        d_match_scores[global_thread_id] = confidence_sum / static_cast<float>(match_count);
    } else {
        d_match_scores[global_thread_id] = 0.0f; // No match
    }
}

// Host function to execute Rabin-Karp Matcher with optimized 2D thread blocks
void runMatcher(const std::vector<klibpp::KSeq>& samples,
               const std::vector<klibpp::KSeq>& signatures,
               std::vector<MatchResult>& matches) {

    int num_samples = samples.size();
    int num_signatures = signatures.size();

    if (num_samples == 0 || num_signatures == 0) {
        return;
    }

    // Number of samples and signatures
    int total_sample_length = 0;
    int total_sig_length = 0;

    std::vector<size_t> h_sample_offsets(num_samples);
    std::vector<int> h_sample_lengths(num_samples);
    size_t offset = 0;
    for (int i = 0; i < num_samples; ++i) {
        h_sample_lengths[i] = samples[i].seq.length();
        total_sample_length += samples[i].seq.length();
        h_sample_offsets[i] = offset;
        offset += samples[i].seq.length();
    }

    std::vector<int> h_sig_lengths(num_signatures);
    std::vector<size_t> h_sig_offsets(num_signatures);
    offset = 0;
    for (int i = 0; i < num_signatures; ++i) {
        h_sig_lengths[i] = signatures[i].seq.length();
        total_sig_length += signatures[i].seq.length();
        h_sig_offsets[i] = offset;
        offset += signatures[i].seq.length();
    }

    // Allocate host memory for samples, signatures, and qualities
    std::vector<char> h_samples_data(total_sample_length);
    std::vector<char> h_signatures_data(total_sig_length);
    std::vector<char> h_quals_data(total_sample_length);

    // Populate host memory
    for (int i = 0; i < num_samples; ++i) {
        memcpy(&h_samples_data[h_sample_offsets[i]], samples[i].seq.c_str(), h_sample_lengths[i]);
        memcpy(&h_quals_data[h_sample_offsets[i]], samples[i].qual.c_str(), h_sample_lengths[i]);
    }
    for (int i = 0; i < num_signatures; ++i) {
        memcpy(&h_signatures_data[h_sig_offsets[i]], signatures[i].seq.c_str(), h_sig_lengths[i]);
    }

    // Allocate device memory
    char* d_samples;
    char* d_signatures;
    char* d_quals;
    float* d_match_scores;

    int* d_sample_lengths;
    int* d_sig_lengths;

    size_t* d_sample_offsets;
    size_t* d_sig_offsets;

    unsigned long* d_sig_hashes;
    unsigned long* d_highest_powers;

    cudaMalloc(&d_samples, total_sample_length * sizeof(char));
    cudaMalloc(&d_signatures, total_sig_length * sizeof(char));
    cudaMalloc(&d_quals, total_sample_length * sizeof(char));
    cudaMalloc(&d_match_scores, num_samples * num_signatures * sizeof(float));

    cudaMalloc(&d_sample_lengths, num_samples * sizeof(int));
    cudaMalloc(&d_sig_lengths, num_signatures * sizeof(int));

    cudaMalloc(&d_sample_offsets, num_samples * sizeof(size_t));
    cudaMalloc(&d_sig_offsets, num_signatures * sizeof(size_t));

    cudaMalloc(&d_sig_hashes, num_signatures * sizeof(unsigned long));
    cudaMalloc(&d_highest_powers, num_signatures * sizeof(unsigned long));

    // Copy data from host to device
    cudaMemcpy(d_samples, h_samples_data.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, h_signatures_data.data(), total_sig_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quals, h_quals_data.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sample_lengths, h_sample_lengths.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_lengths, h_sig_lengths.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sample_offsets, h_sample_offsets.data(), num_samples * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_offsets, h_sig_offsets.data(), num_signatures * sizeof(size_t), cudaMemcpyHostToDevice);

    // Define kernel launch parameters for precomputing signature hashes
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_signatures + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the precompute signature hashes kernel
    precomputeSignatureHashes<<<blocksPerGrid, threadsPerBlock>>>(
        d_signatures, d_sig_lengths, d_sig_offsets, d_sig_hashes, d_highest_powers, num_signatures
    );

    // Copy signature hashes and highest powers to constant memory
    cudaMemcpyToSymbol(d_sig_hashes_const, d_sig_hashes, num_signatures * sizeof(unsigned long));
    cudaMemcpyToSymbol(d_highest_powers_const, d_highest_powers, num_signatures * sizeof(unsigned long));

    // Define kernel launch parameters with 2D thread blocks (16x16)
    dim3 threadsPerBlock2D(16, 16); // 256 threads per block
    int total_pairs = num_samples * num_signatures;
    int threads_per_block_2D = threadsPerBlock2D.x * threadsPerBlock2D.y;
    int blocksPerGrid2D = (total_pairs + threads_per_block_2D - 1) / threads_per_block_2D;

    // Initialize match scores to zero
    cudaMemset(d_match_scores, 0, num_samples * num_signatures * sizeof(float));

    // Launch the optimized Rabin-Karp kernel with 2D thread blocks
    rabinKarpKernelOptimized<<<blocksPerGrid2D, threadsPerBlock2D>>>(
        d_samples, d_sample_lengths, d_sample_offsets,
        d_signatures, d_sig_lengths, d_sig_offsets,
        d_quals, d_match_scores,
        d_sig_hashes, d_highest_powers,
        num_samples, num_signatures
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        // Handle error as needed (e.g., exit, cleanup)
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Allocate host memory for match scores
    std::vector<float> h_match_scores(num_samples * num_signatures);

    // Copy match scores back to host
    cudaMemcpy(h_match_scores.data(), d_match_scores, num_samples * num_signatures * sizeof(float), cudaMemcpyDeviceToHost);

    // Populate the matches vector
    matches.reserve(num_samples * num_signatures); // Reserve space to avoid reallocations
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_signatures; ++j) {
            float score = h_match_scores[i * num_signatures + j];
            if (score > 0.0f) { // Assuming a match is indicated by score > 0
                MatchResult result;
                result.sample_name = samples[i].name;
                result.signature_name = signatures[j].name;
                result.match_score = score;
                matches.push_back(result);
            }
        }
    }

    // Free device memory
    cudaFree(d_samples);
    cudaFree(d_signatures);
    cudaFree(d_quals);
    cudaFree(d_match_scores);

    cudaFree(d_sample_lengths);
    cudaFree(d_sig_lengths);

    cudaFree(d_sample_offsets);
    cudaFree(d_sig_offsets);

    cudaFree(d_sig_hashes);
    cudaFree(d_highest_powers);
}
