// optimized_kernel_multiPair.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include "kseq/kseq.h"
#include "common.h"

// Atomic max for float
__device__ float atomicMax_float(float* address, float val) {
    float old = *address, assumed;
    do {
        assumed = old;
        old = __int_as_float(atomicCAS((int*)address, __float_as_int(assumed),
                                       __float_as_int(fmaxf(val, assumed))));
    } while (assumed != old);
    return old;
}

__global__ void matchKernelMultiPair(const char* __restrict__ d_samples, const int* __restrict__ d_sample_lengths,
                                     const size_t* __restrict__ d_sample_offsets, const char* __restrict__ d_signatures,
                                     const int* __restrict__ d_sig_lengths, const size_t* __restrict__ d_sig_offsets,
                                     const char* __restrict__ d_quals, float* d_match_scores,
                                     int num_samples, int num_signatures) {
    // Each block handles multiple sample-signature pairs
    int block_pairs = blockDim.x; // Number of pairs per block
    int pair_idx = blockIdx.x * block_pairs + threadIdx.x;

    if (pair_idx >= num_samples * num_signatures) return;

    int sample_idx = pair_idx / num_signatures;
    int signature_idx = pair_idx % num_signatures;

    int sample_length = d_sample_lengths[sample_idx];
    int signature_length = d_sig_lengths[signature_idx];

    if (signature_length > sample_length) return;

    const char* sample = &d_samples[d_sample_offsets[sample_idx]];
    const char* signature = &d_signatures[d_sig_offsets[signature_idx]];
    const char* qual = &d_quals[d_sample_offsets[sample_idx]];

    int num_positions = sample_length - signature_length + 1;

    float thread_max = 0.0f;
    for (int i = threadIdx.y; i < num_positions; i += blockDim.y) {
        bool match = true;
        float confidence_sum = 0.0f;

        for (int j = 0; j < signature_length; ++j) {
            char s = __ldg(&sample[i + j]);
            char sig = __ldg(&signature[j]);

            if (s != 'N' && sig != 'N' && s != sig) {
                match = false;
                break;
            }

            int phred = __ldg(&qual[i + j]) - 33;
            confidence_sum += (float)phred;
        }

        if (match) {
            float avg_confidence = confidence_sum / (float)signature_length;
            if (avg_confidence > thread_max) {
                thread_max = avg_confidence;
            }
        }
    }

    // Update the global maximum using atomic operation
    atomicMax_float(&d_match_scores[pair_idx], thread_max);
}

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                                  const std::vector<klibpp::KSeq>& signatures,
                                  std::vector<MatchResult>& matches) {

    // Number of samples and signatures
    int num_samples = samples.size();
    int num_signatures = signatures.size();

    if (num_samples == 0 || num_signatures == 0) {
        return;
    }

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

    cudaMalloc(&d_samples, total_sample_length * sizeof(char));
    cudaMalloc(&d_signatures, total_sig_length * sizeof(char));
    cudaMalloc(&d_quals, total_sample_length * sizeof(char));
    cudaMalloc(&d_match_scores, num_samples * num_signatures * sizeof(float));

    cudaMalloc(&d_sample_lengths, num_samples * sizeof(int));
    cudaMalloc(&d_sig_lengths, num_signatures * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_samples, h_samples_data.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, h_signatures_data.data(), total_sig_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quals, h_quals_data.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sample_lengths, h_sample_lengths.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_lengths, h_sig_lengths.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate and copy offsets to device
    cudaMalloc(&d_sample_offsets, num_samples * sizeof(size_t));
    cudaMalloc(&d_sig_offsets, num_signatures * sizeof(size_t));

    cudaMemcpy(d_sample_offsets, h_sample_offsets.data(), num_samples * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_offsets, h_sig_offsets.data(), num_signatures * sizeof(size_t), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 threadsPerBlock(16, 16); // 256 threads per block (16x16)
    int total_pairs = num_samples * num_signatures;
    int blocksPerGrid = (total_pairs + (threadsPerBlock.x * threadsPerBlock.y) - 1) / (threadsPerBlock.x * threadsPerBlock.y);

    // Initialize match scores to zero
    cudaMemset(d_match_scores, 0, num_samples * num_signatures * sizeof(float));

    // Launch the optimized multi-pair kernel
    matchKernelMultiPair<<<blocksPerGrid, threadsPerBlock>>>(
        d_samples, d_sample_lengths, d_sample_offsets,
        d_signatures, d_sig_lengths, d_sig_offsets,
        d_quals, d_match_scores,
        num_samples, num_signatures
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        // Handle error as needed
    }

    // Allocate host memory for match scores
    std::vector<float> h_match_scores(num_samples * num_signatures);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

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
}
