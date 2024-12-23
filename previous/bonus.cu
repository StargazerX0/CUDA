// kernel_skeleton_rabin_karp.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include "kseq/kseq.h"
#include "common.h"

// CUDA device function for calculating hash on the GPU
__device__ unsigned long calculateDeviceHash(const char* str, int length, int prime, int base) {
    unsigned long hash = 0;
    for (int i = 0; i < length; ++i) {
        hash = ((hash << 8) + str[i]) & (prime - 1);
    }
    return hash;
}

// CUDA Kernel for Rabin-Karp matching signatures within samples
__global__ void rabinKarpKernel(const char* d_samples, int* d_sample_lengths, const size_t* d_sample_offsets,
                                const char* d_signatures, int* d_sig_lengths, const size_t* d_sig_offsets,
                                const char* d_quals, // Phred+33 encoded quality scores
                                double* d_match_scores, int num_samples, int num_signatures,
                                int prime, int base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_samples * num_signatures) {
        int sample_idx = idx / num_signatures;
        int signature_idx = idx % num_signatures;

        int sample_length = d_sample_lengths[sample_idx];
        int signature_length = d_sig_lengths[signature_idx];

        const char* sample = &d_samples[d_sample_offsets[sample_idx]];
        const char* signature = &d_signatures[d_sig_offsets[signature_idx]];
        const char* qual = &d_quals[d_sample_offsets[sample_idx]];

        // Calculate the hash for the current signature on the device
        unsigned long hash_sig = calculateDeviceHash(signature, signature_length, prime, base);

        unsigned long hash_sample = 0;
        int highest_base_power = 1;

        // Calculate the initial hash for the first substring of the sample
        for (int i = 0; i < signature_length; ++i) {
            hash_sample = ((hash_sample << 8) + sample[i]) & (prime - 1);
            if (i < signature_length - 1) {
                highest_base_power = (highest_base_power << 8) & (prime - 1);
            }
        }

        double confidence_sum = 0.0;
        int match_count = 0;
        bool match = false;

        for (int i = 0; i <= sample_length - signature_length; ++i) {
            // If hash matches, check the actual string to confirm
            if (hash_sample == hash_sig) {
                match = true;
                confidence_sum = 0.0;
                match_count = 0;
                for (int j = 0; j < signature_length; j++) {
                    if (sample[i + j] != signature[j] && signature[j] != 'N' && sample[i + j] != 'N') {
                        match = false;
                        break;
                    }
                    int phred = (qual[i + j] - 33);
                    confidence_sum += phred;
                    match_count += 1;
                }
                if (match) {
                    break; // Only the first match is considered
                }
            }

            // Rolling hash: Update hash_sample by removing the leading character and adding the next one
            if (i < sample_length - signature_length) {
                // Remove the effect of previous character
                hash_sample = (hash_sample - (sample[i] * highest_base_power) & (prime - 1) + prime) & (prime - 1);
                // Add effect of next character
                hash_sample = ((hash_sample << 8) + sample[i + signature_length]) & (prime - 1);
            }
        }

        // Calculate average confidence score
        if (match) {
            d_match_scores[idx] = confidence_sum / match_count;
        } else {
            d_match_scores[idx] = 0.0; // No match
        }
    }
}

// Main function to execute Rabin-Karp Matcher
void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches) {

    int num_samples = samples.size();
    int num_signatures = signatures.size();

    if (num_samples == 0 || num_signatures == 0) {
        return;
    }

    int prime = 65536;  //base two for and operation. A prime number used in hash calculations
    int base = 256;   // Base used in hash calculations

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

    std::vector<char> h_samples(total_sample_length);
    std::vector<char> h_signatures(total_sig_length);
    std::vector<char> h_quals(total_sample_length);

    for (int i = 0; i < num_samples; ++i) {
        memcpy(&h_samples[h_sample_offsets[i]], samples[i].seq.c_str(), h_sample_lengths[i]);
        memcpy(&h_quals[h_sample_offsets[i]], samples[i].qual.c_str(), h_sample_lengths[i]);
    }
    for (int i = 0; i < num_signatures; ++i) {
        memcpy(&h_signatures[h_sig_offsets[i]], signatures[i].seq.c_str(), h_sig_lengths[i]);
    }

    char* d_samples;
    char* d_signatures;
    char* d_quals;
    double* d_match_scores;
    int* d_sample_lengths;
    int* d_sig_lengths;

    cudaMalloc(&d_samples, total_sample_length * sizeof(char));
    cudaMalloc(&d_signatures, total_sig_length * sizeof(char));
    cudaMalloc(&d_quals, total_sample_length * sizeof(char));
    cudaMalloc(&d_match_scores, num_samples * num_signatures * sizeof(double));
    cudaMalloc(&d_sample_lengths, num_samples * sizeof(int));
    cudaMalloc(&d_sig_lengths, num_signatures * sizeof(int));

    cudaMemcpy(d_samples, h_samples.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, h_signatures.data(), total_sig_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quals, h_quals.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sample_lengths, h_sample_lengths.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_lengths, h_sig_lengths.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);

    size_t* d_sample_offsets;
    size_t* d_sig_offsets;
    cudaMalloc(&d_sample_offsets, num_samples * sizeof(size_t));
    cudaMalloc(&d_sig_offsets, num_signatures * sizeof(size_t));

    cudaMemcpy(d_sample_offsets, h_sample_offsets.data(), num_samples * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_offsets, h_sig_offsets.data(), num_signatures * sizeof(size_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples * num_signatures + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the Rabin-Karp kernel
    rabinKarpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_samples, d_sample_lengths, d_sample_offsets,
                                                        d_signatures, d_sig_lengths, d_sig_offsets,
                                                        d_quals, d_match_scores, num_samples, num_signatures,
                                                        prime, base);

    cudaDeviceSynchronize();

    std::vector<double> h_match_scores(num_samples * num_signatures);
    cudaMemcpy(h_match_scores.data(), d_match_scores, num_samples * num_signatures * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_signatures; ++j) {
            double score = h_match_scores[i * num_signatures + j];
            if (score > 0.0) {
                MatchResult result;
                result.sample_name = samples[i].name;
                result.signature_name = signatures[j].name;
                result.match_score = score;
                matches.push_back(result);
            }
        }
    }

    cudaFree(d_samples);
    cudaFree(d_signatures);
    cudaFree(d_quals);
    cudaFree(d_match_scores);
    cudaFree(d_sample_lengths);
    cudaFree(d_sig_lengths);
    cudaFree(d_sample_offsets);
    cudaFree(d_sig_offsets);
}
