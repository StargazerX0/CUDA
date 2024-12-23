// kernel_skeleton.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include "kseq/kseq.h"
#include "common.h"

__device__ double atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void matchKernel(const char* __restrict__ d_samples, const int* __restrict__ d_sample_lengths,
                            const size_t* __restrict__ d_sample_offsets, const char* __restrict__ d_signatures,
                            const int* __restrict__ d_sig_lengths, const size_t* __restrict__ d_sig_offsets,
                            const char* __restrict__ d_quals, double* d_match_scores,
                            int num_samples, int num_signatures) {
    // Calculate sample and signature indices
    int pair_idx = blockIdx.x;
    int sample_idx = pair_idx / num_signatures;
    int signature_idx = pair_idx % num_signatures;

    // Bounds check
    if (sample_idx >= num_samples || signature_idx >= num_signatures) return;

    int sample_length = d_sample_lengths[sample_idx];
    int signature_length = d_sig_lengths[signature_idx];

    // Pointers to the current sample and signature
    const char* sample = &d_samples[d_sample_offsets[sample_idx]];
    const char* signature = &d_signatures[d_sig_offsets[signature_idx]];
    const char* qual = &d_quals[d_sample_offsets[sample_idx]];

    int num_positions = sample_length - signature_length + 1;

    // Each thread processes multiple positions within the sample
    for (int i = threadIdx.x; i < num_positions; i += blockDim.x) {
        bool match = true;
        double confidence_sum = 0.0;

        // Loop over the signature length
        for (int j = 0; j < signature_length; ++j) {
            char s = sample[i + j];
            char sig = signature[j];

            // Handle 'N' as a wildcard
            if (s != 'N' && sig != 'N' && s != sig) {
                match = false;
                break;
            }

            // Convert Phred+33 to Phred score
            int phred = qual[i + j] - 33;
            confidence_sum += phred;
        }

        if (match) {
            // Calculate average confidence score
            double avg_confidence = confidence_sum / signature_length;
            int idx = sample_idx * num_signatures + signature_idx;

            // Atomic update to store the highest confidence score
            atomicMax_double(&d_match_scores[idx], avg_confidence);
        }
    }
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
    std::vector<char> h_samples(total_sample_length);
    std::vector<char> h_signatures(total_sig_length);
    std::vector<char> h_quals(total_sample_length);

    // Populate host memory
    for (int i = 0; i < num_samples; ++i) {
        memcpy(&h_samples[h_sample_offsets[i]], samples[i].seq.c_str(), h_sample_lengths[i]);
        memcpy(&h_quals[h_sample_offsets[i]], samples[i].qual.c_str(), h_sample_lengths[i]);
    }
    for (int i = 0; i < num_signatures; ++i) {
        memcpy(&h_signatures[h_sig_offsets[i]], signatures[i].seq.c_str(), h_sig_lengths[i]);
    }

    // Allocate device memory
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

    // Copy data from host to device
    cudaMemcpy(d_samples, h_samples.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, h_signatures.data(), total_sig_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quals, h_quals.data(), total_sample_length * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sample_lengths, h_sample_lengths.data(), num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_lengths, h_sig_lengths.data(), num_signatures * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate and copy offsets to device
    size_t* d_sample_offsets;
    size_t* d_sig_offsets;
    cudaMalloc(&d_sample_offsets, num_samples * sizeof(size_t));
    cudaMalloc(&d_sig_offsets, num_signatures * sizeof(size_t));

    cudaMemcpy(d_sample_offsets, h_sample_offsets.data(), num_samples * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_offsets, h_sig_offsets.data(), num_signatures * sizeof(size_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int num_sample_sig_pairs = num_samples * num_signatures;
    int blocksPerGrid = num_sample_sig_pairs;

    // Initialize match scores to zero
    cudaMemset(d_match_scores, 0, num_samples * num_signatures * sizeof(double));

    // Launch the kernel
    matchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_samples, d_sample_lengths, d_sample_offsets,
                                                    d_signatures, d_sig_lengths, d_sig_offsets,
                                                    d_quals, d_match_scores,
                                                    num_samples, num_signatures);

    // Allocate host memory for match scores
    std::vector<double> h_match_scores(num_samples * num_signatures);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    cudaMemcpy(h_match_scores.data(), d_match_scores, num_samples * num_signatures * sizeof(double), cudaMemcpyDeviceToHost);

    // Populate the matches vector
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_signatures; ++j) {
            double score = h_match_scores[i * num_signatures + j];
            if (score > 0.0) { // Assuming a match is indicated by score > 0
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
