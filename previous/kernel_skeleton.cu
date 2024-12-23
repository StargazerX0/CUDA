// kernel_skeleton.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include "kseq/kseq.h"
#include "common.h"

// CUDA Kernel for matching signatures within samples
__global__ void matchKernel(const char* d_samples, int* d_sample_lengths, const size_t* d_sample_offsets,
                            const char* d_signatures, int* d_sig_lengths, const size_t* d_sig_offsets,
                            const char* d_quals, // Phred+33 encoded quality scores
                            double* d_match_scores,
                            int num_samples, int num_signatures) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one sample-signature pair
    if (idx < num_samples * num_signatures) {
        int sample_idx = idx / num_signatures;
        int signature_idx = idx % num_signatures;

        int sample_length = d_sample_lengths[sample_idx];
        int signature_length = d_sig_lengths[signature_idx];

        // Pointers to the current sample and signature
        const char* sample = &d_samples[d_sample_offsets[sample_idx]];
        const char* signature = &d_signatures[d_sig_offsets[signature_idx]];
        const char* qual = &d_quals[d_sample_offsets[sample_idx]];

        bool match = false;
        double confidence_sum = 0.0;
        int match_count = 0;

        // Brute-force search for signature in sample
        for (int i = 0; i <= sample_length - signature_length; ++i) {
            bool current_match = true;
            double current_confidence = 0.0;
            match_count = 0;

            for (int j = 0; j < signature_length; ++j) {
                char s = sample[i + j];
                char sig = signature[j];

                // Handle 'N' as a wildcard in both sample and signature
                if (s != 'N' && sig != 'N' && s != sig) {
                    current_match = false;
                    break;
                }

                // Convert Phred+33 to Phred score
                int phred = (qual[i + j] - 33);
                current_confidence += phred;
                match_count += 1;
            }

            if (current_match && match_count == signature_length) {
                match = true;
                confidence_sum = current_confidence;
                // match_count = signature_length;
                break; // Only the first match is considered
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

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples * num_signatures + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    matchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_samples, d_sample_lengths, d_sample_offsets,
                                                   d_signatures, d_sig_lengths, d_sig_offsets,
                                                   d_quals,
                                                   d_match_scores,
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
