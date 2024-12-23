// optimized_kernel_corrected.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring> // For memcpy
#include "kseq/kseq.h" // Ensure kseq is properly included
#include "common.h" // Ensure common.h defines MatchResult and other necessary structures

#define MAX_SIGNATURE_LENGTH 10000 // Maximum length of a signature
#define MAX_SIGNATURES_IN_BATCH 6  // Number of signatures to process per batch
#define SAMPLE_CHUNK_SIZE 16384    // Size of sample chunk to load into shared memory

// Declare constant memory for signatures
__constant__ char d_signatures_const[MAX_SIGNATURES_IN_BATCH * MAX_SIGNATURE_LENGTH];
__constant__ int d_sig_lengths_const[MAX_SIGNATURES_IN_BATCH];

// Device function to access constant memory signatures
__device__ char getSignatureChar(int sig_idx, int pos) {
    return d_signatures_const[sig_idx * MAX_SIGNATURE_LENGTH + pos];
}

// Atomic max function for double precision
__device__ double atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(old);
        double new_val = fmax(val, old_val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CUDA Kernel
__global__ void matchKernel(const char* __restrict__ d_samples, const int* __restrict__ d_sample_lengths,
                            const size_t* __restrict__ d_sample_offsets, const char* __restrict__ d_quals,
                            double* d_match_scores, int num_samples, int num_signatures_in_batch,
                            int batch_offset, int num_signatures) {
    // Calculate sample index
    int sample_idx = blockIdx.x;

    if (sample_idx >= num_samples) return;

    int sample_length = d_sample_lengths[sample_idx];
    const char* sample = &d_samples[d_sample_offsets[sample_idx]];
    const char* qual = &d_quals[d_sample_offsets[sample_idx]];

    int tid = threadIdx.x;

    // Process the sample in chunks to fit into shared memory
    for (size_t chunk_start = 0; chunk_start < sample_length; chunk_start += SAMPLE_CHUNK_SIZE) {
        size_t chunk_end = min(chunk_start + SAMPLE_CHUNK_SIZE + MAX_SIGNATURE_LENGTH - 1, static_cast<size_t>(sample_length));
        size_t chunk_size = chunk_end - chunk_start;

        // Shared memory for sample chunk
        extern __shared__ char s_sample[];
        // Load sample chunk into shared memory
        for (size_t i = tid; i < chunk_size; i += blockDim.x) {
            s_sample[i] = sample[chunk_start + i];
        }
        __syncthreads();

        // Each thread processes multiple positions within the chunk
        for (int sig_batch_idx = 0; sig_batch_idx < num_signatures_in_batch; ++sig_batch_idx) {
            int signature_length = d_sig_lengths_const[sig_batch_idx];
            
            // Validate signature_length to prevent negative num_positions
            if (signature_length > chunk_size) continue;

            int num_positions = chunk_size - signature_length + 1;

            for (size_t i = tid; i < num_positions; i += blockDim.x) {
                bool match = true;
                double confidence_sum = 0.0;

                // Loop over the signature length
                for (int j = 0; j < signature_length; ++j) {
                    char s = s_sample[i + j];
                    char sig = getSignatureChar(sig_batch_idx, j);

                    // Handle 'N' as a wildcard
                    if (s != 'N' && sig != 'N' && s != sig) {
                        match = false;
                        break;
                    }

                    // Convert Phred+33 to Phred score
                    int phred = qual[chunk_start + i + j] - 33;
                    confidence_sum += phred;
                }

                if (match) {
                    // Calculate average confidence score
                    double avg_confidence = confidence_sum / signature_length;
                    int sig_idx = batch_offset + sig_batch_idx;

                    // Atomic update to store the highest confidence score
                    atomicMax_double(&d_match_scores[sample_idx * num_signatures + sig_idx], avg_confidence);
                }
            }
        }
        __syncthreads();
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

    // Validate that no signature exceeds MAX_SIGNATURE_LENGTH
    for (int i = 0; i < num_signatures; ++i) {
        if (signatures[i].seq.length() > MAX_SIGNATURE_LENGTH) {
            std::cerr << "Error: Signature \"" << signatures[i].name << "\" exceeds MAX_SIGNATURE_LENGTH of "
                      << MAX_SIGNATURE_LENGTH << " characters." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Calculate total sample length and prepare offsets
    int total_sample_length = 0;

    std::vector<size_t> h_sample_offsets(num_samples);
    std::vector<int> h_sample_lengths(num_samples);
    size_t offset = 0;
    for (int i = 0; i < num_samples; ++i) {
        h_sample_lengths[i] = samples[i].seq.length();
        total_sample_length += samples[i].seq.length();
        h_sample_offsets[i] = offset;
        offset += samples[i].seq.length();
    }

    // Allocate pinned host memory for samples and qualities
    char* h_samples_pinned;
    char* h_quals_pinned;
    int* h_sample_lengths_pinned;
    size_t* h_sample_offsets_pinned;
    double* h_match_scores_pinned;

    cudaError_t err;
    err = cudaHostAlloc((void**)&h_samples_pinned, total_sample_length * sizeof(char), cudaHostAllocDefault);
    if (err != cudaSuccess) { 
        std::cerr << "cudaHostAlloc failed for h_samples_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaHostAlloc((void**)&h_quals_pinned, total_sample_length * sizeof(char), cudaHostAllocDefault);
    if (err != cudaSuccess) { 
        std::cerr << "cudaHostAlloc failed for h_quals_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaHostAlloc((void**)&h_sample_lengths_pinned, num_samples * sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) { 
        std::cerr << "cudaHostAlloc failed for h_sample_lengths_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaHostAlloc((void**)&h_sample_offsets_pinned, num_samples * sizeof(size_t), cudaHostAllocDefault);
    if (err != cudaSuccess) { 
        std::cerr << "cudaHostAlloc failed for h_sample_offsets_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaHostAlloc((void**)&h_match_scores_pinned, num_samples * num_signatures * sizeof(double), cudaHostAllocDefault);
    if (err != cudaSuccess) { 
        std::cerr << "cudaHostAlloc failed for h_match_scores_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Populate pinned host memory
    for (int i = 0; i < num_samples; ++i) {
        memcpy(&h_samples_pinned[h_sample_offsets[i]], samples[i].seq.c_str(), h_sample_lengths[i]);
        memcpy(&h_quals_pinned[h_sample_offsets[i]], samples[i].qual.c_str(), h_sample_lengths[i]);
    }
    memcpy(h_sample_lengths_pinned, h_sample_lengths.data(), num_samples * sizeof(int));
    memcpy(h_sample_offsets_pinned, h_sample_offsets.data(), num_samples * sizeof(size_t));
    memset(h_match_scores_pinned, 0, num_samples * num_signatures * sizeof(double));

    // Allocate device memory
    char* d_samples;
    char* d_quals;
    double* d_match_scores;
    int* d_sample_lengths;
    size_t* d_sample_offsets;

    err = cudaMalloc(&d_samples, total_sample_length * sizeof(char));
    if (err != cudaSuccess) { 
        std::cerr << "cudaMalloc failed for d_samples: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMalloc(&d_quals, total_sample_length * sizeof(char));
    if (err != cudaSuccess) { 
        std::cerr << "cudaMalloc failed for d_quals: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMalloc(&d_sample_lengths, num_samples * sizeof(int));
    if (err != cudaSuccess) { 
        std::cerr << "cudaMalloc failed for d_sample_lengths: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMalloc(&d_sample_offsets, num_samples * sizeof(size_t));
    if (err != cudaSuccess) { 
        std::cerr << "cudaMalloc failed for d_sample_offsets: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMalloc(&d_match_scores, num_samples * num_signatures * sizeof(double));
    if (err != cudaSuccess) { 
        std::cerr << "cudaMalloc failed for d_match_scores: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Create a single CUDA stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Asynchronously copy data from host to device
    err = cudaMemcpyAsync(d_samples, h_samples_pinned, total_sample_length * sizeof(char), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemcpyAsync failed for d_samples: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMemcpyAsync(d_quals, h_quals_pinned, total_sample_length * sizeof(char), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemcpyAsync failed for d_quals: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMemcpyAsync(d_sample_lengths, h_sample_lengths_pinned, num_samples * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemcpyAsync failed for d_sample_lengths: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    err = cudaMemcpyAsync(d_sample_offsets, h_sample_offsets_pinned, num_samples * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemcpyAsync failed for d_sample_offsets: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Initialize match scores to zero on device
    err = cudaMemsetAsync(d_match_scores, 0, num_samples * num_signatures * sizeof(double), stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemsetAsync failed for d_match_scores: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Wait for initial data transfer to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaStreamSynchronize failed after initial data transfer: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    int threadsPerBlock = 256;
    size_t sharedMemSize = (SAMPLE_CHUNK_SIZE + MAX_SIGNATURE_LENGTH - 1) * sizeof(char); // For s_sample

    // Process signatures in batches using a single stream
    int num_signatures_processed = 0;
    while (num_signatures_processed < num_signatures) {
        int signatures_in_batch = std::min(MAX_SIGNATURES_IN_BATCH, num_signatures - num_signatures_processed);

        // Prepare signatures and lengths for the current batch
        std::vector<char> h_signatures_batch(signatures_in_batch * MAX_SIGNATURE_LENGTH, 'N'); // Initialize with 'N' for padding
        std::vector<int> h_sig_lengths_batch(signatures_in_batch, 0);

        for (int i = 0; i < signatures_in_batch; ++i) {
            int sig_idx = num_signatures_processed + i;
            int sig_length = signatures[sig_idx].seq.length();
            h_sig_lengths_batch[i] = sig_length;
            memcpy(&h_signatures_batch[i * MAX_SIGNATURE_LENGTH], signatures[sig_idx].seq.c_str(), sig_length);
            // Remaining characters are already 'N' due to initialization
        }

        // Asynchronously copy signatures and lengths to constant memory in the single stream
        err = cudaMemcpyToSymbolAsync(d_signatures_const, h_signatures_batch.data(),
                                      signatures_in_batch * MAX_SIGNATURE_LENGTH * sizeof(char),
                                      0, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) { 
            std::cerr << "cudaMemcpyToSymbolAsync failed for d_signatures_const: " << cudaGetErrorString(err) << std::endl; 
            exit(EXIT_FAILURE); 
        }

        err = cudaMemcpyToSymbolAsync(d_sig_lengths_const, h_sig_lengths_batch.data(),
                                      signatures_in_batch * sizeof(int),
                                      0, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) { 
            std::cerr << "cudaMemcpyToSymbolAsync failed for d_sig_lengths_const: " << cudaGetErrorString(err) << std::endl; 
            exit(EXIT_FAILURE); 
        }

        // Launch the kernel
        int blocksPerGrid = num_samples;
        matchKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
            d_samples, d_sample_lengths, d_sample_offsets,
            d_quals, d_match_scores,
            num_samples, signatures_in_batch,
            num_signatures_processed, num_signatures
        );

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) { 
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl; 
            exit(EXIT_FAILURE); 
        }

        // Increment the processed signatures
        num_signatures_processed += signatures_in_batch;
    }

    // Wait for all kernel executions and memory transfers to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaStreamSynchronize failed after kernel launches: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Asynchronously copy match scores back to host
    err = cudaMemcpyAsync(h_match_scores_pinned, d_match_scores, num_samples * num_signatures * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaMemcpyAsync failed for h_match_scores_pinned: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Wait for all operations in the stream to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { 
        std::cerr << "cudaStreamSynchronize failed after copying match scores: " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    }

    // Populate the matches vector
    matches.reserve(matches.size() + num_samples * num_signatures); // Reserve enough space to avoid reallocations
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_signatures; ++j) {
            double score = h_match_scores_pinned[i * num_signatures + j];
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
    cudaFree(d_quals);
    cudaFree(d_match_scores);
    cudaFree(d_sample_lengths);
    cudaFree(d_sample_offsets);

    // Free host pinned memory
    cudaFreeHost(h_samples_pinned);
    cudaFreeHost(h_quals_pinned);
    cudaFreeHost(h_sample_lengths_pinned);
    cudaFreeHost(h_sample_offsets_pinned);
    cudaFreeHost(h_match_scores_pinned);

    // Destroy CUDA stream
    cudaStreamDestroy(stream);
}

