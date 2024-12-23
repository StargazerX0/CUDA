#!/bin/bash

#SBATCH --job-name=compile_cuda    # Job name
#SBATCH --output=exp.log        # Output file
#SBATCH --error=exp_error.log          # Error log file
#SBATCH --partition=gpu            # GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Number of tasks (1 node)
#SBATCH --time=01:00:00            # Time limit (1 hour)

echo "Job is running on $(hostname), started at $(date)"

# Set the nvidia compiler directory
NVCC=/usr/local/cuda/bin/nvcc

# Check that it exists and print some version info
[[ -f $NVCC ]] || { echo "ERROR: NVCC Compiler not found at $NVCC, exiting..."; exit 1; }
echo "NVCC info: $($NVCC --version)"

# Actually compile the code
echo -e "\n====> Compiling...\n"
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o matcher kernel_skeleton.cu common.cc
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o coalesced coalesced.cu common.cc
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o shared shared.cu common.cc
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o bonus bonus.cu common.cc
g++ -std=c++20 -O3 -o gen_sample gen_sample.cc
g++ -std=c++20 -O3 -o gen_sig gen_sig.cc

echo -e "\n====> Running bench...\n"
--ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus a100-40 --constraint xgph ./bench-a100 test_samples.fastq test_signatures.fasta > bench_output.txt

echo -e "\n====> Finish\n"







# ./gen_sig 1000 3000 10000 0.1  > test_signatures.fasta
# ./gen_sample test_signatures.fasta 2000 20 1 2 100000 200000 10 30 0.1 > test_samples.fastq
