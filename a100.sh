#!/bin/bash

#SBATCH --job-name=exp-a100    # Job name
#SBATCH --output=a100.log        # Output file
#SBATCH --error=a100.log          # Error log file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=20G
#SBATCH --gpus=a100-40
#SBATCH --constraint=xgph
#SBATCH --ntasks=1                 # Number of tasks (1 node)
#SBATCH --time=01:00:00            # Time limit (1 hour)

echo "a100 Job is running on $(hostname), started at $(date)"

# Set the nvidia compiler directory
NVCC=/usr/local/cuda/bin/nvcc

# Check that it exists and print some version info
[[ -f $NVCC ]] || { echo "ERROR: NVCC Compiler not found at $NVCC, exiting..."; exit 1; }
echo "NVCC info: $($NVCC --version)"

# Actually compile the code
echo -e "\n====> Compiling...\n"
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o matcher kernel_skeleton.cu common.cc
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o bonus bonus.cu common.cc
$NVCC -std=c++20 -O3 -lineinfo -dlto -arch=native -o unoptimize unoptimize.cu common.cc

g++ -std=c++20 -O3 -o gen_sample gen_sample.cc
g++ -std=c++20 -O3 -o gen_sig gen_sig.cc

echo -e "\n====> Generate Test case...\n"

./gen_sig 1000 3000 10000 0.1  > test_signatures.fasta
./gen_sample test_signatures.fasta 2000 20 1 2 100000 200000 10 30 0.1 > test_samples.fastq

echo -e "\n====> Bench...\n"
./bench-a100 test_samples.fastq test_signatures.fasta
echo -e "\n====> Brute Force with no optimization...\n"
./unoptimize test_samples.fastq test_signatures.fasta
echo -e "\n====> Final Submission...\n"
./matcher test_samples.fastq test_signatures.fasta
echo -e "\n====> Bonus...\n"
./bonus test_samples.fastq test_signatures.fasta

echo -e "\n====> Finish...\n"