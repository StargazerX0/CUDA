# Parallel Virus Scanning with CUDA

# Generate and Prepare Test Data
```bash
./gen_sig 1000 3000 10000 0.1  > test_signatures.fasta
./gen_sample test_signatures.fasta 2000 20 1 2 100000 200000 10 30 0.1 > test_samples.fastq
```
# run matcher
```bash
srun --gpus=h100-96 ./matcher test_samples.fastq test_signatures.fasta > test_output.txt
```

```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus a100-40 --constraint xgph ./matcher test_samples.fastq test_signatures.fasta > test_output.txt
```

```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher test_samples.fastq test_signatures.fasta > test_output.txt
```

# Validate Correctness with Benchmarks
```bash
srun --gpus=h100-96 ./bench-h100 test_samples.fastq test_signatures.fasta > bench_output.txt
```

```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus a100-40 --constraint xgph ./bench-a100 test_samples.fastq test_signatures.fasta > bench_output.txt
```

```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./bench-h100 test_samples.fastq test_signatures.fasta > bench_output.txt
```

# Compare outputs
```bash
diff test_output.txt bench_output.txt
if [ $? -eq 0 ]; then
    echo "Outputs match. Correctness verified."
else
    echo "Outputs differ. Check your implementation."
fi
```

# Bonus
```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./bonus test_samples.fastq test_signatures.fasta > bonus_output.txt
```

```bash
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus a100-40 --constraint xgph ./bonus test_samples.fastq test_signatures.fasta > bonus_output.txt
```


```bash
diff bonus_output.txt bench_output.txt
if [ $? -eq 0 ]; then
    echo "Outputs match. Correctness verified."
else
    echo "Outputs differ. Check your implementation."
fi
```

# Batch Script (unoptimized program, matcher, bonus)
```bash
sbatch a100.sh
```

```bash
sbatch h100.sh
```
