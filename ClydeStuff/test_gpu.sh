#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o output.log
#PBS -N gpu_detect_test

# Load modules (if necessary)
cd $PBS_O_WORKDIR
module load miniforge3
module load pytorch/1.11.0-py3-gpu
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Run your Python script
conda activate CS3263-3
conda list
python test_gpu.py

