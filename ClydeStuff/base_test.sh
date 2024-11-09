#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o output.log
#PBS -N hf_inference_job

# Load modules (if necessary)
cd $PBS_O_WORKDIR
module load miniforge3
module load pytorch/1.11.0-py3-gpu
echo $CUDA_VISIBLE_DEVICES

# Run your Python script
conda activate CS3263-3
export HUGGINGFACE_API_KEY=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
pwd
python query_model.py

