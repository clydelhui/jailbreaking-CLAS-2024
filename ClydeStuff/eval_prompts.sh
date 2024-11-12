#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o output.log
#PBS -N hf_inference_job

# Load modules (if necessary)
cd $PBS_O_WORKDIR
module load miniforge3
module load pytorch/1.11.0-py3-gpu
module load cuda
echo $CUDA_VISIBLE_DEVICES

# Run your Python script
conda activate CS3263-3
export HUGGINGFACE_API_KEY=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
pwd
# python test_gpu.py
python convert_back.py
python jailbreak_HPC.py
python eval_loop.py
