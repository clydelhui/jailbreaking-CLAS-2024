#!/bin/bash
#SBATCH --job-name=jailbreak_prompt_dev_pipeline
#SBATCH --output=jailbreak_prompt_dev%j.log
#SBATCH --error=jailbreak_prompt_dev%j.err
#SBATCH --partition=gpu
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100-80:1

# Activate conda
eval "$(/home/c/clydel/miniconda3/bin/conda shell.bash hook)"
conda activate jailbreak
conda list

# Run python script
cd ~/CLAS
pwd
export HUGGINGFACE_API_KEY=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
python query_model.py

echo "Model run has been completed" 

# Deactivate the environment after installation
conda deactivate

