#!/bin/bash
#SBATCH --job-name=install_transformers_conda
#SBATCH --output=install_transformers_conda_%j.log
#SBATCH --error=install_transformers_conda_%j.err
#SBATCH --partition=medium
#SBATCH --time=00:30:00

# Activate conda
eval "$(/home/c/clydel/miniconda3/bin/conda shell.bash hook)"
conda activate jailbreak

# Install necessary packages with conda
conda install -c conda-forge transformers pytorch torchvision torchaudio -y

# Optionally, install other useful packages
conda install -c conda-forge datasets accelerate -y  # Optional packages for data handling and distributed training

echo "All packages have been installed successfully in the conda environment."

# Deactivate the environment after installation
conda deactivate

