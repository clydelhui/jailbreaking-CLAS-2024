`# Attempt for CLAS 2024

**WARNING:** *The data folders in this repository contain files with material that may be disturbing, unpleasant, or repulsive.*

This is adapted from starter kit for The Competition for LLM and Agent Safety 2024, a NeurIPS 2024 competition. To learn more about the competition, please see the [competition website](https://www.llmagentsafetycomp24.com/).


## Repository Structure

`trl`: Consists of all SFT and PPO training code alongside SLURM scripts
`trl/train.py`: Main PPO training file
`trl/sft.py`: SFT Training file
`trl/wandb.zip`: Complete wandb logs for all trainings attempted

`ClydeStuff/query_model.py`: Inference for generation of modified prompts for submission
`ClydeStuff/jailbreak_HPC.py`: Getting responses from jailbreak target model
`ClydeStuff/eval_loop.py`: Grading of all responses
