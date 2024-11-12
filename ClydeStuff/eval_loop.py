
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from huggingface_hub import login
import torch
import json
from tqdm import tqdm
import time
from utils import get_jailbreak_score
from utils import get_stl_score

# Set the Hugging Face API key
# Either read from environment variable or provide it directly
# Login to HuggingFace
api_key = os.getenv("HUGGINGFACE_API_KEY")
if api_key:
    login(token=api_key)
else:
    raise EnvironmentError("Hugging Face API key is not set in the environment.")

def get_prompts(file_name):
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list
original_prompt_list = get_prompts(file_name="prompt_test.jsonl")
jailbreak_prompt_list = get_prompts(file_name="./outputs.jsonl")
response_list = get_prompts(file_name="./responses.jsonl")

def get_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens = 100):
    device = -1
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 0
    else:
       raise EnvironmentError("No GPU detected") 
    pipe = pipeline("text-generation", model=model_id, device=device, max_new_tokens = max_new_tokens, return_full_text=False)
    return pipe
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipe_eval = get_model_inference_pipeline(model_id, 300)

results = []
avg_jailbreak_score, avg_stealthiness_score = 0., 0.
tokenizer = AutoTokenizer.from_pretrained(model_id)

for original_prompt, jailbreak_prompt, response in tqdm(zip(original_prompt_list, jailbreak_prompt_list, response_list)):
    jailbreak_score = get_jailbreak_score(pipe_eval, original_prompt, jailbreak_prompt, response, tokenizer)
    stl_score = get_stl_score(original_prompt, jailbreak_prompt)
    record = {"original_prompt": original_prompt, "jailbreak_prompt": jailbreak_prompt, "response": response, "jailbreak_score": jailbreak_score, "stealthiness_score": stl_score}
    results.append(record)
    avg_jailbreak_score += jailbreak_score / len(jailbreak_prompt_list)
    avg_stealthiness_score += stl_score / len(jailbreak_prompt_list)


file_name = "results.jsonl"
with open(file_name, 'w') as file:
    for dictionary in results:
        json_record = json.dumps(dictionary)
        file.write(json_record + "\n")
print(f'Average jailbreak score: {avg_jailbreak_score}') # 0.151 for PAP jailbreak
print(f'Average stealthiness score: {avg_stealthiness_score}') # 0.216 for PAP jailbreak


