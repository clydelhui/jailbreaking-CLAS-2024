from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from huggingface_hub import login
import torch
import json
from tqdm import tqdm
import time

# Set the Hugging Face API key
# Either read from environment variable or provide it directly
# Login to HuggingFace
api_key = os.getenv("HUGGINGFACE_API_KEY")
if api_key:
    login(token=api_key)
else:
    raise EnvironmentError("Hugging Face API key is not set in the environment.")

start_time = time.time()

def get_prompts(file_name):
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list
original_prompt_list = get_prompts(file_name="prompt_test.jsonl")
jailbreak_prompt_list = get_prompts(file_name="./outputs.jsonl")

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
pipe = get_model_inference_pipeline(model_id)

file_name = "responses.jsonl"

# Clear the file by opening it in write mode, then immediately closing it
with open(file_name, 'w') as file:
    pass

# Open the file in append mode before the loop
with open(file_name, 'a') as file:
    for original_prompt, jailbreak_prompt in tqdm(zip(original_prompt_list, jailbreak_prompt_list)):
        response = pipe(jailbreak_prompt + "\n")
        print(response)
        response = response[0]['generated_text']
        
        # Write each response to the file
        dictionary = {"prompt": response}
        json_record = json.dumps(dictionary)
        file.write(json_record + "\n")



end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
print("----------------------------------------")
