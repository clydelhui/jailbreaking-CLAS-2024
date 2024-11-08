from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch
import json
from tqdm import tqdm
import time
import os

# Set the Hugging Face API key
# Either read from environment variable or provide it directly
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
original_prompt_list = get_prompts(file_name="prompt_develop.jsonl")
jailbreak_prompt_list = get_prompts(file_name="./submission/jailbreak.jsonl")

def get_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipe = get_model_inference_pipeline(model_id)

from utils import get_jailbreak_score
from utils import get_stl_score


results = []
avg_jailbreak_score, avg_stealthiness_score = 0., 0.
tokenizer = AutoTokenizer.from_pretrained(model_id)
for original_prompt, jailbreak_prompt in tqdm(zip(original_prompt_list, jailbreak_prompt_list)):
    messages = [
        {"role": "user", "content": jailbreak_prompt},
    ]
    response = pipe(messages)[0]['generated_text'][1]['content']
    jailbreak_score = get_jailbreak_score(pipe, original_prompt, jailbreak_prompt, response, tokenizer)
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

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")