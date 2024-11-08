import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Variables to modify for HPC run
input_json_file = "input_prompts_llama.json"  # Input file with prompts in LLaMA chat format
output_json_file = "output_responses_llama.json"  # Output file to store LLM responses
model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"  # Pre-trained model name from Huggingface (adjust for LLaMA or other models)

def load_prompts(file_name):
    """Load prompts from the JSON file in LLaMA format (role, content)."""
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} does not exist.")
    
    with open(file_name, 'r') as file:
        data = json.load(file)
        # Extract only the "content" field from messages where "role" is "user"
        return [entry['content'] for entry in data if entry['role'] == 'user']

def save_responses(responses, output_file):
    """Save responses to a new JSON file."""
    with open(output_file, 'w') as file:
        json.dump(responses, file, indent=4)

def main():
    # Login to Hugging Face Hub using the API key from the environment variable
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        login(token=api_key)
        print("Successfully logged in to Hugging Face Hub.")
    else:
        raise EnvironmentError("HUGGINGFACE_API_KEY environment variable not set.")

    # Automatically check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Using {device_type} for inference.")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Model loaded")

    # Load prompts from the input JSON file
    prompts = load_prompts(input_json_file)
    print("Prompts loaded")

    # List to store the outputs in LLaMA format
    outputs = []

    # Loop through each prompt and generate a response
    for prompt in prompts:
        # Encode the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate the response
        output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode the generated tokens to text
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Store the response in the LLaMA format with "assistant" role
        output = {
            "role": "assistant",  # Changed from "system" to "assistant"
            "content": response_text
        }
        outputs.append(output)
        print("Output appended")

    # Save the responses to the output JSON file
    save_responses(outputs, output_json_file)
    print(f"Responses written to {output_json_file}.")

if __name__ == "__main__":
    main()
