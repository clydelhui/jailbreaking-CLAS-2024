import os
import json
from transformers import pipeline

# Variables to modify for HPC run
input_json_file = "input_prompts_llama.json"  # Input file with prompts in LLaMA chat format
output_json_file = "output_responses_llama.json"  # Output file to store LLM responses
model_name = "gpt2"  # Pre-trained model name from Huggingface (adjust for LLaMA or other models)
device = 0  # Use 0 for GPU, -1 for CPU

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
    # Load prompts from input JSON file
    prompts = load_prompts(input_json_file)

    # Initialize the language model pipeline
    generator = pipeline("text-generation", model=model_name, device=device)

    # List to store the outputs in LLaMA format
    outputs = []

    # Loop through each prompt and generate a response
    for prompt in prompts:
        response = generator(prompt, max_length=100, num_return_sequences=1)
        output = {
            "role": "assistant",  # The system role is used for model-generated responses
            "content": response[0]['generated_text']
        }
        outputs.append(output)

    # Save the responses to the output JSON file
    save_responses(outputs, output_json_file)
    print(f"Responses written to {output_json_file}.")

if __name__ == "__main__":
    main()
