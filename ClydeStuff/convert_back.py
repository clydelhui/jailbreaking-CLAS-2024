import json

def convert_json_to_jsonl(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Write each item as a JSON line in the JSONL file
    with open(output_file, 'w') as file:
        for item in data:
            # Transform back to the original format
            original_item = {"prompt": item["content"]}
            # Convert the dictionary to a JSON string and write it as a line
            file.write(json.dumps(original_item) + '\n')

if __name__ == "__main__":
    convert_json_to_jsonl("response_test_sft.json", "outputs_sft.jsonl")
