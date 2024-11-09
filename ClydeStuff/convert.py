import json

def convert_jsonl_to_json(input_file, output_file):
    output_data = []

    # Read the jsonl file and convert each line
    with open(input_file, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            item = json.loads(line)
            # Transform into the new format
            new_item = {"role": "user", "content": item["prompt"]}
            output_data.append(new_item)

    # Write the output data to a json file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    convert_jsonl_to_json("./prompt_test.jsonl", "./prompt_test_converted.json")
