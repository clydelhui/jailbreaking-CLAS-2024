import os
import json

def main():
    # Step 1: Get the file name
    file_name = input("Enter the JSON file name (with .json extension): ")

    # Step 2: Check if the file exists, if not create a new file
    if not os.path.isfile(file_name):
        with open(file_name, 'w') as file:
            json.dump([], file)  # Initialize the file with an empty list

    while True:
        # Step 3: Prompt the user for input
        user_prompt = input("Enter a prompt to add to the JSON file (or type 'exit' to stop): ")

        # Stop the loop if the user types 'exit'
        if user_prompt.lower() == 'exit':
            print("Exiting program.")
            break

        # Step 4: Create the new JSON object in LLaMA format
        new_entry = {"role": "user", "content": user_prompt}

        # Step 5: Read the existing content of the JSON file
        with open(file_name, 'r') as file:
            data = json.load(file)  # Load existing content as a list

        # Step 6: Append the new object
        data.append(new_entry)

        # Step 7: Write the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)  # Write data with indentation for readability

        print(f"Prompt added to {file_name} successfully!")

if __name__ == "__main__":
    main()
