from dotenv import load_dotenv
import random
import json
import os
from openai import OpenAI
import pandas as pd


# Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


output_folder = r"Dataset generator\raw_data"
files = os.listdir(output_folder)
prompt_files = [f for f in files if f.startswith("batch_") and f.endswith(".txt")]
next_number = len(prompt_files) + 1
new_filename = f"batch_{next_number}.txt"
new_filepath = os.path.join(output_folder, new_filename)



def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(r"Dataset generator\prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        system, instruction, user_command, output =  task_dict["SYSTEM"], task_dict["INSTRUCTIONS"], task_dict["USER COMMAND"], task_dict["OUTPUT"]

        prompt += f"\n###\n"
        prompt += f"{idx + 1}./#/ SYSTEM: {system}\n"
        prompt += f"{idx + 1}./#/ INSTRUCTIONS: {instruction}\n"
        prompt += f"{idx + 1}./#/ USER COMMAND: {user_command}\n"
        prompt += f"{idx + 1}./#/ OUTPUT:\n{output}\n"
        
    return prompt



for i in range(1):
    output_folder = r"Dataset generator\raw_data"
    files = os.listdir(output_folder)
    prompt_files = [f for f in files if f.startswith("batch_") and f.endswith(".txt")]
    next_number = len(prompt_files) + 1
    new_filename = f"batch_{next_number}.txt"
    new_filepath = os.path.join(output_folder, new_filename)

    for j in range(1):
        print("\n###########,",i ,",,", j,",############\n")

        # Open and read the JSON file
        with open("Dataset generator\seed_examples.json", 'r') as file:
            data = json.load(file)

        seed_instruction_data = [{"SYSTEM": t["SYSTEM"],"INSTRUCTIONS": t["INSTRUCTIONS"],"USER COMMAND": t["USER COMMAND"], "OUTPUT": t["OUTPUT"]} for t in data]

        # Determine batch size randomly between 1 and 4
        batch_size = random.randint(2, 3)

        # Select a random batch of the determined size
        batch_inputs = random.sample(seed_instruction_data, batch_size)

        # print(batch_inputs)

        prompt_tamplet = encode_prompt(batch_inputs)
        # print(prompt)
        response = client.chat.completions.create(
            model="o1-mini-2024-09-12",
            messages=[
                {"role": "system", 
                "content": "You are a helpful assistant. When responding, provide only the final output (do not repeat any of the prompt or instructions). and do not add charicters like (** ``` ``` )",
                "role": "user", 
                "content": prompt_tamplet}
            ]
        )


        text = response.choices[0].message.content


        with open(new_filepath, "a", encoding="utf-8") as file:
            file.write(text + "\n")

