import sys
import os
import csv
import time
import textwrap
import re
from llama_cpp import Llama
import json


def extract_behavior_tree(response: str) -> str:
    """
    Extracts an XML behavior tree from the given response text.
    Looks for a block of XML enclosed in <root ... </root> tags.
    """
    pattern = re.compile(r'(<root.*?</root>)', re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    else:
        return response.strip()

def generate_behavior_tree(task_prompt: str, llm: Llama) -> str:
    """
    Generates a behavior tree for the provided task prompt by calling the Llama model.
    """
    output = llm(
        task_prompt,
        temperature=0,
        max_tokens=1024,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.1
    )
    response = output.get("choices", [{}])[0].get("text", "").strip()
    return response



def write_results(prompt: dict, generation_time: float, behavior_tree: str, model_name: str, prompt_type: str, filename: str):
    """
    Logs the prompt, prompt type, generation time, behavior tree, and model name to a CSV file.
    """
    file_path = os.path.join(os.getcwd(), filename)
    file_exists = os.path.isfile(file_path)
    fieldnames = ['SYSTEM', 'INSTRUCTIONS', 'USER COMMAND', 'Ground Truth BT', 'Prompt Type', 'Time', 'Behavior Tree', 'Model Name']
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
         writer = csv.DictWriter(file, fieldnames=fieldnames)
         if not file_exists:
             writer.writeheader()
         data_row = {
             'SYSTEM': prompt["SYSTEM"],
             'INSTRUCTIONS': prompt["INSTRUCTIONS"],
             'USER COMMAND': prompt["USER COMMAND"],
             'Ground Truth BT': prompt["OUTPUT"],
             'Prompt Type': prompt_type,
             'Time': generation_time,
             'Behavior Tree': behavior_tree,
             'Model Name': model_name
         }
         writer.writerow(data_row)


def construct_prompt(prompt: str, prompt_type: str) -> str:
    # behaviors = call_behaviors()
    # behaviors_text = "\n".join(f"{name}: {doc}" for name, doc in behaviors.items())

    plan_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
SYSTEM: {prompt["SYSTEM"]}
INSTRUCTIONS: {prompt["INSTRUCTIONS"]}
USER COMMAND: {prompt["USER COMMAND"]} Take another step back and think of the xml structure and the behavior you used.
The output MUST follow this XML structure exactly, including:
- A root element with < root BTCPP_format and main_tree_to_execute attributes.
- A <BehaviorTree> element with an inner structure of Sequences, Fallback, Conditions, and Actions.
- A <TreeNodesModel> section listing all node models.
- No additional text or commentary outside the XML.
Output only the XML behavior tree without extra text.

OUTPUT:
    
    """


    if prompt_type == "zero":
        return plan_prompt
    elif prompt_type == "one":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\finetuning\prompt_types\One_shot.txt"
        with open(path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return f"{file_content} {plan_prompt}"
    elif prompt_type == "two":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\finetuning\prompt_types\Two Shot.txt"
        with open(path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return f"{file_content} {plan_prompt}"
    else:
        raise ValueError("Unknown prompt type provided.")

def main(llm: Llama, model_path: str, prompt_type: str, scenario: str, scenario_idx: int = 0):
    prompt = construct_prompt(scenario, prompt_type)

    start_time = time.time()
    response = generate_behavior_tree(prompt, llm)
    tree_xml = extract_behavior_tree(response)
    elapsed_time = time.time() - start_time

    # Create a unique XML filename based on the model name, prompt type, and scenario index.
    model_base = os.path.splitext(os.path.basename(model_path))[0]
    xml_filename = f"{model_base}_{prompt_type}_{scenario_idx}.xml"
    # save_behavior_tree(tree_xml, file_name=xml_filename)

    # Log the results to a CSV file named based on the model.
    csv_filename = rf"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\finetuning\resultes\{model_base}_log.csv"
    write_results(scenario, elapsed_time, tree_xml, model_base, prompt_type, csv_filename)

    print(f"Saved XML to {xml_filename} (Time taken: {elapsed_time:.2f}s)")

def read_jsonl(path):

    val_dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            val_dataset.append(json.loads(line))
    return val_dataset





# Example usage:
if __name__ == "__main__":
    
    

    path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\finetuning\validation_data.jsonl"
    scenarios = read_jsonl(path)
    # print(construct_prompt(scenarios[0], "one"))

    model_paths = [
    r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-1epochs.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-3epochs.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-5epochs.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-8epochs.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-10epochs.Q4_K_M.gguf",
    ]
    print(os.path.exists(model_paths[0]))  # should be True

    for model in model_paths:
        llm = Llama(model_path=model, n_ctx=1024*4)
        for prompt_type in ['zero','one','two']:
            for idx, scenario in enumerate(scenarios):
                main(llm, model, prompt_type, scenario, scenario_idx=idx)