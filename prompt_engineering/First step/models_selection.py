import sys
import os
import csv
import time
import textwrap
import re
from llama_cpp import Llama

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(parent_dir)

from simulator_env import SwarmAgent

def call_behaviors() -> dict:
    behavior_dict = {}
    for name, attribute in SwarmAgent.__dict__.items():
        if callable(attribute) and not name.startswith("_") \
           and not name.startswith("update") and not name.startswith("obstacle"):
            doc = attribute.__doc__
            if doc is not None:
                cleaned_doc = " ".join(textwrap.dedent(doc).strip().split())
            else:
                cleaned_doc = ""
            behavior_dict[name] = cleaned_doc
    return behavior_dict

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

# def save_behavior_tree(tree_xml: str, file_name: str) -> None:
#     """
#     Saves the behavior tree XML to a file with the given file name.
#     """
#     with open(file_name, "w", encoding="utf-8") as f:
#         f.write(tree_xml)

def write_results(prompt: str, generation_time: float, behavior_tree: str, model_name: str, prompt_type: str, filename: str):
    """
    Logs the prompt, prompt type, generation time, behavior tree, and model name to a CSV file.
    """
    file_path = os.path.join(os.getcwd(), filename)
    file_exists = os.path.isfile(file_path)
    fieldnames = ['Prompt', 'Prompt Type', 'Time', 'Behavior Tree', 'Model Name']
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
         writer = csv.DictWriter(file, fieldnames=fieldnames)
         if not file_exists:
             writer.writeheader()
         data_row = {
             'Prompt': prompt,
             'Prompt Type': prompt_type,
             'Time': generation_time,
             'Behavior Tree': behavior_tree,
             'Model Name': model_name
         }
         writer.writerow(data_row)

def construct_prompt(prompt: str, prompt_type: str) -> str:
    behaviors = call_behaviors()
    behaviors_text = "\n".join(f"{name}: {doc}" for name, doc in behaviors.items())

    plan_prompt = f"""
    <s>
    <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
    INSTRUCTIONS: It is CRITICAL to use only the following behaviors structured as a dictionary: {behaviors_text} to construct behavior tree in XML format for the following command. Including any behavior that is not in the provided dictionary can result in damage to the agents and potentially humans, therefore you are not allowed to do so. AVOID AT ALL COSTS.
    USER COMMAND: generate behavior tree to "{prompt}". Take a step back and think deeply about the behavior you need for this command. Consider the XML structure and the behaviors you use.
    The output MUST follow this XML structure exactly, including:
    - A root element with BTCPP_format and main_tree_to_execute attributes.
    - A <BehaviorTree> element with an inner structure of Sequences, Fallback, Conditions, and Actions.
    - A <TreeNodesModel> section listing all node models.
    - No additional text or commentary outside the XML.
    Output only the XML behavior tree without extra text.
    OUTPUT:
    """

    if prompt_type == "zero":
        return plan_prompt
    elif prompt_type == "one":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\prompt_engineering\First step\prompt_types\One_shot.txt"
        with open(path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return f"{file_content} {plan_prompt}"
    elif prompt_type == "two":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\prompt_engineering\First step\prompt_types\Two Shot.txt"
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
    csv_filename = rf"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\prompt_engineering\First step\resultes\{model_base}_log.csv"
    write_results(scenario, elapsed_time, tree_xml, model_base, prompt_type, csv_filename)

    print(f"Saved XML to {xml_filename} (Time taken: {elapsed_time:.2f}s)")

# Example usage:
if __name__ == "__main__":
    # scenarios = ["form a line.", "find target change color to green"]
    scenarios = [
    "detect an obstacle, avoid it, and then changing color to green",
    "wander randomly until a target is detected, approach it, and signal achievement by changing color to red",
    "check if path is clear, form a line at the center",
    "find the goal, signal success by changing color to red, and align movement with other swarm agents",
    "detect target, freeze movement upon reaching it"
    ]

    model_paths = [
    # r"G:\Inventors Hub Projects\SwarmChat\model\deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\model\deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\model\DeepSeek-R1-Distill-llama-8B-Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\model\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",

    # r"G:\Inventors Hub Projects\SwarmChat\model\DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",

    r"G:\Inventors Hub Projects\SwarmChat\models\EuroLLM-9B-Instruct-Q4_K_M.gguf",
    # r"G:\Inventors Hub Projects\SwarmChat\model\eurollm-1.7b-instruct-q4_k_m.gguf",
    # r"G:\Inventors Hub Projects\SwarmChat\model\Falcon3-7B-Instruct-q4_k_m.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\model\Falcon3-10B-Instruct-q4_k_m.gguf",

    r"G:\Inventors Hub Projects\SwarmChat\models\phi-4-Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\models\CodeLlama-7b-hf-Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\model\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    r"G:\Inventors Hub Projects\SwarmChat\models\deepseek-coder-7b-base-v1.5.Q4_K_M.gguf",

    r"G:\Inventors Hub Projects\SwarmChat\model\Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",    
    # r"G:\Inventors Hub Projects\SwarmChat\model\Qwen2-VL-7B-Instruct-Q4_K_M.gguf",

    r"G:\Inventors Hub Projects\SwarmChat\model\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    ]


    for model in model_paths:
        llm = Llama(model_path=model, n_ctx=1024*4)
        for prompt_type in ['zero','one','two']:
            for idx, scenario in enumerate(scenarios):
                main(llm, model, prompt_type, scenario, scenario_idx=idx)
    