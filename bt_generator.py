"""
Generate & save Behavior Tree XML using a base/fine-tuned LLM.

Functions:
    call_behaviors() -> Dict[str,str]:
        Reflectively collects SwarmAgent’s ACTION/CONDITION docstrings into a dict.
    construct_prompt(prompt: str, prompt_type: str="one") -> str:
        Builds zero/one/two-shot LLM prompt including available behaviors.
    generate_behavior_tree(task_prompt: str) -> str:
        Calls the LLM, extracts <root>…</root>, saves to tree.xml, and returns the XML.
"""

from simulator_env import SwarmAgent
import textwrap
import re
from llama_cpp import Llama




model_path = r"G:\Inventors Hub Projects\SwarmChat\finetuned models\Falcon3-10B-Instruct-BehaviorTree-3epochs.Q4_K_M.gguf"

llm = Llama(model_path=model_path, n_ctx=1024*4)

def call_behaviors() -> dict:
    """
    Reflectively collect all public action and condition methods from SwarmAgent.

    Scans the SwarmAgent class for callable attributes whose names do not start
    with an underscore and are not lifecycle or obstacle checks. For each such
    method, extracts its docstring, normalizes whitespace into a single line,
    and builds a dictionary mapping method names to their cleaned descriptions.

    Returns:
        dict: A mapping from behavior method name (str) to its one-line description (str).
              Methods without a docstring will have an empty string as their description.
    """
    
    behavior_dict = {}
    for name, attribute in SwarmAgent.__dict__.items():
        if callable(attribute) and not name.startswith("_") \
            and not name.startswith("update") and not name.startswith("obstacle"):
            doc = attribute.__doc__
            if doc is not None:
                # Dedent, strip, and join into one line by replacing newlines and tabs
                cleaned_doc = " ".join(textwrap.dedent(doc).strip().split())
            else:
                cleaned_doc = ""
            behavior_dict[name] = cleaned_doc
    return behavior_dict

def extract_behavior_tree(response: str) -> str:
    """
    Extracts an XML behavior tree from the given response text.
    Looks for a block of XML enclosed in <root...</root> tags.
    """
    pattern = re.compile(r'(<root.*?</root>)', re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    else:
        # If no valid XML block is found, return the original response.
        return response.strip()

def save_behavior_tree(tree_xml: str, file_name: str = "tree.xml") -> None:
    """
    Saves the behavior tree XML to a file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(tree_xml)


def construct_prompt(prompt: str, prompt_type: str="one") -> str:

    behaviors = call_behaviors()
    behaviors_text = "\n".join(f"{name}: {doc}" for name, doc in behaviors.items())

    plan_prompt = f"""
    <s>
    <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
    INSTRUCTIONS: It is CRITICAL to use only the following behaviors structured as a dictionary: {behaviors_text} to construct behavior tree in XML format for the following command. Including any behavior that is not in the provided dictionary can result in damage to the agents and potentially humans, therefore you are not allowed to do so. AVOID AT ALL COSTS.
    USER COMMAND: generate behavior tree to "{prompt}". Take a step back and think deeply about the behavior you need for this command. Consider the XML structure and the behaviors you use.
    The output MUST follow this XML structure exactly, including:
    - A root element with <root BTCPP_format and main_tree_to_execute attributes.
    - A <BehaviorTree> element with an inner structure of Sequences, Fallback, Conditions, and Actions.
    - A <TreeNodesModel> section listing all node models.
    - No additional text or commentary outside the XML.
    Output only the XML behavior tree without extra text.
    OUTPUT:
    """

    if prompt_type == "zero":
        return plan_prompt
    elif prompt_type == "one":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\prompt_types\One_shot.txt"
        with open(path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return f"{file_content} {plan_prompt}"
    elif prompt_type == "two":
        path = r"C:\Users\moham\Desktop\SwarmChat_github\SwarmChat\prompt_types\Two Shot.txt"
        with open(path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return f"{file_content} {plan_prompt}"
    else:
        raise ValueError("Unknown prompt type provided.")
    

def generate_behavior_tree(task_prompt: str) -> str:

    prompt = construct_prompt(task_prompt)

    # llm = Llama(model_path=model_path, n_ctx=1024*4)

    print("\n\n",prompt,"\n\n")

    output = llm(
        prompt,
        temperature=0,
        max_tokens=1024,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.1
    )
    response = output.get("choices", [{}])[0].get("text", "").strip()
    tree_xml = extract_behavior_tree(response)
    save_behavior_tree(tree_xml)
    print("\n response: \n", response)
    return tree_xml


# Example usage:
if __name__ == "__main__":
    task = "Generate a behavior tree to just form a line."
    response = generate_behavior_tree(task)
    print("Generated behavior tree response:")
    print(response)
