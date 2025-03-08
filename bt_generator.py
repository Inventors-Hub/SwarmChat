from simulator_env import SwarmAgent
import textwrap
import re
from llama_cpp import Llama

# Load the Llama model for safety classification
model_path = r"G:\Inventors Hub Projects\SwarmChat\model\DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=1024*4)

def call_behaviors() -> dict:
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

def generate_behavior_tree(task_prompt: str) -> str:
    """
    Generates a behavior tree for the provided task prompt.
    It gathers available behaviors, constructs an informative prompt including an XML template,
    calls the Llama model, extracts the behavior tree XML, saves it, and returns the response.
    """
    behaviors = call_behaviors()
    behaviors_text = "\n".join(f"{name}: {doc}" for name, doc in behaviors.items())
           
    # Construct a plain prompt that instructs the model to generate XML strictly following the example.
    plain_prompt = f"""
    <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
    INSTRUCTIONS: It is CRITICAL to use only the following behaviors structured as a dictionary: {behaviors_text} to construct behavior tree in XML format to the following command, including in the behaviour tree a behaviour that is not in the provided dictionary can result in damage to the agents, and potentially humans, therefore you are not allowed to do so, AVOID AT ALL COSTS.
    USER COMMAND: generate behavior tree to "{task_prompt}". Take a step back and think deeply about the behavior you need for this command. Take another step back and think of the xml structure and the behavior you used.
    The output MUST follow this XML structure exactly, including:
    - A root element with BTCPP_format and main_tree_to_execute attributes.
    - A <BehaviorTree> element with an inner structure of Sequences, Fallback, Conditions, and Actions.
    - A <TreeNodesModel> section listing all node models.
    - No additional text or commentary outside the XML.
    Output only the XML behavior tree without extra text.
    OUTPUT:
    """

    print(plain_prompt)

    output = llm(
        plain_prompt,
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
    return response


# Example usage:
if __name__ == "__main__":
    task = "Generate a behavior tree to just form a line."
    response = generate_behavior_tree(task)
    print("Generated behavior tree response:")
    print(response)
