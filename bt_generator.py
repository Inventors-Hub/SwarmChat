from simulator_env import SwarmAgent
import textwrap
import re
from llama_cpp import Llama

# Load the Llama model for safety classification
model_path = r"G:\Inventors Hub Projects\SwarmChat\model\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=1024*4)

def call_behaviors() -> dict:
    behavior_dict = {}
    for name, attribute in SwarmAgent.__dict__.items():
        if callable(attribute) and not name.startswith("_") \
            and not name.startswith("update") \
            and not name.startswith("obstacle"):
            doc = attribute.__doc__
            if doc is not None:
                # Remove common indentation and extra whitespace/newlines
                cleaned_doc = textwrap.dedent(doc).strip()
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
    
    # XML template as expected output example.
    xml_template = """<?xml version="1.0" encoding="UTF-8"?>
    <root BTCPP_format="3" main_tree_to_execute="FindAndFlock">
    <BehaviorTree ID="FindAndFlock">
        <Sequence name="Main Sequence">
        <Fallback name="Find Goal">
            <Sequence>
            <is_target_detected/>
            <change_color color="red"/>
            <flocking/>
            </Sequence>
            <wander/>
        </Fallback>
        <!-- Subtree call to handle returning to nest -->
        <Sequence name="Return To Nest">
            <is_agent_in_nest/>
            <SubTree ID="BackToNest" __shared_blackboard="true"/>
        </Sequence>
        </Sequence>
    </BehaviorTree>

    <!-- Subtree definition for returning to nest -->
    <BehaviorTree ID="BackToNest">
        <Sequence name="BackToNest Sequence">
        <change_color color="green"/>
        </Sequence>
    </BehaviorTree>

    <TreeNodesModel>
        <Condition ID="is_target_detected"/>
        <Action ID="wander"/>
        <Action ID="flocking"/>
        <Action ID="change_color" editable="true">
        <input_port name="color"/>
        </Action>
        <Condition ID="is_agent_in_nest"/>
        <SubTree ID="BackToNest"/>
    </TreeNodesModel>
    </root>
    """
        
    # Construct a plain prompt that instructs the model to generate XML strictly following the example.
    plain_prompt = f"""
    Generate a behavior tree in XML format for the following task:
    "{task_prompt}"

    Use only the following behaviors (do not invent any new ones):
    {behaviors_text}

    The output MUST follow this XML structure exactly, including:
    - A root element with BTCPP_format and main_tree_to_execute attributes.
    - A <BehaviorTree> element with an inner structure of Sequences, Fallback, Conditions, and Actions.
    - A <TreeNodesModel> section listing all node models.
    - No additional text or commentary outside the XML.

    Example XML Format:
    {xml_template}

    Please output only the XML.
    Response:
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
