import xml.etree.ElementTree as ET
from typing import List, Dict
import py_trees as pt
# from simulator_env import SwarmAgent

########################################################################
# 1. XML Parsing Classes and Functions
########################################################################

class Node:
    """
    A generic node representing a behavior tree element.
    It holds:
      - tag: the element's tag (e.g., "Sequence", "say", "SubTree", etc.)
      - attributes: a dict of the element's attributes (e.g., name, num_cycles, port values)
      - children: a list of child Node instances (which may be other behaviors or sub-elements)
      - ports: a dict grouping any port definitions found as child elements (input_port, output_port, inout_port)
    """
    def __init__(self, tag: str, attributes: Dict[str, str]):
        self.tag = tag
        self.attributes = attributes.copy()
        self.children: List['Node'] = []
        self.ports: Dict[str, List[Dict[str, str]]] = {}

    def __repr__(self):
        return (f"Node(tag={self.tag!r}, attributes={self.attributes!r}, "
                f"children={self.children!r}, ports={self.ports!r})")


def parse_node(element: ET.Element) -> Node:
    """
    Recursively parse an XML element into a Node.
    This function:
      - Reads the element's tag and attributes.
      - Checks for child elements that define ports (input_port, output_port, inout_port) and stores them.
      - Recursively parses any other child elements as behavior nodes.
    """
    node = Node(element.tag, element.attrib)

    for child in element:
        # Check if the child defines a port (this covers the new "inout_port" as well)
        if child.tag in ['input_port', 'output_port', 'inout_port']:
            if child.tag not in node.ports:
                node.ports[child.tag] = []
            node.ports[child.tag].append(child.attrib)
        else:
            # Otherwise, treat the child as a regular behavior node.
            child_node = parse_node(child)
            node.children.append(child_node)
    
    return node


def parse_behavior_trees(xml_file: str) -> List[Node]:
    """
    Parses the given XML file and returns a list of BehaviorTree nodes.
    Each <BehaviorTree> element is considered a complete behavior tree (or subtree).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    behavior_trees = []
    for bt_elem in root.findall('BehaviorTree'):
        bt_node = parse_node(bt_elem)
        behavior_trees.append(bt_node)
    return behavior_trees

########################################################################
# 2. Functions that will be executed by the BT (your actions, conditions, etc.)
########################################################################


def get_function_mapping():
    from simulator_env import SwarmAgent
    mapping = {
        name: func
        for name, func in SwarmAgent.__dict__.items()
        if callable(func) and not name.startswith("_") and name not in ['update','_inject_agent','obstacle','_speak']
    }
    # print("mapping: \n", mapping)
    return mapping

########################################################################
# 3. Helpers and Custom py_trees Behavior Wrappers
########################################################################

def convert_param(val: str):
    """
    Attempt to convert a string parameter to int or float if possible.
    Otherwise, return the string.
    """
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val




# A simple leaf node that wraps a function call.
class FunctionAction(pt.behaviour.Behaviour):
    def __init__(self, name, function, params):
        super(FunctionAction, self).__init__(name=name)
        self.function = function
        self.params = params
        self.agent = None  # Will be set later

    def update(self):
        # Pass the agent (context) into the function
        status = self.function(self.agent, **self.params)
        return status

# A decorator node that wraps a child behavior and calls a function.
class FunctionDecorator(pt.decorators.Decorator):
    def __init__(self, name, function, params, child):
        super(FunctionDecorator, self).__init__(name=name, child=child)
        self.function = function
        self.params = params

    def update(self):
        # Ensure the child is updated.
        self.decorated.tick_once()
        child_status = self.decorated.status
        # Call the decorator function (for side effects)
        self.function(**self.params)
        # For this example, we simply pass through the child's status.
        return child_status

# A control node that has one child and then calls a function.
class FunctionControl(pt.behaviour.Behaviour):
    def __init__(self, name, function, params, child):
        super(FunctionControl, self).__init__(name=name)
        self.function = function
        self.params = params
        self.child = child

    def update(self):
        self.child.tick_once()
        return self.function(**self.params)

# Define an AlwaysSuccess behavior to use when an unknown node is encountered.
class AlwaysSuccess(pt.behaviour.Behaviour):
    def __init__(self, name="AlwaysSuccess"):
        super(AlwaysSuccess, self).__init__(name=name)
    
    def update(self):
        return pt.common.Status.SUCCESS

########################################################################
# 4. Convert the Parsed Node Tree into a py_trees Behavior Tree
########################################################################

def build_behavior(node: Node, subtree_mapping: Dict[str, Node]) -> pt.behaviour.Behaviour:
    """
    Recursively converts a parsed Node (from XML) into a py_trees behavior.
    """
    # Special case: unwrap the BehaviorTree container.
    if node.tag == "BehaviorTree":
        if node.children:
            return build_behavior(node.children[0], subtree_mapping)
        else:
            return AlwaysSuccess(name="Empty BehaviorTree")
    
    # Define which tags represent which kinds of nodes.
    composite_tags = ["Sequence", "Fallback"]
    repeat_tags = ["Repeat"]
    decorator_tags = ['testDecorator',"Inverter", "Success", "Failure", "AlwaysFailure", "AlwaysSuccess", "ForceSuccess", "ForceFailure"]
    control_tags = ["test1"]

    mapping = get_function_mapping()


    if node.tag == "Sequence":
        composite = pt.composites.Sequence(
            name=node.attributes.get('name', 'Sequence'),
            memory=True  # Added memory parameter
        )
        for child in node.children:
            composite.add_child(build_behavior(child, subtree_mapping))
        return composite

    elif node.tag == "Fallback":
        composite = pt.composites.Selector(
            name=node.attributes.get('name', 'Fallback'),
            memory=True  # Added memory parameter
        )
        for child in node.children:
            composite.add_child(build_behavior(child, subtree_mapping))
        return composite

    elif node.tag in repeat_tags:
        if len(node.children) != 1:
            print("Repeat node must have exactly one child!")
        child_behavior = build_behavior(node.children[0], subtree_mapping)
        # Read the number of cycles from the XML; default to 1 if not provided.
        num_cycles = int(node.attributes.get('num_cycles', 1))
        # Create the Repeat decorator, providing the required 'num_success' parameter.
        repeat_decorator = pt.decorators.Repeat(
            name=node.attributes.get('name', 'Repeat'),
            child=child_behavior,
            num_success=num_cycles  # Provide the required parameter here.
        )
        return repeat_decorator


    elif node.tag == "SubTree":
        subtree_id = node.attributes.get('ID')
        if subtree_id in subtree_mapping:
            return build_behavior(subtree_mapping[subtree_id], subtree_mapping)
        else:
            print(f"SubTree with ID {subtree_id} not found!")
            return AlwaysSuccess(name="Missing SubTree")

    elif node.tag in decorator_tags:
        if len(node.children) != 1:
            print("Decorator node must have exactly one child!")
        child_behavior = build_behavior(node.children[0], subtree_mapping)
        params = {k: convert_param(v) for k, v in node.attributes.items() if k != "name"}
        return FunctionDecorator(
            name=node.attributes.get('name', node.tag),
            function=mapping[node.tag],
            params=params,
            child=child_behavior
        )

    elif node.tag in control_tags:
        if len(node.children) != 1:
            print("Control node must have exactly one child!")
        child_behavior = build_behavior(node.children[0], subtree_mapping)
        params = {k: convert_param(v) for k, v in node.attributes.items() if k != "name"}
        return FunctionControl(
            name=node.attributes.get('name', node.tag),
            function=mapping[node.tag],
            params=params,
            child=child_behavior
        )

    else:
        if node.tag in mapping:
            params = {k: convert_param(v) for k, v in node.attributes.items() if k != "name"}
            return FunctionAction(
                name=node.attributes.get('name', node.tag),
                function=mapping[node.tag],
                params=params
            )
        else:
            return AlwaysSuccess(name=node.attributes.get('name', node.tag))


########################################################################
# 5. Main: Parse XML, Build the py_trees Tree, and Execute It
########################################################################

def print_node(node, indent=0):
    ind = "  " * indent
    print(f"{ind}{node.tag}: {node.attributes}")
    # Optionally print any ports
    for port_type, port_list in node.ports.items():
        for port in port_list:
            print(f"{ind}  {port_type}: {port}")
    for child in node.children:
        print_node(child, indent + 1)

# Usage in your main:
if __name__ == "__main__":
    file_path = 'tree.xml'
    trees = parse_behavior_trees(file_path)
    for tree in trees:
        print_node(tree)

# if __name__ == "__main__":
#     # The XML file with your behavior tree.
#     file_path = 'tree.xml'
    
#     # 1. Parse the XML into a list of BehaviorTree nodes.
#     trees = parse_behavior_trees(file_path)
#     # Build a mapping of BehaviorTree ID to Node.
#     print(trees)
#     subtree_mapping = { tree.attributes.get("ID"): tree for tree in trees }
#     print()

#     print(subtree_mapping)
