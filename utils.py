"""
Script containing functions for generating and verifying NN code requirements.
"""


import ast
import re
import argparse
from itertools import product
import importlib.util
from pathlib import Path
import torch
from torch.fx import symbolic_trace
from torch import nn
import config



def load_and_parse_model_class(file_path: str):
    """
    Load a Python file, parse it into an AST, and return the first class
    that is a subclass of nn.Module along with the AST tree.
    Parameters:
        file_path (str): Path to the Python file.

    Returns:
        ast.ClassDef | None: The first class subclassing `nn.Module`, 
            or None if not found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (
                    isinstance(base, ast.Attribute)
                    and base.attr == "Module"
                ) or (
                    isinstance(base, ast.Name)
                    and base.id == "Module"
                ):
                    return node

    return None


def import_class_from_file(file_path: str, class_name: str):
    """
    Dynamically import a class from a Python file.

    Parameters:
        file_path (str | Path): Path to the Python file.
        class_name (str): Name of the class to import.

    Returns:
        type: The imported class object.
    """

    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def symbolic_check(model_class: type[nn.Module], input_shape: tuple,
                   discrete: bool = False, high: int = 10):
    """
    Perform symbolic tracing on a model class using a dummy input.

    Parameters:
        model_class (type[nn.Module]): The model class to trace.
        input_shape (tuple | None): Shape of the input.
        discrete (bool, optional): Whether input should be integer
            tokens. Defaults to False. Used for text data.
        high (int, optional): Maximum value for discrete inputs.
            Defaults to 10. Used for text data.

    Returns:
        bool: True if symbolic tracing succeeds, False otherwise.
    """

    if input_shape is None:
        return False
    try:
        model_instance = model_class()
        if discrete:  # for text token IDs
            dummy_input = torch.randint(low=0, high=high, size=input_shape)
        else:  # for images, tabular, timeseries
            dummy_input = torch.empty(input_shape, device='meta')
        traced = symbolic_trace(model_instance)
        traced(dummy_input)
        return True
    except (TypeError, RuntimeError, ValueError):
        return False

def get_first_layer(layers: list, arch: str):
    """
    Return the first relevant layer from a list of layers 
        based on the architecture.

    Parameters:
        layers (list): List of layer dictionaries with a "type" key.
        arch (str): Model architecture name (e.g., "MLP", "CNN-2D", "RNN").

    Returns:
        dict | None: The first matching layer dictionary,
            or None if not found.
    """
    batch_norm = config.BATCH_NORM
    emb = "Embedding"

    def is_relevant(l):
        t = l["type"]
        return (
            t not in batch_norm
            and (
                (arch.startswith("MLP") and t == "Linear")
                or (arch.startswith("CNN-2D") and t == "Conv2d")
                or (arch.startswith("CNN-3D") and t == "Conv3d")
                or (arch.startswith("RNN") and t in ["RNN", "LSTM", "GRU"])
                or (arch.startswith("CNN-1D") and t == "Conv1d")
                or ((not arch.startswith(("CNN-2D", "CNN-3D"))) and t == emb)
            )
        )

    return next((l for l in layers if is_relevant(l)), None)



def check_tabular_input(model_class: type[nn.Module], layers: list,
                        input_desc: str, architecture: str):
    """
    Validate a tabular input against the NN first layer and
        input description.

    Parameters:
        model_class (type[nn.Module]): The PyTorch model class to test.
        layers (list): List of layer dictionaries.
        input_desc (str): Input requirement.
        architecture (str): Model architecture requirement 
            (e.g., "MLP", "CNN-2D").

    Returns:
        dict: A dictionary indicating whether the input type and scale
            match, or describing the mismatch.
    """
    first_layer = get_first_layer(layers, architecture)
    key_return = "Input Type + Scale"
    if first_layer["type"] != "Linear":
        return {key_return: "Mismatch (First layer is not linear)"}
    in_feat_layer = first_layer["constants"][0]
    input_dim = parse_input_desc(input_desc)["features"]

    if input_dim:
        if not input_dim[0] <= in_feat_layer <= input_dim[1]:
            msg = f"(in_features={in_feat_layer} outside required range)"
            return {key_return: f"Mismatch: {msg}"}

    try:
        symbolic_check(model_class, (2, in_feat_layer))
        return {key_return: "Match"}
    except (TypeError, RuntimeError, ValueError) as e:
        return {key_return: f"Mismatch (symbolic trace failed: {e})"}



def check_text_input(layers: list, model_class: type[nn.Module],
                     input_desc: str):
    """
    Validate text input compatibility for a model.

    Checks that the first layer is an Embedding and that its 
    vocabulary size falls within the required range. It also 
    runs symbolic tracing for sequence length checks.

    Parameters:
        layers (list): List of layer dictionaries with layer details.
        model_class (type[nn.Module]): The PyTorch model class to test.
        input_desc (str): Input requirement.

    Returns:
        dict: A dictionary indicating whether the input type and scale
            match, or describing the mismatch.
    """
    requirements = parse_input_desc(input_desc)
    first_layer = get_first_input_layer_text(layers)
    key_return = "Input Type + Scale"
    if first_layer is None:
        return {key_return: "Mismatch (no Embedding layer found)"}

    vocab_range = requirements.get("vocab_size")
    if vocab_range:
        lyr_emb = first_layer["num_embeddings"]
        if not vocab_range[0] <= lyr_emb <= vocab_range[1]:
            msg = f"(Embedding vocab={lyr_emb} outside required range)"
            return {key_return: f"Mismatch {msg}"}

    try:
        seq_range = requirements.get("seq_length")
        seq_range = [int(d) for d in seq_range if (d!=1 and d!=float('inf'))]
        assert len(seq_range)>0
        for value in seq_range:
            symbolic_check(
                model_class, (2, value), discrete=True, high=lyr_emb
            )
        return {key_return: "Match"}
    except (TypeError, RuntimeError, ValueError) as e:
        return {key_return: f"Mismatch (symbolic trace failed: {e})"}


def get_first_input_layer_text(layers: list):
    """
    Return the first Embedding layer if present.

    Parameters:
        layers (list): List of layer dictionaries with layer details.

    Returns:
        dict | None: Dictionary with Embedding layer info,
            or None if not found.
    """
    if layers[0]["type"] == "Embedding" and len(layers[0]["constants"]) >= 2:
        num_embeddings, embedding_dim = layers[0]["constants"][:2]
        return {
            "layer_type": "Embedding",
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "comment": "Embedding layer"
        }
    return None


def check_image_input(model_class: type[nn.Module], input_desc: str,
                      architecture: str):
    """
    Validate image input compatibility for a model using symbolic tracing.

    Parameters:
        model_class (type[nn.Module]): The PyTorch model class to test.
        input_desc (str): Input requirement.
        architecture (str): Architecture requirement.

    Returns:
        dict: A dictionary indicating whether the input type and scale match,
            or describing the mismatch.
    """

    in_dim = parse_input_desc(input_desc)["resolution"]
    key_return = "Input Type + Scale"
    try:
        input_dim = [int(d) for d in in_dim if (d != 1 and d != float('inf'))]
        assert len(input_dim) > 0

        for dim in input_dim:
            if architecture == "CNN-2D":
                symbolic_check(model_class, (2, 3, dim, dim))
            elif architecture == "CNN-3D":
                symbolic_check(model_class, (2, 3, dim, dim, dim))
            else:
                msg = "Mismatch (Architecture is neither CNN-2D nor CNN-3D)"
                return {key_return: msg}

        return {key_return: "Match"}
    except (TypeError, RuntimeError, ValueError) as e:
        return {key_return: f"Mismatch (symbolic trace failed: {e})"}


def check_timeseries_input(layers: list, model_class: type[nn.Module],
                           input_desc: str):
    """
    Validate time series input compatibility for a model.

    Checks that the first input-related layer matches the feature scale and
        performs a symbolic tracing run.

    Parameters:
        layers (list): List of layer dictionaries with layer details.
        model_class (type[nn.Module]): The PyTorch model class to test.
        input_desc (str): Input requirement.

    Returns:
        dict: A dictionary indicating whether the input type and scale match,
            or describing the mismatch.
    """

    requirements = parse_input_desc(input_desc)
    first_layer = get_first_input_layer_timeseries(layers)
    key_return = "Input Type + Scale"
    if not first_layer:
        return {key_return: "Mismatch (no input-related layer found)"}

    features = requirements.get("features")
    if features:
        if not features[0] <= first_layer["in_dim"] <= features[1]:
            msg = f"(Features={first_layer['in_dim']} outside required range)"
            return {key_return: f"Mismatch {msg}"}
    try:
        seq_range = requirements.get("seq_length")
        seq_range = [int(d) for d in seq_range if (d!=1 and d!=float('inf'))]
        assert len(seq_range)>0
        input_shape = None
        for value in seq_range:
            flyr_type = first_layer["layer_type"]
            if flyr_type == "Conv1d":
                in_channels = first_layer["in_dim"]
                input_shape = (2, in_channels, value)
            elif flyr_type in ["RNN", "LSTM", "GRU", "Linear"]:
                input_size = first_layer["in_dim"]
                input_shape = (2, value, input_size)
            symbolic_check(model_class, input_shape)
        return {key_return: "Match"}
    except (TypeError, RuntimeError, ValueError) as e:
        return {key_return: f"Mismatch (symbolic trace failed: {e})"}


def get_first_input_layer_timeseries(layers: list):
    """
    Return the first time series-related input layer if present.

    Parameters:
        layers (list): List of layer dictionaries with layer details.

    Returns:
        dict | None: Dictionary with layer info for Conv1d or RNN-type layers,
                     or None if no suitable input layer is found.
    """
    rnns = ["RNN", "LSTM", "GRU"]
    for layer in layers:
        if layer["type"] == "Conv1d" and len(layer["constants"]) >= 2:
            return {
                "layer_type": "Conv1d",
                "in_dim": layer["constants"][0],
                "out_channels": layer["constants"][1]
            }
        elif layer["type"] in rnns and len(layer["constants"]) >= 2:
            return {
                "layer_type": layer["type"],
                "in_dim": layer["constants"][0],
                "hidden": layer["constants"][1]
            }
        elif layer["type"] == "Linear":
            return None

    return None


def extract_layer_range(target_complexity: str):
    """
    Extract numeric minimum and maximum layer depth values from 
    target_complexity.

    Parameters:
        desc (str): Layer description string.

    Returns:
        tuple[float | None, float | None]: (min_value, max_value) extracted 
            from target_complexity, or (None, None) if not found.
    """
    if "layers" in target_complexity:
        part = target_complexity.split("layers", 1)[1]
    else:
        part = target_complexity
    part = part.strip()

    numbers = list(map(int, re.findall(r"\d+", part)))

    if "up to" in part and numbers:
        return (1, numbers[-1])
    elif "at least" in part and numbers:
        return (numbers[0], float("inf"))
    elif "between" in part and len(numbers) >= 2:
        return (numbers[0], numbers[1])
    else:
        return (None, None)

def calculate_threshold_complexity(first_layer_type: str, value_input: int,
                                   input_desc: str):
    """
    Compute the width complexity threshold value based on the first 
    layer and input.

    Parameters:
        first_layer_type (str): Type of the first layer.
        value_input (int): upper bound or lower bound of input 
            (features, vocab size, etc.).
        input_desc (str): Input requirement.

    Returns:
        int: Calculated threshold value for width complexity.
    """
    if "Text" in input_desc: #either conv1d or rnns
        threshold = min(128, value_input)
    elif first_layer_type == "Linear":
        threshold = min(128, value_input)
    elif first_layer_type.startswith("Conv1d"):
        threshold = min(128, value_input)
    elif first_layer_type.startswith(("Conv2d", "Conv3d")):
        threshold = min(8, value_input // 8)
    elif first_layer_type.startswith(("RNN", "LSTM", "GRU")):
        threshold = min(128, value_input)
    else:
        print("Threshold could not be calculated")
        threshold = value_input
    return threshold


def parse_input_desc(input_desc: str):
    """
    Parse the input description string into numeric ranges.

    Parameters:
        input_desc (str): Input requirement.

    Returns:
        dict: Dictionary mapping field names ("features", "seq_length",
              "vocab_size", "resolution") to numeric ranges as 
              lists [min, max].
    """
    desc = input_desc.lower()
    result = {}

    field_names = ["features", "seq_length", "vocab_size", "resolution"]
    for field in field_names:
        m = re.search(
            rf"{field}\s*:\s*([^,]+?)(?:\s+and|$)", desc, flags=re.IGNORECASE
        )
        if not m:
            continue
        value_str = m.group(1).strip()
        tokens = re.findall(r"[\d\.]+[kK]?", value_str)
        nums = []
        for t in tokens:
            if t.lower().endswith("k"):
                nums.append(int(float(t[:-1]) * 1000))
            else:
                nums.append(int(t))

        if "<" in value_str and nums:
            result[field] = [1, nums[0]-1]
        elif ">" in value_str and nums:
            result[field] = [nums[0]+1, float('inf')]
        elif "-" in value_str and len(nums) >= 2:
            result[field] = [nums[0], nums[1]]

    return result


def parse_input_for_file_name(input_desc: str):
    """
    Convert an input description into a normalized key like 
    'text_small_small' to use for the NN file name.

    Parameters:
        input_desc (str): Input requirement.

    Returns:
        str: Normalized key, e.g., "text_small_small".
    """
    parts = input_desc.split(",", 1)
    first = parts[0].strip().replace(" ", "-")

    keywords = []
    if len(parts) > 1:
        # Find keywords Small / Medium / Large / Extra Large (ignore numbers)
        matches = re.findall(r"\b(?:Small|Medium|Large|Extra\s+Large)\b",
                             parts[1], flags=re.IGNORECASE)
        keywords = [m.replace(" ", "-") for m in matches]

    name = "-".join([first] + keywords)
    return name


def construct_input_statements(input_dict: dict):
    """
    Generate the list of input description strings from a nested
    input dictionary.

    Parameters:
    input_dict (dict): Dictionary where keys are input categories
        (e.g., "Tabular") and values are dictionaries mapping
        field names to possible values 
        (e.g., {"features": ["Small <50", "Large >2000"]}).

    Returns:
    list[str]: List of combined input description strings for
        all value combinations.
    """
    inputs = []
    for category, subdict in input_dict.items():
        keys = list(subdict.keys())
        values = list(subdict.values())
        for combo in product(*values):
            parts = [f"{k}: {v}" for k, v in zip(keys, combo)]
            inputs.append(f"{category}, " + " and ".join(parts))
    return inputs

def parse_args_requirements():
    """
    Parse command-line arguments for NN requirement processing.

    Returns:
    argparse.Namespace: Parsed arguments containing task, input, 
        architecture, complexity, and file name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the NN requirements."
    )
    parser.add_argument("task", type=str, help="Task")
    parser.add_argument("input", type=str, help="Input")
    parser.add_argument("architecture", type=str, help="Architecture")
    parser.add_argument("complexity", type=str, help="Complexity")
    parser.add_argument("fname", type=str, help="File name")
    return parser.parse_args()

def is_functional_call(node: ast.AST):
    """
    Check if an AST node is a call to a functional API (e.g., torch or F).

    Parameters:
    node (ast.AST): The AST node to check.

    Returns:
    bool: True if the node is a function call to F or torch, False otherwise.
    """
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    is_attr = isinstance(f, ast.Attribute)
    is_name = isinstance(f.value, ast.Name) if is_attr else False
    is_func = f.value.id in {"F", "torch"} if is_name else False
    return is_attr and is_name and is_func


def find_self_calls(node: ast.AST, ordered_layers: list, layer_defs: dict):
    """
    Recursively traverse an AST node to find and collect calls.

    Parameters:
    node (ast.AST): The AST node to inspect.
    ordered_layers (list): List to append found layers in the order
        they appear.
    layer_defs (dict): Dictionary mapping attribute names to layer 
        definitions.

    Returns:
    None: The function updates ordered_layers in place.
    """
    if isinstance(node, ast.Call):
        for arg in node.args:
            find_self_calls(arg, ordered_layers, layer_defs)
        for kw in node.keywords:
            find_self_calls(kw.value, ordered_layers, layer_defs)
        if is_functional_call(node):
            ordered_layers.append(
                {"type": node.func.attr, "constants": []}
            )
        elif isinstance(node.func, ast.Attribute):
            func_val = node.func.value
            if isinstance(func_val, ast.Call):
                find_self_calls(func_val, ordered_layers, layer_defs)
                return
            if isinstance(func_val, ast.Name) and func_val.id == "self":
                attr = node.func.attr
                layers = layer_defs.get(attr)
                if layers:
                    ordered_layers.extend(l.copy() for l in layers)
            return
    for child in ast.iter_child_nodes(node):
        find_self_calls(child, ordered_layers, layer_defs)


def parse_layer(call_node: ast.Call):
    """
    Parse an AST Call node representing a layer and extract its type
    and constants.

    Parameters:
    call_node (ast.Call): The AST node representing the layer call.

    Returns:
    list[dict]: A list of dictionaries with keys "type" (layer type) and
    "constants" (list of numeric constant arguments). Handles
    nested Sequential layers recursively.
    """
    func = call_node.func
    if isinstance(func, ast.Attribute):
        layer_type = func.attr
    elif isinstance(func, ast.Name):
        layer_type = func.id
    else:
        return []
    if layer_type == "Sequential":
        inner = []
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                inner.extend(parse_layer(arg))
        return inner
    args = call_node.args
    constants = [a.value for a in args if isinstance(a, ast.Constant)]
    constants += [
        kw.value.value
        for kw in call_node.keywords
        if isinstance(kw.value, ast.Constant)
    ]
    return [{"type": layer_type, "constants": constants}]

def extract_layers(class_node: ast.ClassDef):
    """
    Parse the NN AST to extract the list of ordered layers.

    Parameters:
    class_node (ast.ClassDef): AST node representing the NN class definition.

    Returns:
    layers (list): List of layer dictionaries with layer details.
    """

    layer_defs = {}
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            for sub in stmt.body:
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.value, ast.Call)
                ):
                    tgt = sub.targets[0]
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        name = tgt.attr
                        layer_defs[name] = parse_layer(sub.value)
    ordered_layers = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "forward":
            for node in stmt.body:
                find_self_calls(node, ordered_layers, layer_defs)

    return ordered_layers


def get_last_layer_for_task(layers: list, task: str):
    """
    Return the last relevant layer for a task along with its output dimension.

    Parameters:
    layers (list): List of layer dictionaries with layer details.
    task (str): Task requirement.

    Returns:
    dict | None: Dictionary with keys "layer_type" and "out_dim" of the last
    matching layer, or None if no suitable layer is found.
    """
    lyr_types = ["Linear", "RNN", "LSTM", "GRU", "Conv1d", "Conv2d", "Conv3d"]
    representation = None
    if "representation" in task.lower():
        representation = True
    for layer in list(reversed(layers)):
        if len(layer["constants"]) >= 2:
            out_dim = layer["constants"][1]

            if (not representation) and (layer["type"] == "Linear"):
                return {"layer_type": layer["type"], "out_dim": out_dim}
            if representation and layer["type"] in lyr_types:
                return {"layer_type": layer["type"], "out_dim": out_dim}
    return None

def format_range(min_v, max_v, label):
    """
    Format a numeric range into a human-readable string with a label.

    Parameters:
    min_v (str | float): Minimum value of the range.
    max_v (str | float): Maximum value of the range.
    label (str): Label describing the range.

    Returns:
    str: Formatted range string, e.g., "features between 0 and 50".
    """

    min_str = str(min_v) if isinstance(min_v, str) else str(int(min_v))
    if isinstance(max_v, str):
        max_str = max_v
    elif max_v == float("inf"):
        max_str = None
    else:
        max_str = str(int(max_v))

    if max_str is None:
        return f"{label} at least {min_str}"
    elif min_str == "0" or min_str == "1":
        return f"{label} up to {max_str}"
    else:
        return f"{label} between {min_str} and {max_str}"

def generate_complexity_def(arch_type: str, input_desc: str):
    """
    Generate a dictionary mapping complexity levels to descriptive sentences.

    Parameters:
    arch_type (str): Architecture type (e.g., "MLP", "CNN-2D", "RNN-LSTM").
    complexity_rules (dict): Rules defining min/max depth and width values.
    input_desc (str): Input requirement.

    Returns:
    dict | None: Dictionary where keys are complexity levels and values are
    formatted strings describing numeric ranges for depths and widths,
    or None if no rules exist for the given architecture.
    """

    rules = config.COMPLEXITY_RULES.get(arch_type)
    if not rules:
        return None

    # Determine label for width based on architecture
    width_label = None
    if "Text" in input_desc:
        width_label = "embedding_dim of the Embedding layer"
    elif arch_type == "MLP":
        width_label = "out_features of first linear layer"
    elif arch_type.startswith(("CNN-1D", "CNN-2D", "CNN-3D")):
        width_label = "out_channels of first Conv layer"
    elif arch_type in ["RNN-Simple", "RNN-LSTM", "RNN-GRU"]:
        width_label = f"hidden_size of first {arch_type} layer"

    lyr_type = "linear" if arch_type == "MLP" else arch_type
    if "Text" in input_desc:
        embed_rule = config.COMPLEXITY_RULES.get("Embedding")
        for level in rules:
            rules[level]["min_val"] = embed_rule[level]["min_val"]
            rules[level]["max_val"] = embed_rule[level]["max_val"]
    result = {}
    for level, rule in rules.items():
        min_layers, max_layers = rule["min_layers"], rule["max_layers"]
        min_val = rule.get("min_val", 1)
        max_val = rule.get("max_val", float("inf"))
        parts = []
        if level in {"Simple", "Wide-Deep", "Deep"}:
            parts.append(format_range(
                min_layers, max_layers, f'Number of {lyr_type} layers'
            ))
        if level in {"Simple", "Wide-Deep", "Wide"}:
            parts.append(format_range(min_val, max_val, width_label))

        desc = ", ".join(parts) + "." if parts else "Unknown"
        result[level] = desc
    return result

def import_nn_from_file(file_path: str):
    """
    Import a Python file and return the first class that is 
    a subclass of nn.Module.

    Parameters:
        file_path (str): Path to the Python file.

    Returns:
        type: The first class subclassing `nn.Module`.
    """
    spec = importlib.util.spec_from_file_location("model_module", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nn_classes = [cls for cls in mod.__dict__.values()
                  if isinstance(cls, type) and issubclass(cls, nn.Module)]
    if not nn_classes:
        raise ValueError("No nn.Module subclass found in file.")
    return nn_classes[0]()

def extract_layer_and_func_types(model: nn.Module):
    """
    Extract layers and functional operations used in a given NN model.

    Parameters:
        model (nn.Module): The NN to analyze.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - layers: Names of all layers in the NN.
            - tensor_ops: Names of tensor-level operations.
    """

    traced = symbolic_trace(model)
    layers = []
    tensor_ops = []

    for node in traced.graph.nodes:
        if node.op == 'call_module':
            submod = dict(model.named_modules())[node.target]
            name = type(submod).__name__
            if name != "Identity":
                if hasattr(submod, "num_layers"):
                    layers.extend([name] * submod.num_layers)
                else:
                    layers.append(name)
        elif node.op == 'call_function':
            filtered = {'getattr', 'getitem', 'add', 'truediv', 'ne'}
            if node.target.__name__ not in filtered:
                tensor_ops.append(node.target.__name__)
    return layers, tensor_ops
