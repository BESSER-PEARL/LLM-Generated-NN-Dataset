"""Script containing functions for verifying NN code requirements."""
from torch import nn
import utils
import config



def check_architecture(layers: list, expected_architecture: str):
    """
    Check if the extracted layers match the expected architecture.

    Parameters:
    layers (list): List of layer dictionaries with layer details.
    expected_architecture (str): Name of the expected architecture.

    Returns:
    dict: Dictionary with key "architecture" and value "Match" or "Mismatch"
    (or a message if the expected architecture is unknown).
    """

    expected_layer_types = config.ARCH_TO_LAYER.get(expected_architecture)
    if expected_layer_types is None:
        return {"architecture": "Mismatch (Unknown expected architecture)"}

    found = any(layer["type"] == expected_layer_types for layer in layers)
    return {"architecture": "Match" if found else "Mismatch"}


def check_task(layers: list, task: str):
    """
    Check if the model's last layer matches the task.

    Parameters:
    layers (list): List of layer dictionaries with layer details.
    task (str): Task requirement (e.g., "classification-binary")

    Returns:
    dict: Dictionary with key "task" and value "Match" or "Mismatch" (with
        message if no output layer is found).
    """

    last_layer_info = utils.get_last_layer_for_task(layers, task)
    if last_layer_info is None:
        return {"task": "Mismatch (no output layer found)"}

    out_dim = last_layer_info["out_dim"]
    task_lower = task.lower()

    match = False
    if task_lower == "classification-binary":
        match = out_dim in [1, 2]
    elif any(x in task_lower for x in ["multiclass", "representation"]):
        match = out_dim >= 3
    elif task_lower == "regression":
        match = out_dim == 1
    return {"task": "Match" if match else "Mismatch"}


def check_input(model_class: type[nn.Module], input_desc: str,
                layers: list, architecture: str):
    """
    Check the model input compatibility based on the input requirement.

    Parameters:
    model_class (type[nn.Module]): The PyTorch NN class.
    input_desc (str): Input requirement.
    layers (list): List of layer dictionaries with layer details.
    architecture (str): Architecture requirement.

    Returns:
    dict: Dictionary indicating whether the input type and scale match,
        or describing a mismatch.
    """

    if "image" in input_desc.lower():
        return utils.check_image_input(model_class, input_desc, architecture)
    elif "tabular" in input_desc.lower():
        return utils.check_tabular_input(
            model_class, layers, input_desc, architecture
        )
    elif "time series" in input_desc.lower():
        return utils.check_timeseries_input(layers, model_class, input_desc)
    elif "text" in input_desc.lower():
        return utils.check_text_input(layers, model_class, input_desc)
    else:
        return {"Input Type + Scale": "Mismatch"}


def compute_depth(layers: list, layer_type: str):
    """
    Compute the total depth of the model based on layer type.

    Parameters:
        layers (list): List of layer dictionaries with layer details.
        layer_type (str): The charactering layer of the NN type
            (e.g., 'Linear', 'Conv2d').

    Returns:
        int: Number of charactering layers of the NN.
    """

    depth = 0
    for l in layers:
        if l["type"] in ["RNN", "LSTM", "GRU"]:
            num_layers = l["constants"][2] if len(l["constants"]) > 2 else 1
            depth += num_layers
        elif l["type"] == layer_type:
            depth += 1
    return depth


def get_input_extremes(input_desc: str):
    """
    Extract numeric lower and upper bound from input description,
        ignoring sequence length.

    Parameters:
        input_desc (str): Input requirement string.

    Returns:
        list: List of numeric extremes for the first relevant input field.
    """

    in_extremes = utils.parse_input_desc(input_desc)
    return list(next(v for k, v in in_extremes.items() if k != "seq_length"))



def check_width(first_layer: dict, target_complexity: str,
                input_extremes: list, input_desc: str):
    """
    Check if the first layer's width satisfies the threshold for
    the target complexity.

    Parameters:
        first_layer (dict): Dictionary representing the first relevant layer.
        target_complexity (str): Complexity (e.g., 'Simple', 'Wide').
        input_extremes (list): List of lower bound and upper bound from
            input description.
        input_desc (str): Input requirement.

    Returns:
        bool or None: True if width satisfies the target complexity,
            None otherwise.
    """

    width_value = first_layer["constants"][1]
    width_ok = None
    if target_complexity.startswith("Simple"):
        extreme = input_extremes[1]
        threshold = utils.calculate_threshold_complexity(
            first_layer["type"], extreme, input_desc
        )
        if 1 <= width_value <= threshold:
            width_ok = True
    elif target_complexity.startswith("Wide"):
        extreme = input_extremes[0]
        threshold = utils.calculate_threshold_complexity(
            first_layer["type"], extreme, input_desc
        )
        if threshold <= width_value:
            width_ok = True
    return width_ok


def check_complexity(layers: list, arch_type: str, target_complexity: str,
                     input_desc: str):
    """
    Check compliance of the NN with the complexity requirement.

    Parameters:
        layers (list): List of layer dictionaries with layer details.
        arch_type (str): Architecture requirement (e.g., 'MLP', 'CNN-2D').
        target_complexity (str): Complexity (e.g., 'Simple', 'Wide', 'Deep').
        input_desc (str): Input requirement.

    Returns:
        dict: Dictionary with key 'Complexity' and value 'Match' or 'Mismatch' 
            with explanation if mismatched.
    """

    if arch_type not in config.ARCH_TO_LAYER:
        return {"Complexity": "Mismatch (Architecture type not supported)"}

    layer_type = config.ARCH_TO_LAYER[arch_type]
    first_layer = utils.get_first_layer(layers, arch_type)
    if not first_layer or len(first_layer.get("constants", [])) < 2:
        msg = "first layer params could not be retrieved"
        return {"Complexity": f"Mismatch ({msg})"}

    min_layers, max_layers = utils.extract_layer_range(target_complexity)
    depth = compute_depth(layers, layer_type)

    if target_complexity.startswith("Deep"):
        if min_layers <= depth <= max_layers:
            return {"Complexity": "Match"}
        else:
            msg = "Number of layers is inconsistent"
            return {"Complexity": f"Mismatch ({msg})"}

    input_extremes = get_input_extremes(input_desc)
    width_ok = check_width(first_layer, target_complexity, input_extremes,
                           input_desc)

    if width_ok:
        if target_complexity.startswith("Wide:"):
            return {"Complexity": "Match"}
        if min_layers <= depth <= max_layers:
            return {"Complexity": "Match"}
        else:
            msg = "Number of layers is inconsistent"
            return {"Complexity": f"Mismatch ({msg})"}

    return {"Complexity": "Mismatch (Width is inconsistent)"}

def check_all_requirements(task: str, input_: str, archi: str,
                           complexity: str, file_path: str):
    """
    Check all NN requirements: architecture, task, input, and complexity.

    Parameters:
        task (str): Task requirement (e.g., 'classification-binary').
        input_ (str): Input requirement string (e.g., 'Medium 1k-10k').
        archi (str): Architecture requirement (e.g., 'MLP', 'CNN-2D').
        complexity (str): Complexity requirement (e.g., 'Simple', 'Wide').
        file_path (str): Path to the Python file containing the NN.

    Returns:
        None: Prints 'Success' if all checks pass, otherwise prints
            the results of each check.
    """

    class_node = utils.load_and_parse_model_class(file_path)

    if class_node:
        layers = utils.extract_layers(class_node)
        model_class = utils.import_class_from_file(file_path, class_node.name)

        arch_check = check_architecture(layers, archi)
        task_check = check_task(layers, task)
        input_check = check_input(model_class, input_, layers, archi)
        complexity_check = check_complexity(layers, archi, complexity, input_)
        all_checks = [arch_check, task_check, input_check, complexity_check]

        if all(list(d.values())[0] == "Match" for d in all_checks):
            print("Success")
        else:
            print(arch_check, task_check, input_check, complexity_check)

    else:
        print("No nn.Module subclass found in the file.")


if __name__ == "__main__":
    input_ = utils.construct_input_statements(config.inputs_dict)[1]
    archi = config.architectures[0]
    task = config.tasks[0]
    cmpl = config.complexities[0]
    complexity_summary = utils.generate_complexity_def(archi, input_)
    complexity = f"{cmpl}: {complexity_summary[cmpl]}"
    input_parsed = utils.parse_input_for_file_name(input_)
    kwds = [archi, task, input_parsed, cmpl]
    file_name = "_".join(k.lower() for k in kwds) + ".py"
    file_path = f"dataset_nns/{file_name}"
    check_all_requirements(task, input_, archi, complexity, file_path)