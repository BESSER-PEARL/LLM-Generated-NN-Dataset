"""
Module defining requirements, mappings, and data structures 
used for NN dataset generation.
"""


# ---- Possible Requirements ----

inputs_dict = {
    "Tabular": {"features": ["Small <50", "Large >2000"]},
    "Time series": {
        "seq_length": ["Small <50", "Large >2000"],
        "features": ["Small <50", "Large >2000"]
    },
    "Text": {
        "seq_length": ["Small <50", "Large >2000"],
        "vocab_size": ["Small <1k", "Large >100k"]
    },
    "Image": {"resolution": ["Small <64", "Large >1024"]}
}

tasks = [
    "Classification-binary",
    "Classification-multiclass",
    "Regression",
    "Representation-learning"
]

architectures = ["MLP", "CNN-1D", "CNN-2D", "CNN-3D",
                 "RNN-Simple", "RNN-LSTM", "RNN-GRU"]
complexities =  ["Simple", "Wide", "Deep", "Wide-Deep"]


# ---- Complexity rules ----
INF = float("inf")

ANY_DEPTH = {"min_layers": 1, "max_layers": INF}
SHALLOW_1_4 = {"min_layers": 1, "max_layers": 4}
SHALLOW_1_2 = {"min_layers": 1, "max_layers": 2}
DEEP_4P = {"min_layers": 4, "max_layers": INF}
DEEP_2P = {"min_layers": 2, "max_layers": INF}


COMPLEXITY_RULES = {
    "MLP": {
        "Simple": {
            **SHALLOW_1_4,
            **{"min_val": 1, "max_val": "min(128, upper_bound_of_features)"}
        },
        "Wide-Deep": {
            **DEEP_4P,
            **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
        },
        "Wide": {
            **ANY_DEPTH,
            **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
        },
        "Deep": {
            **DEEP_4P,
            **{"min_val": 1, "max_val": INF}
        },
    },
}

CNN_1D = {
    "Simple": {
        **SHALLOW_1_4,
        **{"min_val": 1, "max_val": "min(128, upper_bound_of_features)"}
    },
    "Wide-Deep": {
        **DEEP_4P,
        **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
    },
    "Wide": {
        **ANY_DEPTH,
        **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
    },
    "Deep": {
        **DEEP_4P,
        **{"min_val": 1, "max_val": INF}
    },
}
CNN_2D = {
    "Simple": {
        **SHALLOW_1_4,
        **{"min_val": 1, "max_val": "min(8, image_width//8)"}
    },
    "Wide-Deep": {
        **DEEP_4P,
        **{"min_val": "min(8, image_width//8)", "max_val": INF}
    },
    "Wide": {
        **ANY_DEPTH,
        **{"min_val": "min(8, image_width//8)", "max_val": INF}
    },
    "Deep": {
        **DEEP_4P,
        **{"min_val": 1, "max_val": INF}
    },
}
CNN_3D = {
    "Simple": {
        **SHALLOW_1_4,
        **{"min_val": 1, "max_val": "min(8, image_width//8)"}
    },
    "Wide-Deep": {
        **DEEP_4P,
        **{"min_val": "min(8, image_width//8)", "max_val": INF}
    },
    "Wide": {
        **ANY_DEPTH,
        **{"min_val": "min(8, image_width//8)", "max_val": INF}
    },
    "Deep": {
        **DEEP_4P,
        **{"min_val": 1, "max_val": INF}
    },
}
COMPLEXITY_RULES["CNN-1D"] = CNN_1D
COMPLEXITY_RULES["CNN-2D"] = CNN_2D
COMPLEXITY_RULES["CNN-3D"] = CNN_3D


RNN_RULES = {
    "Simple": {
        **SHALLOW_1_2,
        **{"min_val": 1, "max_val": "min(128, upper_bound_of_features)"}
    },
    "Wide-Deep": {
        **DEEP_2P,
        **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
    },
    "Wide": {
        **ANY_DEPTH,
        **{"min_val": "min(128, upper_bound_of_features)", "max_val": INF}
    },
    "Deep": {
        **DEEP_2P,
        **{"min_val": 1, "max_val": INF}
    },
}
for rnn_type in ["RNN-Simple", "RNN-LSTM", "RNN-GRU"]:
    COMPLEXITY_RULES[rnn_type] = RNN_RULES


Embedding = {
    "Simple": {
        **{"min_val": 1, "max_val": "min(128, upper_bound_of_vocab_size)"}
    },
    "Wide-Deep": {
        **{"min_val": "min(128, upper_bound_of_vocab_size)", "max_val": INF}
    },
    "Wide": {
        **{"min_val": "min(128, upper_bound_of_vocab_size)", "max_val": INF}
    },
    "Deep": {
        **{"min_val": 1, "max_val": INF}
    },
}
COMPLEXITY_RULES["Embedding"] = Embedding


# ---- Prompt Template ----

PROMPT_TEMPLATE = """
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: {architecture} 
Learning Task: {task}
Input Type + Scale: {input_} 
Complexity: {complexity}


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
"""


ARCH_TO_LAYER = {
    "MLP": "Linear",
    "CNN-1D": "Conv1d",
    "CNN-2D": "Conv2d",
    "CNN-3D": "Conv3d",
    "RNN-Simple": "RNN",
    "RNN-LSTM": "LSTM",
    "RNN-GRU": "GRU",
}

BATCH_NORM = ["BatchNorm1d","BatchNorm2d","BatchNorm3d"]
