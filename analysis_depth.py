"""
Module containing the code for generating the depth boxplots.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import import_nn_from_file, extract_layer_and_func_types




NN_DIR = "dataset_nns"
MAIN_LAYERS = {"Linear", "Conv1d", "Conv2d", "Conv3d", "RNN", "LSTM", "GRU"}
CATEGORIES = ["All NNs", "Simple", "Wide-Deep", "Deep", "Wide"]

all_depths = []
main_depths = []
all_layers = {}

for filename in os.listdir(NN_DIR):
    if filename.endswith(".py"):
        try:
            nn_model = import_nn_from_file(os.path.join(NN_DIR, filename))
            layers, _ = extract_layer_and_func_types(nn_model)
            all_layers[filename] = layers
        except ValueError as e:
            print(f"Error processing {filename}: {e}")

groups = {
    "MLP": [f for f in all_layers if f.lower().startswith("mlp")],
    "RNN & CNN-1D": [
        f for f in all_layers if f.lower().startswith("rnn")
        or f.lower().startswith("cnn-1d")
    ],
    "CNN-2D & CNN-3D": [
        f for f in all_layers if f.lower().startswith("cnn-2d")
        or f.lower().startswith("cnn-3d")
    ]
}

for cat in CATEGORIES:
    depths_cat = []
    main_cat = []
    for filename, layers in all_layers.items():
        name_without_ext = filename.lower().replace(".py", "")
        if (cat == "All NNs") or (name_without_ext.endswith(cat.lower())):
            depths_cat.append(len(layers))
            main_cat.append(sum(1 for l in layers if l in MAIN_LAYERS))
    all_depths.append(depths_cat)
    main_depths.append(main_cat)


data_to_plot = []
positions = []
WIDTH = 0.35
for i in range(len(CATEGORIES)):
    data_to_plot.append(all_depths[i])
    data_to_plot.append(main_depths[i])
    positions.append(i*1.5)
    positions.append(i*1.5 + WIDTH)


fig, ax = plt.subplots(figsize=(5,2.5))
box = ax.boxplot(data_to_plot, positions=positions,
                 widths=WIDTH*0.9, patch_artist=True)


for i, b in enumerate(box['boxes']):
    if i % 2 == 0:
        b.set(facecolor='darkseagreen')
    else:
        b.set(facecolor='lightskyblue')


xticks = [i*1.5 + WIDTH/2 for i in range(len(CATEGORIES))]
ax.set_xticks(xticks)
ax.set_xticklabels(CATEGORIES, fontsize=11)
ax.set_ylim(bottom=0)

ax.set_ylabel("Nbr of Layers (Depth)", fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

legend_elements = [
    Patch(facecolor='darkseagreen', label='All Layers'),
    Patch(facecolor='lightskyblue', label='Characterising Layers')
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, fontsize=11)


plt.tight_layout()
plt.show()
