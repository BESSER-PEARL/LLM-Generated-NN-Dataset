# LLM-Generated-NN-Dataset

This repository hosts the dataset accompanying the paper **"On the Use of LLMs to Generate a Dataset of Neural Networks"**.  
The dataset contains PyTorch-based neural network code automatically generated using GPT-5 according to specific requirements.

## Overview

The dataset was generated to support research on neural network code verification, refactoring, and migration — with a focus on improving the reliability and adaptability of network implementations.  


Each network is generated based on a set of requirements describing:
- **Architecture** 
- **Task**
- **Input type and scale**
- **Complexity level**

All networks are implemented in PyTorch.

## Generating Data

To generate a neural network implementation using GPT-5, run:
```bash
python generate_nn.py
```
Each `.py` file in `dataset_nns/` starts with the prompt that was used for its generation, followed by the produced PyTorch implementation.

## Validation Tool

The **`verify_nn.py`** script validates the generated networks against their specification.  
It checks compliance with specified architecture, task, input type and scale, and complexity requirements to ensure consistency and correctness.

To run validation:

```bash
python verify_nn.py
```

## Reproducing the Depth Analysis

The repository includes the `analysis_depth.py` script, which reproduces the plot of NN depth as presented in the paper.  
Run it with:

```bash
python analysis_depth.py
```

## Usage

You can clone and explore the dataset locally:

```bash
git clone https://github.com/BESSER-PEARL/LLM-Generated-NN-Dataset.git
cd LLM-Generated-NN-Dataset
```