import os
import numpy as np


dataset_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "gameplay", "processed")
dataset_path = os.path.join(dataset_dir_path, "0_0_10_1.npz")

# Load the dataset from the .npz file
loaded_data = np.load(dataset_path, allow_pickle=True)
dataset = loaded_data['dataset']

# Iterate through each example in the dataset
for example in dataset:
    input_data = example['input']
    output_data = example['output']

    # Convert input and output to 2D arrays
    input_array = np.array(input_data)
    output_array = np.array(output_data)

    print("Input:")
    print(input_array)
    print("Output:")
    print(output_array)
    print("\n")
