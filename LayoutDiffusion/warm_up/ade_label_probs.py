import json
import torch

# Path to the JSON file
json_file_path = "/home/hepe00001/Desktop/neuro_explicit/generative_diffusion/LayoutGeneration/LayoutDiffusion/results/checkpoint/ade20k/ade20k_class_probabilities.json"  # Replace this with the actual file path

# Load the JSON file into a Python dictionary
with open(json_file_path, "r") as f:
    data_dict = json.load(f)

# Extract the values from the dictionary
values = list(data_dict.values())

# Convert the list of values to a PyTorch tensor
probs_tensor = torch.tensor(values)

print(probs_tensor)

# Path to save the tensor
tensor_save_path = "/home/hepe00001/Desktop/neuro_explicit/generative_diffusion/LayoutGeneration/LayoutDiffusion/results/checkpoint/ade20k/ade20k_class_prob_without_label.json"  # Replace with your desired file path

# Save the tensor to the specified path
torch.save(probs_tensor, tensor_save_path)

print(f"Tensor saved to {tensor_save_path}")