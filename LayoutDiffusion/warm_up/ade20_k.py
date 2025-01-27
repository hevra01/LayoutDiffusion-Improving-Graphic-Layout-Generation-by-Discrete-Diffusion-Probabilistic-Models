import torch


import os

file_path = '/share_chairilg/data/ade20k/20660.pt'


# Load the .pt file
dataset = torch.load(file_path)

# Print the contents of the .pt file
print("Contents of the .pt file:")
for key, value in dataset.items():
    print(f"{key}: ")


file_path = '/share_chairilg/data/ade20k/23618.pt'


# Load the .pt file
dataset = torch.load(file_path)

# Print the contents of the .pt file
print("Contents of the .pt file:")
print((dataset["coords"]))
for key, value in dataset.items():
    print(f"{key}:  ")

