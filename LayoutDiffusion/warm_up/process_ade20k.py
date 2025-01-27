import os
import torch


def convert_tensor_to_standard(coords_tensor, standard_frame_height):
    # Assuming coords_tensor has shape (N, 4) where N is the number of objects
    x_coord, y_coord, width, height = coords_tensor[:, 0], coords_tensor[:, 1], coords_tensor[:, 2], coords_tensor[:, 3]

    half_size = True
    center_size = False
    center_xy = True
    to_centerpoints = True

    if half_size:
        width = width * 2
        height = height * 2

    if center_size:
        width = width + 0.5
        height = height + 0.5

    if center_xy:
        x_coord = x_coord + 0.5
        y_coord = y_coord + 0.5

    if to_centerpoints:
        x_min = x_coord - (width / 2)
        y_min = y_coord - (height / 2)
    else:
        x_min = x_coord
        y_min = y_coord

    # Scale back to standard frame dimensions (e.g., 512x512)
    x_min = (x_min * standard_frame_height).int()
    y_min = (y_min * standard_frame_height).int()
    width = (width * standard_frame_height).int()
    height = (height * standard_frame_height).int()

    # Ensure minimum width and height
    width = torch.clamp(width, min=1)
    height = torch.clamp(height, min=1)

    return torch.stack([x_min, y_min, width, height], dim=1)




# Directory containing .pt files
data_dir = "/share_chairilg/data/ade20k"

# Output .txt file
output_file = "LayoutDiffusion/data/processed_datasets/ade20k/src1_train.txt"

standard_frame_height=512

# Row to add to the beginning
background_coordinate = torch.tensor([[0, 0, standard_frame_height - 1, standard_frame_height - 1]])  

# Open the output file in write mode
with open(output_file, "w") as f:
    # Iterate from 00000 to 27573
    for i in range(27574):
        # Construct the filename (e.g., 00000.pt, 00001.pt, ..., 27573.pt)
        filename = f"{i:05d}.pt"  # Format as 5-digit number with leading zeros
        filepath = os.path.join(data_dir, filename)
        
        # Load the .pt file
        data = torch.load(filepath)
        
        # Extract coords and labels
        coords = data["coords"]  # Assuming coords is a tensor of shape [N, 4]
        coords = convert_tensor_to_standard(coords, standard_frame_height)

        # Concatenate the new row with the existing tensor
        coords = torch.cat((background_coordinate, coords))
        
        labels = data["labels"]  # Assuming labels is a list of N strings

        # Process each label-coordinate pair
        pairs = []
        for label, coord in zip(labels, coords):
            # Format: "label x1 y1 x2 y2"
            print(label)
            pair = f"{label} {int(coord[0])} {int(coord[1])} {int(coord[2])} {int(coord[3])}"
            pairs.append(pair)
        
        # Join pairs with " | " and write to the output file
        line = " | ".join(pairs)
        f.write(line + "\n")
print(f"Processed data saved to {output_file}")

