import os
import splitfolders

image_dir = "./dataset"

# Check image folder structure
print(os.listdir(image_dir)[:10])

# Split the image dataset into 'train' and 'val' directories
ratio = 0.2
splitfolders.ratio(image_dir,
                   output='./data',
                   seed=1337,
                   ratio=(1-ratio, ratio),
                   group_prefix=None)