import os
import splitfolders
from src.helpers.verify_data import remove_empty_dirs, remove_invalid_imgs

image_dir = "./images"
data_dir = "./data"

# Clean image files
remove_empty_dirs(image_dir)
remove_invalid_imgs(image_dir)

# Check image folder structure
print(os.listdir(image_dir)[:10])

# Split the image dataset into 'train' and 'val' directories
ratio = 0.2
splitfolders.ratio(
    image_dir, output=data_dir, seed=1337, ratio=(1 - ratio, ratio), group_prefix=None
)
