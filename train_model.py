import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

image_train_dir = "./data" # need to be data/train and data/val but idk why
image_val_dir = "./data"
image_size = 64
batch_size = 20
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Data transforms (normalization & data augmentation)
train_tfms = tt.Compose(
    [
        tt.Resize((80, 80)),  # Resize image to 80x80
        tt.RandomHorizontalFlip(),  # Apply random horizontal flip
        tt.RandomApply(
            [
                # Apply random rotation with degree from -20 to +20
                tt.RandomAffine(degrees=20),
                # Apply random horizontal shift
                tt.RandomAffine(degrees=0, translate=(0.2, 0.2), fill=(255, 255, 255)),
                # Apply random scale
                tt.RandomAffine(degrees=0, scale=(0.7, 0.7)),
                # Apply random image shear
                tt.RandomAffine(degrees=0, shear=(0, 0, 0, 10)),
            ],
            p=0.2,  # Image transformation will be applied with probability 20%
        ),
        tt.CenterCrop((image_size, image_size)),  # Corp image size to 64*64
        tt.ToTensor(),  # Transform image to tensor data
        tt.Normalize(*stats),
    ]
)  # Normalize tensor data

# For validation dataset, only normalization and resize will be applied
valid_tfms = tt.Compose(
    [tt.ToTensor(), tt.Resize((image_size, image_size)), tt.Normalize(*stats)]
)

# PyTorch datasets
train_ds = ImageFolder(image_train_dir, train_tfms)
valid_ds = ImageFolder(image_val_dir, valid_tfms)
train_dl = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True
)
valid_dl = DataLoader(valid_ds, batch_size, num_workers=8, pin_memory=True)


# Denormalization function to denormalize image for display
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


# Display denormalized images
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


# Display all images in one single batch
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


show_batch(train_dl)
plt.show()
