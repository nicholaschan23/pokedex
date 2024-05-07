import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

image_train_dir = "./data/train"
image_val_dir = "./data/val"
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
train_ds = ImageFolder(image_train_dir, transform=train_tfms)
valid_ds = ImageFolder(image_val_dir, transform=valid_tfms)
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


# show_batch(train_dl)
# plt.show()


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensors to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)



