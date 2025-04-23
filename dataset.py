from torchvision.datasets import EMNIST
from torchvision import transforms
import torch

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


def get_dataset(dataset_dir: str = "./data") -> EMNIST:

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        ]
    )

    train_dataset = EMNIST(
        root=dataset_dir,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = EMNIST(
        root=dataset_dir,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 64, dataset_dir: str = "./data"
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset, test_dataset = get_dataset(dataset_dir=dataset_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
