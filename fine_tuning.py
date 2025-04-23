from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
from PIL import Image
from vae import VAE


class HandwritingDataset(Dataset):
    def __init__(self, dataset_dir: str, image_size: int = 28) -> None:
        self.paths = [
            os.path.join(dataset_dir, fname)
            for fname in os.listdir(dataset_dir)
            if self.is_image(fname)
        ]

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[idx]).convert("L")
        return self.transform(image), 0
        # Dummy label, not used in VAE training

    def is_image(self, filename: str) -> bool:
        return (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        )


def fine_tune(
    model: VAE,
    dataset_dir: str = "./data",
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):

    for param in model.encoder.parameters():
        param.requires_grad = False

    dataset = HandwritingDataset(dataset_dir=dataset_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(data_loader):
            if verbose and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(data_loader)}")

            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = model.loss_function(
                data, recon_batch, mu, log_var, annealing_factor=epoch / n_epochs
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss / len(data_loader.dataset)}"
            )
    return model
