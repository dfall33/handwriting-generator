import torch
from generate import generate_image


def train(
    model: torch.nn.Module,
    n_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    verbose: bool = True,
    generate_images: bool = False,
) -> dict:

    model.to(device)

    history = {
        "train_loss": [],
        "test_loss": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):

            if verbose and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

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
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss / len(train_loader.dataset)}"
            )

        # Validation step
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss = model.loss_function(data, recon_batch, mu, log_var)
                test_loss += loss.item()

        if generate_images:
            generate_image(model=model, device=device)

        if verbose:
            print(f"Test Loss: {test_loss / len(test_loader.dataset)}")

        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["test_loss"].append(test_loss / len(test_loader.dataset))

    return history
