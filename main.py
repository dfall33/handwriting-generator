from dataset import get_data_loaders
from train import train
from vae import VAE
import torch
import utils
from generate import generate_image


def main():

    # constants and hyperparameters
    batch_size = 64
    n_epochs = 10
    image_width = 28
    vae_latent_dim = 32
    verbose = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 1e-3
    model_path = "./models/vae_model.pth"

    new_model = True

    if new_model:

        if verbose:
            print("Starting VAE training...")
            print(f"Using device: {device}")

        # Load data
        dataset_dir = "./data"
        train_loader, test_loader = get_data_loaders(
            batch_size=batch_size, dataset_dir=dataset_dir
        )

        if verbose:
            print("Got dataloaders")

        # Initialize model
        model = VAE(image_size=image_width, latent_dim=vae_latent_dim)

        if verbose:
            print("Initialized VAE model")
            print(f"Model architecture: {model}")

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Train the model
        history = train(
            model=model,
            n_epochs=n_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=verbose,
            generate_images=True,
        )

        utils.plot_history(history=history, title="VAE Training History")

        # Save the model
        utils.save_model(model=model, path=model_path)

        del model
        if verbose:
            print("Model saved and deleted successfully")

    # Load the model
    model = VAE(image_size=image_width, latent_dim=vae_latent_dim)
    utils.load_model(model=model, path=model_path, eval=True)
    if verbose:
        print("Model loaded successfully")

    n_images = 5
    for i in range(n_images):
        generate_image(model=model, device=device)
        if verbose:
            print(f"Generated image {i + 1}/{n_images}")


if __name__ == "__main__":
    main()
