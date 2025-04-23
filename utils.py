import torch
import matplotlib.pyplot as plt


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save the model to the specified path.

    Args:
        model (nn.Module): The model to save.
        path (str): The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, eval: bool = True) -> None:
    """
    Load the model from the specified path.

    Args:
        model (nn.Module): The model to load the weights into.
        path (str): The file path from where the model will be loaded.
    """
    model.load_state_dict(torch.load(path))
    if eval:
        model.eval()


def plot_history(
    history: dict,
    title: str = "Training History",
    save_path: str = None,
) -> None:
    """
    Plot the training and validation loss.

    Args:
        history (dict): A dictionary containing the training and validation loss.
        title (str): The title of the plot.
        save_path (str): The file path to save the plot. If None, the plot will not be saved.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
