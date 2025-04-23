import torch
import matplotlib.pyplot as plt


def generate_image(model, device: str = "cuda"):

    model.to(device)
    z = torch.randn(1, model.latent_dim).to(device)

    img = model.decode(z)

    # plt.imshow(img.squeeze().cpu().numpy(), cmap="gray")
    plt.imshow(img.squeeze().cpu().detach().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
