# vae_utils.py — provided utilities, do not modify

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def reset_seed(number: int) -> None:
    """Reset random seed for reproducibility."""
    random.seed(number)
    torch.manual_seed(number)


def rel_error(x: Tensor, y: Tensor, eps: float = 1e-10) -> float:
    """Relative error between two tensors."""
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot


def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    """Convert label tensor of shape (N,) to one-hot matrix (N, C)."""
    return F.one_hot(labels, num_classes=num_classes).float()


def show_images(images: Tensor) -> None:
    images = images.reshape(images.shape[0], -1)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(img.reshape(sqrtimg, sqrtimg).cpu(), cmap='Greys_r')


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_vae(
    epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cond: bool = False,
) -> None:
    """Train a VAE or CVAE for one epoch."""
    from vae import loss_function  # imported here to avoid circular import at module level
    device = next(model.parameters()).device
    model.train()
    train_loss = 0.0
    num_classes = 10
    for data, labels in loader:
        data = data.to(device)
        if cond:
            one_hot_vec = one_hot(labels, num_classes).to(device)
            x_hat, mu, logvar = model(data, one_hot_vec)
        else:
            x_hat, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}: loss = {train_loss / len(loader):.4f}')
