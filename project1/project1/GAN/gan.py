import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

NOISE_DIM = 96

device = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))


def sample_noise(batch_size: int, dim: int, seed: int | None = None) -> Tensor:
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(batch_size, dim) * 2 - 1


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ##############################################################################
        # TODO: Implement architecture                                               #
        #                                                                            #
        # HINT: nn.Sequential might be helpful. You'll start by calling nn.Flatten()#
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.noise_dim = noise_dim

        ##############################################################################
        # TODO: Implement architecture                                               #
        #                                                                            #
        # HINT: nn.Sequential might be helpful.                                      #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        return self.model(x)


def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, 1) giving scores.
    - target: PyTorch Tensor of shape (N, 1) containing 0 and 1 giving targets.
          uses torch.float32, not an integer type.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(input, target)


def discriminator_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N, 1) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.

    HINT: Use torch.ones_like / torch.zeros_like to create label tensors that
    automatically match the device and dtype of the input.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    loss = bce_loss(logits_real, real_labels) + bce_loss(logits_fake, fake_labels)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def generator_loss(logits_fake: Tensor) -> Tensor:
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N, 1) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.

    HINT: Use torch.ones_like to create label tensors that automatically match
    the device and dtype of the input.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    real_labels = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, real_labels)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def get_optimizer(model: nn.Module) -> torch.optim.Adam:
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    """
    return optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))


def ls_discriminator_loss(scores_real: Tensor, scores_fake: Tensor) -> Tensor:
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N, 1) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5 * torch.mean((scores_real - 1) ** 2) + 0.5 * torch.mean(scores_fake ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def ls_generator_loss(scores_fake: Tensor) -> Tensor:
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N, 1) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ##############################################################################
        # TODO: Implement architecture                                               #
        #                                                                            #
        # HINT: nn.Sequential might be helpful.                                      #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(0.01),
            nn.Linear(4 * 4 * 64, 1),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        return self.model(x)


class DCGenerator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.noise_dim = noise_dim

        ##############################################################################
        # TODO: Implement architecture                                               #
        #                                                                            #
        # HINT: nn.Sequential might be helpful.                                      #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        return self.model(x)


def run_a_gan(
    D: nn.Module,
    G: nn.Module,
    D_solver: torch.optim.Optimizer,
    G_solver: torch.optim.Optimizer,
    discriminator_loss,
    generator_loss,
    loader_train: DataLoader,
    show_every: int = 250,
    batch_size: int = 128,
    noise_size: int = 96,
    num_epochs: int = 10,
) -> list[np.ndarray]:
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    images = []
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            D_solver.zero_grad()
            real_data = x.to(device)
            logits_real = D(2 * (real_data - 0.5)).to(device)

            g_fake_seed = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if iter_count % show_every == 0:
                print(f'Iter: {iter_count}, D: {d_total_error.item():.4f}, G: {g_error.item():.4f}')
                imgs_numpy = fake_images.data.cpu().numpy()
                images.append(imgs_numpy[0:16])

            iter_count += 1

    return images


def initialize_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


def preprocess_img(x: Tensor) -> Tensor:
    return 2 * x - 1.0


def deprocess_img(x: Tensor) -> Tensor:
    return (x + 1.0) / 2.0


def rel_error(x: np.ndarray, y: np.ndarray) -> float:
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params(model: nn.Module) -> int:
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())
