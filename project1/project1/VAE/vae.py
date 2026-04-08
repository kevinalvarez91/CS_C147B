import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int = 15):
        super().__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the encoder and decoder architectures described in the notebook.        #
        #                                                                                          #
        # Encoder: self.encoder maps (N, 1, H, W) → (N, H_d) via nn.Sequential.                  #
        #   Then self.mu_layer and self.logvar_layer each map (N, H_d) → (N, Z).                  #
        #   Set self.hidden_dim = 400.                                                             #
        #                                                                                          #
        # Decoder: self.decoder maps (N, Z) → (N, 1, H, W) via nn.Sequential.                    #
        #   Use nn.Unflatten to reshape the output back to image dimensions.                       #
        ############################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.hidden_dim = 400
        H = self.hidden_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(H, latent_size)
        self.logvar_layer = nn.Linear(H, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, input_size), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models.

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Estimated posterior mu of shape (N, Z)
        - logvar: Estimated variance in log-space of shape (N, Z)
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass:                                                        #
        # (1) Pass x through self.encoder, then self.mu_layer and self.logvar_layer               #
        # (2) Call reparametrize(mu, logvar) to get the latent vector z                           #
        # (3) Pass z through self.decoder to reconstruct x                                         #
        ############################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10, latent_size: int = 15):
        super().__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # K
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Same architecture as VAE, but:                                                     #
        # - Encoder input size is (H*W + K): the flattened image concatenated with one-hot label  #
        # - Decoder input size is (Z + K): the latent vector concatenated with one-hot label       #
        # The flatten + cat operations happen in forward(), not here.                              #
        ############################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.hidden_dim = 400
        H = self.hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size + num_classes, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(H, latent_size)
        self.logvar_layer = nn.Linear(H, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, input_size), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Performs forward pass through FC-CVAE model.

        Inputs:
        - x: Input data of shape (N, 1, H, W)
        - labels: One-hot class vector of shape (N, K)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Estimated posterior mu of shape (N, Z)
        - logvar: Estimated variance in log-space of shape (N, Z)
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass:                                                        #
        # (1) Flatten x and cat with labels, pass through encoder → mu, logvar                    #
        # (2) Reparametrize to get z                                                               #
        # (3) Cat z with labels, pass through decoder → x_hat                                     #
        ############################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_flat = torch.flatten(x, start_dim=1)
        h = self.encoder(torch.cat([x_flat, labels], dim=1))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(torch.cat([z, labels], dim=1))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Differentiably sample random Gaussian data with specified mean and variance
    using the reparameterization trick.

    We sample epsilon ~ N(0, 1) and return z = mu + sigma * epsilon, where
    sigma = exp(0.5 * logvar). This lets gradients flow through z to mu and logvar.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Tensor of shape (N, Z), each z[i,j] ~ N(mu[i,j], exp(logvar[i,j]))

    HINT: Use torch.randn_like(mu) to sample epsilon — it automatically matches
    the device and dtype of mu so you don't need to move anything to GPU manually.
    """
    z = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    epsilon = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * epsilon

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return z


def loss_function(x_hat: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Computes the negative variational lower bound loss term of the VAE.

    Loss = reconstruction_loss + KL_divergence, averaged over the batch.

    Reconstruction loss: binary cross entropy between x_hat and x.
    KL divergence (closed form for diagonal Gaussian vs standard normal):
        -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Original input data of shape (N, 1, H, W)
    - mu: Posterior mean of shape (N, Z)
    - logvar: Posterior log-variance of shape (N, Z)

    Returns:
    - loss: Scalar tensor containing the mean loss over the batch.

    HINT: Use F.binary_cross_entropy(x_hat, x, reduction='sum') for the
    reconstruction term, then divide the total loss by batch size (mu.shape[0]).
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    reconstruction = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    loss = (reconstruction + kl) / mu.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss
