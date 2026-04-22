import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalFM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels,
            self.modelconfig.num_feat,
            self.modelconfig.num_classes,
            self.modelconfig.input_dim,
        )

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  


        pass



        # ==================================================== #
        return loss

    def sample(self, conditions, omega):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  


        pass


        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images