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
        B = images.shape[0]
        # Sample t uniformly in (0, 1)
        t = torch.rand(B, device=images.device)

        # One-hot encode conditions, randomly mask to unconditional (-1)
        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        mask = (torch.rand(B, device=images.device) < self.modelconfig.mask_p).unsqueeze(1)
        c = torch.where(mask, torch.full_like(c, self.modelconfig.condition_mask_value), c)

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        x_0 = torch.randn_like(images)
        t_4d = t.view(B, 1, 1, 1)
        x_t = (1 - t_4d) * x_0 + t_4d * images

        # Target velocity: dx_t/dt = x_1 - x_0
        u_t = images - x_0

        v_pred = self.network(x_t, t_4d, c)
        loss = self.loss_fn(v_pred, u_t)
        # ==================================================== #
        return loss

    def sample(self, conditions, omega):
        # ==================================================== #
        T = self.modelconfig.T
        B = conditions.shape[0]
        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float().to(conditions.device)
        c_uncond = torch.full_like(c, self.modelconfig.condition_mask_value)

        img_size, channels = self.modelconfig.input_dim, self.modelconfig.num_channels
        x = torch.randn(B, channels, img_size, img_size, device=conditions.device)
        dt = 1.0 / T

        with torch.no_grad():
            for i in tqdm(range(T), leave=False, desc='sampling'):
                t_val = i / T
                t_norm = torch.full((B, 1, 1, 1), t_val, device=conditions.device)

                v_cond = self.network(x, t_norm, c)
                v_uncond = self.network(x, t_norm, c_uncond)

                # Classifier-free guidance
                v_guided = (1 + omega) * v_cond - omega * v_uncond
                x = x + dt * v_guided
        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images