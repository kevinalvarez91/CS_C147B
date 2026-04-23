import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # Linear beta schedule from beta_1 (t=1) to beta_T (t=T)
        t_s = t_s.float()
        beta_t = beta_1 + (t_s - 1) / (T - 1) * (beta_T - beta_1)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1.0 - beta_t
        oneover_sqrt_alpha = 1.0 / torch.sqrt(alpha_t)

        # alpha_bar_t = prod_{s=1}^{t} alpha_s, computed via cumprod lookup
        t_max = int(t_s.max().item())
        t_range = torch.arange(1, t_max + 1, dtype=torch.float32, device=t_s.device)
        betas_all = beta_1 + (t_range - 1) / (T - 1) * (beta_T - beta_1)
        alphas_cumprod = torch.cumprod(1.0 - betas_all, dim=0)
        alpha_t_bar = alphas_cumprod[t_s.long() - 1]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1.0 - alpha_t_bar)


        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        # ==================================================== #
        B = images.shape[0]
        t_s = torch.randint(1, self.modelconfig.T + 1, (B,), device=images.device)

        # One-hot encode conditions, then randomly mask to unconditional (-1)
        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        mask = (torch.rand(B, device=images.device) < self.modelconfig.mask_p).unsqueeze(1)
        c = torch.where(mask, torch.full_like(c, self.modelconfig.condition_mask_value), c)

        schedule = self.scheduler(t_s)
        sqrt_alpha_bar = schedule['sqrt_alpha_bar'].view(B, 1, 1, 1)
        sqrt_oneminus_alpha_bar = schedule['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)

        epsilon = torch.randn_like(images)
        x_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * epsilon

        t_norm = (t_s.float() / self.modelconfig.T).view(B, 1, 1, 1)
        epsilon_pred = self.network(x_t, t_norm, c)
        noise_loss = self.loss_fn(epsilon_pred, epsilon)
        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
        # ==================================================== #
        B = conditions.shape[0]
        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float().to(conditions.device)
        c_uncond = torch.full_like(c, self.modelconfig.condition_mask_value)

        img_size, channels = self.modelconfig.input_dim, self.modelconfig.num_channels
        X_t = torch.randn(B, channels, img_size, img_size, device=conditions.device)

        with torch.no_grad():
            for t in tqdm(range(T, 0, -1), leave=False, desc='sampling'):
                t_s = torch.tensor([t], dtype=torch.float32, device=conditions.device)
                schedule = self.scheduler(t_s)

                beta_t = schedule['beta_t']
                sqrt_beta_t = schedule['sqrt_beta_t']
                oneover_sqrt_alpha = schedule['oneover_sqrt_alpha']
                sqrt_oneminus_alpha_bar = schedule['sqrt_oneminus_alpha_bar']

                t_norm = (t_s / T).view(1, 1, 1, 1).expand(B, 1, 1, 1)
                eps_cond = self.network(X_t, t_norm, c)
                eps_uncond = self.network(X_t, t_norm, c_uncond)

                # Classifier-free guidance
                eps_tilde = (1 + omega) * eps_cond - omega * eps_uncond

                z = torch.randn_like(X_t) if t > 1 else torch.zeros_like(X_t)
                X_t = oneover_sqrt_alpha * (X_t - beta_t / sqrt_oneminus_alpha_bar * eps_tilde) + sqrt_beta_t * z
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images