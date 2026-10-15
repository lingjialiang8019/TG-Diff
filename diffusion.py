"""Forward mask corruption and deterministic DDIM sampling."""
import torch
from torch import nn


class MaskDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if timesteps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Invalid diffusion schedule.")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("alpha_bar", torch.cat([torch.ones(1), alpha_bar]))

    def q_sample(self, clean_mask, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_mask)
        alpha = self.alpha_bar[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * clean_mask + (1 - alpha).sqrt() * noise

    def loss_weights(self, timesteps, gamma=5.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        alpha = self.alpha_bar[timesteps]
        weights = (alpha / (1 - alpha).clamp_min(1e-12)).clamp(max=gamma)
        return weights / weights.mean().clamp_min(1e-12)

    def sampling_timesteps(self, steps):
        """Return exactly the requested number of denoiser evaluations."""
        if not 1 <= steps <= self.timesteps:
            raise ValueError("Sampling steps must lie between 1 and timesteps.")
        return torch.linspace(self.timesteps, 1, steps).round().long().tolist()

    @torch.no_grad()
    def sample(self, model, image, input_ids, attention_mask, steps=3, generator=None):
        batch, _, height, width = image.shape
        noisy = torch.randn((batch, 1, height, width), device=image.device,
                            dtype=image.dtype, generator=generator)
        times = self.sampling_timesteps(steps)
        for index, time in enumerate(times):
            t = torch.full((batch,), time, device=image.device, dtype=torch.long)
            clean = model(image, input_ids, attention_mask, noisy, t).sigmoid()
            if index == len(times) - 1:
                return clean
            next_time = times[index + 1]
            alpha, next_alpha = self.alpha_bar[time], self.alpha_bar[next_time]
            noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
