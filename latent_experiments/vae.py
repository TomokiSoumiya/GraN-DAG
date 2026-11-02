"""Small one-factor Gaussian VAEs, compatible with the saved PyTorch 1.1 environment."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value):
    return math.log(math.expm1(value))


class LatentVAE(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        width = config['width']
        slope = config['negative_slope']
        self.input_dim = input_dim
        self.posterior_floor = config['posterior_std_floor']
        self.observation_floor = config['observation_std_floor']
        # Construct the decoder FIRST: resetting the seed gives identical decoder
        # tensors in A/B despite the different encoder input sizes.
        self.decoder = nn.Sequential(nn.Linear(1, width), nn.LeakyReLU(slope),
                                     nn.Linear(width, width), nn.LeakyReLU(slope),
                                     nn.Linear(width, 3))
        self.raw_observation_std = nn.Parameter(torch.full(
            (3,), inverse_softplus(1.0 - self.observation_floor)))
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.LeakyReLU(slope),
                                     nn.Linear(width, width), nn.LeakyReLU(slope))
        self.posterior_head = nn.Linear(width, 2)
        with torch.no_grad():
            self.posterior_head.bias[1] = inverse_softplus(1.0 - self.posterior_floor)

    def encode(self, features):
        params = self.posterior_head(self.encoder(features))
        return params[:, :1], F.softplus(params[:, 1:]) + self.posterior_floor

    def observation_std(self):
        return F.softplus(self.raw_observation_std) + self.observation_floor


def make_model(input_dim, config):
    torch.manual_seed(config['seed'])
    return LatentVAE(input_dim, config).double()


def gaussian_kl(mu, std):
    return 0.5 * (mu ** 2 + std ** 2 - 1.0 - 2.0 * std.log()).sum(-1)


def gaussian_log_prob(target, mean, std):
    return (-0.5 * ((target - mean) / std) ** 2 - std.log()
            - 0.5 * math.log(2.0 * math.pi)).sum(-1)


def stochastic_terms(model, features, target):
    mu, std = model.encode(features)
    u = mu + std * torch.randn_like(std)
    reconstruction = gaussian_log_prob(target, model.decoder(u), model.observation_std())
    kl = gaussian_kl(mu, std)
    return reconstruction - kl, reconstruction, kl


def quadrature_rule(points):
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return (torch.as_tensor(np.sqrt(2.0) * nodes, dtype=torch.float64),
            torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float64))


def quadrature_terms(model, features, target, points=64):
    """Deterministic q expectation for ELBO and posterior predictive MEAN of X."""
    nodes, weights = quadrature_rule(points)
    mu, std = model.encode(features)
    u = mu[:, None, :] + std[:, None, :] * nodes[None, :, None]
    means = model.decoder(u.reshape(-1, 1)).reshape(len(features), points, 3)
    logp = gaussian_log_prob(target[:, None, :], means, model.observation_std())
    reconstruction = (logp * weights[None, :]).sum(1)
    kl = gaussian_kl(mu, std)
    return dict(mu=mu, std=std, kl=kl, reconstruction=reconstruction,
                elbo=reconstruction-kl,
                x_mean=(means * weights[None, :, None]).sum(1))


def marginal_log_prob(model, target, points=64):
    """Common prior quadrature; deliberately independent of either encoder."""
    nodes, weights = quadrature_rule(points)
    means = model.decoder(nodes[:, None])
    logp = gaussian_log_prob(target[:, None, :], means[None, :, :], model.observation_std())
    return torch.logsumexp(logp + weights.log()[None, :], dim=1)


def epoch_orders(seed, epochs, n_train):
    rng = np.random.RandomState(seed)
    return np.stack([rng.permutation(n_train) for _ in range(epochs)])
