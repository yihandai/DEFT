import math
import torch
from torch.distributions import Normal


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy


def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)


def get_reward(discrim, state, action):
    # state = torch.Tensor(state)
    # action = torch.Tensor(action)
    if action.dim() == 1:
        action = action.unsqueeze(1)
    state_action = torch.cat([state, action], dim=1)
    with torch.no_grad():
        return -torch.log(discrim(state_action))
        # score = discrim(state_action)
        # return (1 - score.log()) - (score).log()
        # return -torch.log(1 - discrim(state_action) + 1e-8)


def save_checkpoint(state, filename):
    torch.save(state, filename)
