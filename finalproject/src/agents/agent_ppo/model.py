"""
CNN Actor-Critic for the Collector environment.

Input:  (B, C, 16, 16) float tensor of stacked feature maps.
Output: action logits (B, 4) and value estimate (B,).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_init(layer, std=1.0, bias_const=0.0):
    """Orthogonal init - the standard PPO trick for stable learning."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Small CNN trunk -> shared MLP -> separate policy and value heads.
    Sized for a 16x16 grid; runs comfortably on CPU.
    """

    def __init__(self, in_channels: int, num_actions: int = 4, hidden_dim: int = 256):
        super().__init__()

        # Conv trunk - all same-size, ReLU activations.
        # 3 conv layers give ~7x7 effective receptive field, enough to see
        # most of the 16x16 board through stacking + spatial patterns.
        self.conv = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, 32, 3, padding=1), std=1.414),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 3, padding=1), std=1.414),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, 3, padding=1), std=1.414),
            nn.ReLU(),
            # 1x1 compression to keep flatten size reasonable on CPU
            _layer_init(nn.Conv2d(64, 16, 1), std=1.414),
            nn.ReLU(),
        )
        # 16 * 16 * 16 = 4096
        self.mlp = nn.Sequential(
            _layer_init(nn.Linear(16 * 16 * 16, hidden_dim), std=1.414),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.414),
            nn.ReLU(),
        )

        self.policy_head = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        h = self.mlp(h)
        return h

    def forward(self, x: torch.Tensor):
        h = self._trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False):
        """Select an action for a batch of observations. Used in rollouts and inference."""
        logits, value = self.forward(x)
        if deterministic:
            action = logits.argmax(dim=-1)
            logprob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Used during PPO update: returns logprobs, entropy, values."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value