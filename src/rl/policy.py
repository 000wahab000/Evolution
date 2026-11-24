"""SB3 policy that is exactly `CreatureBrain` plus a value head.

`CreatureBrain` is `tanh(w2 @ tanh(w1 @ x + b1) + b2)` (src/model.py). A stock SB3 policy
cannot reproduce that: its `net_arch=[16]` builds `Linear(15,16) -> Tanh -> Linear(16,16)`
(one extra layer), and its action head is linear. So this policy builds the extractor itself:

    policy head: Linear(15, 16) -> Tanh, then Linear(16, 4) -> Tanh
    value head:  Linear(15, 16) -> Tanh, then Linear(16, 1)

That makes the deterministic action bit-for-bit a CreatureBrain forward pass, so a trained
policy exports into `champion.npz` and the existing robustness/gait/replay tooling reads it
unchanged. It also makes GA vs PPO a clean "same brain, different learning rule" ablation.

Caveat: the Gaussian is centred on the squashed mean without a log-det correction, so the
exploration distribution is not a true squashed Gaussian. The env clips to [-1, 1] anyway;
this is a pragmatic baseline, not a textbook squashed policy.
"""

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class TwoHeadMLP(nn.Module):
    """One hidden layer per head, with the latent dims SB3 expects."""

    def __init__(self, features_dim, hidden_dim=16):
        super().__init__()
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim
        self.policy_net = nn.Sequential(nn.Linear(features_dim, hidden_dim), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(features_dim, hidden_dim), nn.Tanh())

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class TanhActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_dim=16, **kwargs):
        self.hidden_dim = hidden_dim
        kwargs.setdefault("activation_fn", nn.Tanh)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = TwoHeadMLP(self.features_dim, self.hidden_dim)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Wrapping keeps the same Linear module (and its Parameters), so the optimizer built
        # inside super()._build() still references them; only the state_dict keys change.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())
