"""Turn a trained PPO policy back into CreatureBrain weights.

The policy's deterministic action is `tanh(action_net(policy_net(x)))`, which is exactly
`CreatureBrain.get_action` when `net_arch=[16]` and the policy uses TanhActorCriticPolicy.
`log_std` is not exported: evaluation uses the mean action, as usual for PPO.
"""

import numpy as np
import torch.nn as nn


def _single_linear(module, name):
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 1:
        raise ValueError(
            f"{name} has {len(linears)} Linear layers; the CreatureBrain export needs exactly "
            "one. Use TanhActorCriticPolicy (net_arch=[16]).")
    return linears[0]


def policy_to_state_dict(policy):
    """{w1, b1, w2, b2} float32 arrays in CreatureBrain's (out, in) layout."""
    lin1 = _single_linear(policy.mlp_extractor.policy_net, "policy_net")
    lin2 = _single_linear(policy.action_net, "action_net")

    def arr(t):
        return t.detach().cpu().numpy().astype(np.float32)

    return {"w1": arr(lin1.weight), "b1": arr(lin1.bias),
            "w2": arr(lin2.weight), "b2": arr(lin2.bias)}
