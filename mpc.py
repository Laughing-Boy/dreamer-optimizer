from math import inf
import torch
from torch import jit
from torch import nn, optim


# Model-predictive control planner with cross-entropy method and learned transition model
class Action_Optimizer(nn.Module):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters',
                     'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
    def forward(self, belief, state,actions):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        # Sample next states
        belief, state = belief.view(-1, H), state.view(-1, Z)
        beliefs, states, _, _ = self.transition_model(
              state, actions.unsqueeze(0), belief)
        returns = self.reward_model(
            beliefs.view(-1, H), states.view(-1, Z))
        retur = returns.view(actions.size(0), -1)
        # retur.retain_grad()
        ret = -1*retur.sum(dim=0)

        return  ret
