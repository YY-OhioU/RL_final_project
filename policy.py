import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_sizes, device='cuda:0'):
        super(Policy, self).__init__()
        if isinstance(h_sizes, int):
            h_sizes = [h_sizes]
        layers = [nn.Linear(s_size, h_sizes[0]), nn.ReLU()]
        for i in range(len(h_sizes) - 1):
            layers += [nn.Linear(h_sizes[i], h_sizes[i + 1]),
                       nn.ReLU()]
        layers.append(nn.Linear(h_sizes[-1], a_size))
        self.layers = nn.Sequential(*layers)
        self.value_dev = device

    def forward(self, x):
        x = self.layers(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.value_dev)
        # state = torch.Tensor(list(state)).float().unsqueeze(0).to(self.value_dev)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
