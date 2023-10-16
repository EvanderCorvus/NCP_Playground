import numpy as np
import torch as tr
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP, NCP


class DeepQ_LTC_NCP(nn.Module):
    def __init__(self, genome, hyperparams):
        super(DeepQ_LTC_NCP, self).__init__()
        sensory_dim = genome[0]
        hidden_dim = genome[1]
        # self.lr = genome[2]
        # self.lr_decay = genome[3]
        self.fc = nn.Linear(hyperparams.state_dim, sensory_dim)

        wiring = AutoNCP(hidden_dim, hyperparams.act_dim)
        self.rnn = LTC(sensory_dim, wiring, batch_first=True)


    def forward(self, state, h0 = None):
        x = self.fc(state)
        x, h = self.rnn(x, h0)
        return x, h
        

# This Class implements a parallel forward operation for the entire population
class DeepQ_Population(nn.Module):
    def __init__(self, hyperparams):
        super(DeepQ_Population, self).__init__()
        pass


def policy(q_values, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(q_values))
    else:
        action_idx = tr.argmax(q_values).item()

    angle = 2*np.pi*action_idx/len(q_values)

    return angle