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
        x = x.unsqueeze(1)
        x, h = self.rnn(x, h0)
        return x.squeeze(1), h
        

class DeepQ(nn.Module):
    def __init__(self, genome, hyperparams,):
        super(DeepQ, self).__init__()
        num_layers = genome[0]
        hidden_dim = genome[1]
        layers = [nn.Linear(hyperparams.state_dim, hidden_dim), nn.LeakyReLU()]
        for i in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()]
        self.fc = nn.Sequential(*layers, 
                                nn.Linear(hidden_dim, hyperparams.act_dim)
                            )
        
    def forward(self, state):
        # if not state.requires_grad: raise Exception("state doesn't require grad")
        if tr.isnan(state).any() or tr.isinf(state).any(): raise Exception("state is nan")
        return self.fc(state)
        



# This Class implements a parallel forward operation for the entire population
class DeepQ_Population(nn.Module):
    def __init__(self, hyperparams):
        super(DeepQ_Population, self).__init__()
        pass

