import numpy as np
import torch as tr
from environment_utils import *

def flow(state, t):
    x, y = state[:,0], state[:,1]

#What is faster, to store the hyperparameters as an attribute or pass it multiple times?

# Initializes all agents on the left side of the box
class BoxEnvironment:
    def __init__(self, hyperparams, device):
        self.state = tr.zeros((hyperparams.population_size,
                               hyperparams.agent_batch_size,
                               hyperparams.state_dim)
                            ).to(device)
        self.state[:,:,0] = -0.5*tr.ones((hyperparams.state_dim))
        self.space = Box(hyperparams.width, hyperparams.height)
        self.goal = Circle2D(hyperparams.goal_radius, 
                            np.array([hyperparams.goal_x, hyperparams.goal_y])
                        )
        self.population_size = hyperparams.population_size
        self.agent_batch_size = hyperparams.agent_batch_size
        self.dt = hyperparams.dt
        self.characteristic_length = hyperparams.characteristic_length

    def step(self, action):
        self.state.reshape(-1, self.state.shape[-1])
        x, y, F_x, F_y = self.state[:,0], self.state[:,1], self.state[:,2], self.state[:,4]

        pass

    def reset(self):
        self.state = tr.zeros((hyperparams.population_size,
                               hyperparams.agent_batch_size,
                               hyperparams.state_dim)
                            ).to(device)
        self.state[:,:,0] = -0.5*tr.ones((hyperparams.state_dim))