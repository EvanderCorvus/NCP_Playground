# import numpy as np
import torch as tr
from environment_utils import *

def sinusoidal_flow(x, t, U0, omega, k):

    F_y = U0*tr.sin(k*x + omega*t)
    F_x = tr.zeros(F_y.shape)

    return F_x, F_y

def vorticity(x, t, omega, k, U0):

    vorticity = U0*k*tr.cos(k*x + omega*t)
    return vorticity



#What is faster, to store the hyperparameters as an attribute or pass it multiple times?

# Initializes all agents on the left side of the box
class BoxEnvironment:
    def __init__(self, hyperparams, device):
        self.state = None
        self.space = Box(hyperparams.width, hyperparams.height)
        self.goal = Circle2D(hyperparams.goal_radius, 
                            tr.tensor([hyperparams.goal_x, hyperparams.goal_y])
                        )
        self.hyperparams = hyperparams
        self.dt = tr.tensor(hyperparams.dt).to(device)
        self.characteristic_length = tr.tensor(hyperparams.characteristic_length).to(device)
        self.U0 = tr.tensor(hyperparams.u0).to(device)
        self.omega = tr.tensor(hyperparams.omega).to(device)
        self.k = tr.tensor(hyperparams.k).to(device)
        self.device = device
        
    def step(self, action, t):
        x, y = self.state[:,0], self.state[:,1]
        theta = action
        F_x, F_y = self.state[:,2], self.state[:,3]
        
        #thermal noise
        noise = tr.normal(tr.zeros(self.hyperparams.agent_batch_size),
                          tr.ones(self.hyperparams.agent_batch_size)).to(self.device)
        theta = theta + self.dt*vorticity(x,t, self.omega, self.k, self.U0) + tr.sqrt(self.dt)*self.characteristic_length*noise

        e_x = tr.cos(theta)
        v_x = e_x + F_x
        x_new = x + v_x*self.dt

        e_y = tr.sin(theta)
        v_y = e_y + F_y
        y_new = y + v_y*self.dt

        F_x_new, F_y_new = sinusoidal_flow(x_new, y_new, self.U0, self.omega, self.k)

        inside_space = self.space.contains(x_new, y_new)

        # No-slip boundary
        self.state[:,0] = tr.clamp(x_new, min=-self.hyperparams.width/2,
                                   max=self.hyperparams.width/2)
        self.state[:,1] = tr.clamp(y_new, min=-self.hyperparams.height/2,
                                   max=self.hyperparams.height/2)
        self.state[:,2] = F_x_new
        self.state[:,3] = F_y_new
        self.state[:,4] = theta

        reward = self.reward(self.dt, inside_space)

        return reward
    
    def reward(self, dt, inside_space):
        # Compute reward
        not_inside_space = tr.logical_not(inside_space)
        reward = -dt*tr.ones(self.state.shape[0])
        wincondition = self.goal_check()
        reward += wincondition*1
        reward -= not_inside_space*0.5

        return reward
    
    def goal_check(self):
        x = self.state[:, 0]
        y = self.state[:, 1]
        wincondition = self.goal.contains(x,y)
        return wincondition


    def reset(self, hyperparams, device):
        self.state = tr.zeros(hyperparams.agent_batch_size,
                                hyperparams.state_dim
                            ).to(device)
        self.state[:,0] = -0.5*tr.ones((hyperparams.agent_batch_size))
