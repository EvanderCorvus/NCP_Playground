# import numpy as np
import torch as tr
from environment_utils import *

def sinusoidal_flow(x, t, hyperparams):
    omega = hyperparams.omega
    k = hyperparams.k
    U0 = hyperparams.u0

    F_y = U0*tr.sin(k*x + omega*t)
    F_x = tr.zeros(F_y.shape)

    return F_x, F_y

def vorticity(x, t, hyperparams):
    omega = hyperparams.omega
    k = hyperparams.k
    U0 = hyperparams.u0

    vorticity = U0*k*tr.cos(k*x + omega*t)
    return vorticity



#What is faster, to store the hyperparameters as an attribute or pass it multiple times?

# Initializes all agents on the left side of the box
class BoxEnvironment:
    def __init__(self, hyperparams):
        self.state = None
        self.space = Box(hyperparams.width, hyperparams.height)
        self.goal = Circle2D(hyperparams.goal_radius, 
                            tr.tensor([hyperparams.goal_x, hyperparams.goal_y])
                        )
        self.hyperparams = hyperparams
        # ToDo: Store hyperparam variables as attributes and as tensors in GPU!

    def step(self, action, t):
        x, y = self.state[:,0], self.state[:,1]
        theta = action
        F_x, F_y = self.state[:,2], self.state[:,3]
        
        #thermal noise
<<<<<<< HEAD
        noise = tr.normal(tr.zeros(self.hyperparams.agent_batch_size),
                          tr.ones(self.hyperparams.agent_batch_size))
        theta = theta + vorticity(x,t, self.hyperparams)*self.hyperparams.dt + np.sqrt(self.hyperparams.dt)*self.hyperparams.characteristic_length*noise
=======
        noise = np.random.normal(np.zeros(self.agent_batch_size), np.ones(self.agent_batch_size))
        theta = theta + vorticity(x,t)*self.hyperparams.dt + np.sqrt(self.hyperparams.sdt)*self.hyperparams.characteristic_length*noise
>>>>>>> d23e4acfcdf6b803b932ea5ae3b06971b7e0b40c

        e_x = tr.cos(theta)
        v_x = e_x + F_x
        x_new = x + v_x*self.hyperparams.dt

        e_y = tr.sin(theta)
        v_y = e_y + F_y
        y_new = y + v_y*self.hyperparams.dt

<<<<<<< HEAD
        F_x_new, F_y_new = sinusoidal_flow(x_new, y_new, self.hyperparams.u0)
=======
        F_x_new, F_y_new = sinusoidal_flow(x_new, y_new, self.hyperparams.U0)
>>>>>>> d23e4acfcdf6b803b932ea5ae3b06971b7e0b40c

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
<<<<<<< HEAD
        self.state[:,0] = -0.5*tr.ones((hyperparams.agent_batch_size))
=======
        self.state[:,0] = -0.5*tr.ones((hyperparams.state_dim))
>>>>>>> d23e4acfcdf6b803b932ea5ae3b06971b7e0b40c
