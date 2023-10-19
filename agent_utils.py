import numpy as np
import torch as tr

def policy(q_values, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(q_values))
    else:
        action_idx = tr.argmax(q_values).item()

    angle = 2*np.pi*action_idx/len(q_values)

    return angle


def update_deepQ(agent, target_agent):
    pass