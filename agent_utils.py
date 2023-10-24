import numpy as np
import torch as tr
from torch.nn import MSELoss

def policy(q_values, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(q_values))
    else:
        action_idx = tr.argmax(q_values).item()

    angle = 2*np.pi*action_idx/len(q_values)

    return angle


def update_deepQ(agent, target_agent, transition, hyperparams):
    Q1, h, reward, next_state, done = transition
    prediction = tr.max(Q1)

    Q2 = tr.max(target_agent(next_state,h)[0])
    target = reward + hyperparams.future_discount * Q2 * (1-done)
    loss = MSELoss()(prediction, target)
    loss.backward()

def update_target_agent(agent, target_agent, polyak_tau):
    with tr.no_grad():
        for p, p_target in zip(agent.parameters(), target_agent.parameters()):
            p_target.data.mul_(polyak_tau)
            p_target.data.add_((1-polyak_tau) * p.data)