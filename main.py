import numpy as np
import torch as tr
from ncps.torch import LTC
from ncps.wirings import AutoNCP, NCP
from deap import base, creator, tools, algorithms
from utils import *
from agents import DeepQ_LTC_NCP
from agent_utils import policy, update_deepQ
from environment import BoxEnvironment

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
tr.autograd.set_detect_anomaly(True)
tr.set_default_tensor_type(tr.FloatTensor)

# from concurrent.futures import ProcessPoolExecutor

# Initialize Hyperparameters and DEAP toolbox
hyperparams = Hyperparameters()
toolbox = init_toolbox(hyperparams)
# environment = BoxEnvironment(hyperparams)


# with ProcessPoolExecutor() as executor:
#     results = list(executor.map(my_function, my_iterable))


#ToDo: Decay epsilon ?

# episode should return an array of all the cumulative rewards for each individual
def episode(agent, target_agent):
    pass

def episode_batch(individual):
    cumulative_reward = np.zeros((hyperparams.population_size,
                                hyperparams.agent_batch_size)
                            )
    # Initialize the agent and target agent
    agent = DeepQ_LTC_NCP(individual).to(device)
    target_agent = DeepQ_LTC_NCP(individual).to(device)
    target_agent.load_state_dict(agent.state_dict())
    for p in target_agent.parameters():
        p.requires_grad = False

    for _ in range(hyperparams.episode_batch_length):
        cumulative_reward += episode(hyperparams.n_steps)
    return cumulative_reward

def simulation(n_generations):
    population = toolbox.population(hyperparams.population_size)

    fitness = map(episode_batch, population)
    







'''
state_dim = 5
act_dim = 8 # number of angles
batch_size = 3
hidden_dim = 10
memory_length = 1

inter_neurons = 10
command_neurons = 10
motor_neurons = 8
sensory_fanout = 6
inter_fanout = 6
recurrent_command_synapses = 4
motor_fanin = 6

hidden_dim2 = inter_neurons + command_neurons + motor_neurons


wiring = AutoNCP(hidden_dim, act_dim)
wiring2 = NCP(inter_neurons, command_neurons,
                motor_neurons, sensory_fanout,
                inter_fanout, 
                recurrent_command_synapses,
                motor_fanin
            )

rnn = LTC(state_dim, wiring, batch_first=True)

x = tr.randn(batch_size, memory_length, state_dim) # (batch, time, features)
h0 = tr.zeros(batch_size, hidden_dim) # (batch, units)
output, hn = rnn(x,h0)
# print(output.shape)

plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
plt.show()
'''^1   