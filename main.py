import numpy as np
import torch as tr
from utils import *
from agents import DeepQ_LTC_NCP
from deap import algorithms
from agent_utils import policy, update_deepQ, update_target_agent
from environment import BoxEnvironment

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
tr.autograd.set_detect_anomaly(True)
tr.set_default_tensor_type(tr.FloatTensor)

# from concurrent.futures import ProcessPoolExecutor

# Initialize Hyperparameters and DEAP toolbox
hyperparams = Hyperparameters()
toolbox = init_toolbox(hyperparams)
environment = BoxEnvironment(hyperparams).to(device)


# with ProcessPoolExecutor() as executor:
#     results = list(executor.map(my_function, my_iterable))


#ToDo: Decay epsilon ?

# episode should return an array of all the cumulative rewards for each individual
def episode(agent, target_agent, optimizer):
    agent.optimizer.zero_grad()
    environment.reset(hyperparams, device)
    state = environment.state
    h0 = None
    total_reward = 0
    for _ in range(hyperparams.episode_length):
        q_values, h = agent(state, h0)
        action = policy(q_values, hyperparams.epsilon)
        next_state, reward, done = environment.step(action)
        update_deepQ(agent, target_agent, 
                    (q_values, h, reward, next_state, done)
                    )
        update_target_agent(agent, target_agent,
                            hyperparams.polyak_tau
                        )
        
        optimizer.step()
        h = h.detach()
        h0 = h
        state = next_state
        total_reward += reward
        if done: break
    return total_reward

def episode_batch(individual):
    cumulative_reward = np.zeros((hyperparams.population_size,
                                hyperparams.agent_batch_size)
                            )
    # Initialize the agent and target agent
    agent = DeepQ_LTC_NCP(individual).to(device)
    optimizer = tr.optim.Adam(agent.parameters(), lr = hyperparams.lr)
    target_agent = DeepQ_LTC_NCP(individual).to(device)
    target_agent.load_state_dict(agent.state_dict())
    for p in target_agent.parameters():
        p.requires_grad = False

    for _ in range(hyperparams.episode_batch_length):
        cumulative_reward += episode(agent, target_agent, optimizer)
    return (cumulative_reward,) # Has to be a tuple




toolbox.register('evaluate', episode_batch)

final_pop = algorithms.eaSimple(toolbox.population(),toolbox,
                                cxpb = 0, mutpb = hyperparams.mutation_rate,
                                ngen = hyperparams.n_generations, verbose = False
                            )

# def simulation(n_generations):
#     population = toolbox.population(hyperparams.population_size)

#     fitness = map(episode_batch, population)
    

