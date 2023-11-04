import numpy as np
import torch as tr
from utils import *
from agents import DeepQ_LTC_NCP, DeepQ
from deap import algorithms
from agent_utils import policy, update_deepQ, update_deepQ_LTC, update_target_agent
from environment import BoxEnvironment

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
tr.autograd.set_detect_anomaly(True)
tr.set_default_tensor_type(tr.FloatTensor)

# from concurrent.futures import ProcessPoolExecutor

# Initialize Hyperparameters and DEAP toolbox
hyperparams = Hyperparameters()
toolbox = init_toolbox(hyperparams)
environment = BoxEnvironment(hyperparams, device)


# with ProcessPoolExecutor() as executor:
#     results = list(executor.map(my_function, my_iterable))


#ToDo: Decay epsilon ?
dt = tr.tensor(hyperparams.dt).to(device)

def episode(agent, target_agent, optimizer):
    optimizer.zero_grad()
    environment.reset(hyperparams, device)
    state = environment.state
    h0 = None
    total_reward = 0
    t = tr.tensor(0).to(device)
    for _ in range(hyperparams.episode_length):
        q_values, h = agent(state, h0)
        action = policy(q_values, hyperparams, device)
        next_state, reward, done = environment.step(action, t)
        update_deepQ_LTC(agent, target_agent, 
                    (q_values, h, reward, next_state, done),
                    tr.nn.MSELoss(), hyperparams
                    )
        update_target_agent(agent, target_agent,
                            hyperparams.polyak_tau
                        )
        
        optimizer.step()
        h = h.detach()
        h0 = h
        state = next_state
        t = t + dt

        # Average reward over the batch
        total_reward += reward.mean().item()
        if tr.max(done).item(): break
    return total_reward

def episode_deepQ(agent, target_agent, optimizer):
    optimizer.zero_grad()
    environment.reset(hyperparams, device)
    state = environment.state
    total_reward = 0
    t = tr.tensor(0).to(device)
    for _ in range(hyperparams.episode_length):
        q_values = agent(state)
        action = policy(q_values, hyperparams, device)
        next_state, reward, done = environment.step(action, t)
        update_deepQ(agent, target_agent, 
                    (q_values, reward, next_state, done),
                    tr.nn.MSELoss(), hyperparams
                    )
        update_target_agent(agent, target_agent,
                            hyperparams.polyak_tau
                        )
        
        optimizer.step()
        state = next_state
        t = t + dt

        # Average reward over the batch
        total_reward += reward.mean().item()
        if tr.max(done).item(): break
    return total_reward


def episode_batch(individual):
    cumulative_reward = np.zeros((hyperparams.population_size,
                                hyperparams.agent_batch_size)
                            )
    # Initialize the agent and target agent
    agent = DeepQ(individual, hyperparams).to(device)
    optimizer = tr.optim.Adam(agent.parameters(), lr = hyperparams.learning_rate)
    target_agent = DeepQ(individual, hyperparams).to(device)
    target_agent.load_state_dict(agent.state_dict())
    for p in target_agent.parameters():
        p.requires_grad = False

    for _ in range(hyperparams.episode_batch_length):
        cumulative_reward += episode_deepQ(agent, target_agent, optimizer)
    return (cumulative_reward,) # Has to be a tuple





toolbox.register('evaluate', episode_batch)
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('mean', np.mean)

final_pop, logbook = algorithms.eaSimple(toolbox.population(),toolbox,
                                cxpb = 0, mutpb = hyperparams.mutation_rate,
                                ngen = hyperparams.n_generations, stats = stats,
                                verbose = True
                            )

print('finished!')
# def simulation(n_generations):
#     population = toolbox.population(hyperparams.population_size)

#     fitness = map(episode_batch, population)
    

