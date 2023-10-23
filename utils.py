import configparser
import numpy as np
from deap import base, creator, tools

# Currently only works for floats, ints and lists

class Hyperparameters:
    def __init__(self, config_file = 'hyperparameters.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)

        for key in config['Hyperparameters']:
            try:
                value = int(config['Hyperparameters'][key])
            except ValueError:
                try:
                    value = float(config['Hyperparameters'][key])
                except ValueError:
                    try:
                        value = np.array(list(config['Hyperparameters'][key]))
                    except ValueError:
                        value = config['Hyperparameters'][key]
            setattr(self, key, value)




def init_genome():
    sensory_dim = np.random.randint(3, 50)
    hidden_dim = np.random.randint(3, 50)
    # lr = np.random.uniform(0.0001, 0.1)
    # lr_decay = np.random.uniform(0.5, 0.99)

    genome = [
        sensory_dim,
        hidden_dim#,
        # lr,
        # lr_decay
    ]
    return genome

def init_toolbox(hyperparams):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register('individual', tools.initIterate, creator.Individual, lambda: init_genome()
                    )
    toolbox.register('population', tools.initRepeat,
                    list, toolbox.individual,
                    n = hyperparams.population_size
                )
    toolbox.register('mutate', tools.mutUniformInt,
                    low = 4,
                    up = 50,
                    indpb = hyperparams.mutation_rate
                )
    toolbox.register('select', tools.selBest,
                    k = 0.5*hyperparams.population_size,
                    fit_attr = 'fitness'
                )
    #ToDo: register 'mate'
    return toolbox
