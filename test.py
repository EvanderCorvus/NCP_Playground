from utils import *

hyperparams = Hyperparameters()
toolbox = init_toolbox(hyperparams)

ind = toolbox.individual()
print(int(round(4.5)))



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
'''