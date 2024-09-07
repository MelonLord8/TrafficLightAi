import random
import numpy as np
"""
Creates an array of length num_ticks + 1, 
First element being the previous states of the traffic light 0 being one orientation and 1 being the other
The rest being the number of incoming cars on the 0 lane, then 1 lane, then pedestrians in the same order.
"""
def makeScenario(num_ticks : int, avg_car0, avg_car1, avg_ped0, avg_ped1, prev = None):
    if prev == None:
        prev = []
        for i in range(4):
            prev.append(random.randint(0,1))
    out = [prev]
    for i in range(num_ticks):
        tick = []
        tick.append(np.random.poisson(avg_car0))
        tick.append(np.random.poisson(avg_car1))
        tick.append(np.random.poisson(avg_ped0))
        tick.append(np.random.poisson(avg_ped1))
        out.append(tick)
    return out
"""
Creates the weights and biases for a neural network with the layer lengths using the normal distribution.
"""
def initNeuralNetwork(layer_lengths):
    network = []
    for i in range(len(layer_lengths) - 1):
        new_weights = np.random.normal(0, 1, (layer_lengths[i + 1], layer_lengths[i]))
        new_biases = np.random.normal(0, 0.5, (layer_lengths[i+1]))
        network.append([new_weights, new_biases])

    return network
"""
Mutates the neural network using the normal distribution to generate changes to the network
"""
def mutateNetwork(network, mutation_rate):
    for layer in network:
        layer[0] += np.random.normal(0, mutation_rate, layer[0].shape)
        layer[1] += np.random.normal(0, mutation_rate/3, layer[1].shape)
    return network
"""
Allows the parameters to pass through the network, producing an output 
"""
def forwardProp(network, params):
    cur = params
    for layer in network:
        cur = np.matmul(layer[0], cur) + layer[1]
        cur = np.tanh(cur)
    return cur