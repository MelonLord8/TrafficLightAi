import numpy as np
import heapq
from copy import deepcopy

TIME_PER_TICK = 20
NUM_TICKS = 180

LAYER_LENGTHS = [7,32,32,1]

HIGH_MUTATION_RATE = 1
MEDIUM_MUTATION_RATE = 0.5
LOW_MUTATION_RATE = 0.25
ULTRA_LOW_MUTATION_RATE = 0.1

NORMAL_AVG_RATES = [TIME_PER_TICK/5,TIME_PER_TICK/5, TIME_PER_TICK/20, TIME_PER_TICK/20]
MORE_PED_AVG_RATES = [TIME_PER_TICK/5,TIME_PER_TICK/5, TIME_PER_TICK/4, TIME_PER_TICK/4]
HIGHWAY_INTERSECTION_RATES = [TIME_PER_TICK/3, TIME_PER_TICK/3, TIME_PER_TICK/30, TIME_PER_TICK/30]
BIASED_RATES = [TIME_PER_TICK/3,TIME_PER_TICK/6,TIME_PER_TICK/3,TIME_PER_TICK/6]

"""
Creates an array of length num_ticks + 1, 
First element being the previous states of the traffic light 0 being one orientation and 1 being the other
The rest being the number of incoming cars on the 0 lane, then 1 lane, then pedestrians in the same order.
"""
def makeScenario(avg_rates, num_ticks = NUM_TICKS):
    out = []
    for i in range(num_ticks):
        tick = [[],[]]
        tick[0].append(np.random.poisson(avg_rates[0]))
        tick[0].append(np.random.poisson(avg_rates[0]))
        tick[0].append(np.random.poisson(avg_rates[1]))
        tick[0].append(np.random.poisson(avg_rates[1]))
        tick[1].append(np.random.poisson(avg_rates[2]))
        tick[1].append(np.random.poisson(avg_rates[3]))
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
    cur = deepcopy(params)
    for layer in network:
        cur = np.matmul(layer[0], cur) + layer[1]
        cur = np.tanh(cur)
    return cur
"""
Puts the neural network through each tick of the scenario generated 
in each tick, cars and pedestrians are added to the end of their queue, then the program will check the direction of the intersection
allowing some cars to pass, as they pass the time spent is put through a function to see how annoyed they are.
All this data is then passed to the neural network, allowing it to decide wether or not to switch the direction.
Then we move onto the next tick
Returns the "total complaints" made
"""
def testScenario(network, scenario):
    total_complaints = 0
    last_switch = 0 #ticks since the last switch
    second_last = 0 #ticks since the second last switch

    cars = [[],[],[],[]]
    num_cars = [0,0,0,0]
    peds = [[],[]]
    num_peds = [0,0]
    direction = 0
    num_ticks = 0

    for tick in scenario:
        for i in range(4):
            cars[i].append([tick[0][i], num_ticks])
            num_cars[i] += tick[0][i]

        for i in range(2):
            peds[i].append((tick[1][i], num_ticks))
            num_cars[i] += tick[1][i]

        cars_passed = 0
        car_passed_0 = 0
        car_passed_1 = 0

        while cars[2*direction] and car_passed_0 < TIME_PER_TICK:
                cur_group = cars[2*direction][0]
                if cur_group[0] <= TIME_PER_TICK - car_passed_0:
                    total_complaints += cur_group[0]*(num_ticks - cur_group[1])
                    car_passed_0 += cur_group[0]
                    cars[2*direction].pop(0)
                else:
                    cars_passing = TIME_PER_TICK - car_passed_0
                    total_complaints += (cars_passing)*(num_ticks - cur_group[1])
                    car_passed_0 += cars_passing
                    cars[2*direction][0][0] -= cars_passing

        while cars[2*direction + 1] and car_passed_1 < TIME_PER_TICK:
                cur_group = cars[2*direction + 1][0]
                if cur_group[0] <= TIME_PER_TICK - car_passed_1:
                    total_complaints += cur_group[0]*(num_ticks - cur_group[1])
                    car_passed_1 += cur_group[0]
                    cars[2*direction + 1].pop(0)
                else:
                    cars_passing = TIME_PER_TICK - car_passed_1
                    total_complaints += (TIME_PER_TICK - car_passed_1)*(num_ticks - cur_group[1])
                    car_passed_1 += cars_passing
                    cars[2*direction + 1][0][0] -= cars_passing
        cars_passed += car_passed_0 + car_passed_1
        for ped_group in peds[direction]:
            total_complaints += 0.75*ped_group[0] * (num_ticks - ped_group[1])
        peds[direction] = []
        num_peds[direction] = 0
        car0_ratio = 0
        car1_ratio = 0
        if sum(num_cars) != 0:
            car0_ratio = (num_cars[2*direction] + num_cars[2*direction + 1]) / sum(num_cars)
            car1_ratio = 1 - car0_ratio
        ped0_ratio = 0 #not what you think
        ped1_ratio = 0
        if sum(num_peds) != 0:
            ped0_ratio = num_peds[direction]/sum(num_peds)
            ped1_ratio = 1 - ped0_ratio
        params = np.array([car0_ratio, car1_ratio, cars_passed/ (2*TIME_PER_TICK), ped0_ratio, ped1_ratio, np.tanh(last_switch), np.tanh(second_last)])
        switch = forwardProp(network, params)
        if switch > 0:
            if direction == 1:
                direction = 0
            else:
                direction = 1
            
            second_last = last_switch
            last_switch = 0
        last_switch += 1
        num_ticks += 1
    for i in range(4):
        if num_cars[i] > 0:
            for car_group in cars[i]:
                total_complaints += 1.5*car_group[0]*(num_ticks - car_group[1])
    for i in range(2):
        if num_peds[i] > 0:
            for ped_group in peds[i]:
                total_complaints += ped_group[0]*(num_ticks - ped_group[1])
    return total_complaints


def testScenarioWIthNormalTrafficLight(scenario):
    total_complaints = 0
    last_switch = 0 #ticks since the last switch
    second_last = 0 #ticks since the second last switch

    cars = [[],[],[],[]]
    num_cars = [0,0,0,0]
    peds = [[],[]]
    num_peds = [0,0]
    direction = 0
    num_ticks = 0

    for tick in scenario:
        for i in range(4):
            cars[i].append([tick[0][i], num_ticks])
            num_cars[i] += tick[0][i]

        for i in range(2):
            peds[i].append((tick[1][i], num_ticks))
            num_cars[i] += tick[1][i]

        cars_passed = 0
        car_passed_0 = 0
        car_passed_1 = 0

        while cars[2*direction] and car_passed_0 < TIME_PER_TICK:
                cur_group = cars[2*direction][0]
                if cur_group[0] <= TIME_PER_TICK - car_passed_0:
                    total_complaints += cur_group[0]*(num_ticks - cur_group[1])
                    car_passed_0 += cur_group[0]
                    cars[2*direction].pop(0)
                else:
                    cars_passing = TIME_PER_TICK - car_passed_0
                    total_complaints += (cars_passing)*(num_ticks - cur_group[1])
                    car_passed_0 += cars_passing
                    cars[2*direction][0][0] -= cars_passing

        while cars[2*direction + 1] and car_passed_1 < TIME_PER_TICK:
                cur_group = cars[2*direction + 1][0]
                if cur_group[0] <= TIME_PER_TICK - car_passed_1:
                    total_complaints += cur_group[0]*(num_ticks - cur_group[1])
                    car_passed_1 += cur_group[0]
                    cars[2*direction + 1].pop(0)
                else:
                    cars_passing = TIME_PER_TICK - car_passed_1
                    total_complaints += (TIME_PER_TICK - car_passed_1)*(num_ticks - cur_group[1])
                    car_passed_1 += cars_passing
                    cars[2*direction + 1][0][0] -= cars_passing
        cars_passed += car_passed_0 + car_passed_1
        for ped_group in peds[direction]:
            total_complaints += 0.75*ped_group[0] * (num_ticks - ped_group[1])
        peds[direction] = []
        num_peds[direction] = 0
        if direction == 1:
            direction = 0
        else:
            direction = 1
        num_ticks += 1
    for i in range(4):
        if num_cars[i] > 0:
            for car_group in cars[i]:
                total_complaints += 1.5*car_group[0]*(num_ticks - car_group[1])
    for i in range(2):
        if num_peds[i] > 0:
            for ped_group in peds[i]:
                total_complaints += ped_group[0]*(num_ticks - ped_group[1])
    return total_complaints
"""
Puts the network through an 10 hours worth of scenarios, and gathering the complaints, returns the best performing networks. 
"""
def testAndSelect(networks, top_ranking = 5):
    scenarios = [makeScenario(NORMAL_AVG_RATES),makeScenario(NORMAL_AVG_RATES), makeScenario(NORMAL_AVG_RATES),makeScenario(NORMAL_AVG_RATES),
                 makeScenario(MORE_PED_AVG_RATES), makeScenario(MORE_PED_AVG_RATES),
                 makeScenario(HIGHWAY_INTERSECTION_RATES), makeScenario(HIGHWAY_INTERSECTION_RATES),
                 makeScenario(BIASED_RATES), makeScenario(BIASED_RATES)]
    min_heap = []
    tie_breaker = 0
    control = 0
    for scenario in scenarios:
        control += testScenarioWIthNormalTrafficLight(scenario)
    for network in networks:
        total_complaints = 0
        for scenario in scenarios:
            total_complaints += testScenario(network, scenario)
        heapq.heappush(min_heap, (total_complaints, tie_breaker, network, total_complaints - control))
        tie_breaker += 1
    return heapq.nsmallest(top_ranking, min_heap)
"""
Generates the children networks by mutating the parents network slightly
"""
def generateNetworksFromParents(parents, generation):
    new_networks = deepcopy(parents)
    if generation < 100:
        for parent in parents:
            for i in range(6):
                new_networks.append(mutateNetwork(parent, HIGH_MUTATION_RATE))
            for i in range(3):
                new_networks.append(mutateNetwork(parent, MEDIUM_MUTATION_RATE))
    elif generation < 200:
        for parent in parents:
            for i in range(3):
                new_networks.append(mutateNetwork(parent, HIGH_MUTATION_RATE))
            for i in range(6):
                new_networks.append(mutateNetwork(parent, MEDIUM_MUTATION_RATE))
    elif generation < 500:
        for parent in parents:
            for i in range(2):
                new_networks.append(mutateNetwork(parent, HIGH_MUTATION_RATE))
            for i in range(3):
                new_networks.append(mutateNetwork(parent, MEDIUM_MUTATION_RATE))
            for i in range(4):
                new_networks.append(mutateNetwork(parent, LOW_MUTATION_RATE))
    else:
        for parent in parents:
            for i in range(2):
                new_networks.append(mutateNetwork(parent, MEDIUM_MUTATION_RATE))
            for i in range(3):
                new_networks.append(mutateNetwork(parent, LOW_MUTATION_RATE))
            for i in range(4):
                new_networks.append(mutateNetwork(parent, ULTRA_LOW_MUTATION_RATE))
    return new_networks

def __main__():
    generation = 0
    networks = []

    for i in range(50):
        networks.append(initNeuralNetwork(LAYER_LENGTHS))

    while generation <= 200:
        ranking = testAndSelect(networks, 5)
        parents = []
        for rank in ranking:
            parents.append(rank[2])
        networks = generateNetworksFromParents(parents, generation)
        print(generation, ranking[0][0], ranking[0][-1])
        generation += 1

__main__()