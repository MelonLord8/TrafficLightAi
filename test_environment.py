import torch
from copy import deepcopy
from math import log

TIME_PER_TICK = 20 #seconds
NUM_TICKS = 30
#Different rates that cars and pedestrains flow in each direction.
NORMAL_AVG_RATES = [TIME_PER_TICK/5,TIME_PER_TICK/5, TIME_PER_TICK/20, TIME_PER_TICK/20]
MORE_PED_AVG_RATES = [TIME_PER_TICK/5,TIME_PER_TICK/5, TIME_PER_TICK/4, TIME_PER_TICK/4]
HIGHWAY_INTERSECTION_RATES = [TIME_PER_TICK/2, TIME_PER_TICK/2, TIME_PER_TICK/30, TIME_PER_TICK/30]
BIASED_RATES = [TIME_PER_TICK/2,TIME_PER_TICK/3,TIME_PER_TICK/3,TIME_PER_TICK/6]

"""
Creates an array of length num_ticks,
Each element being a pair of matrices which represent the number of cars and pedestrians flowing in. 
"""
def makeTrainingSet(num_norm, num_ped, num_highway, num_biased , num_ticks = NUM_TICKS):
    out = []
    for i in range(num_ticks):
      cars = torch.stack([torch.cat([torch.poisson(NORMAL_AVG_RATES[0]*torch.ones(2)), torch.poisson(NORMAL_AVG_RATES[1]*torch.ones(2))]) for j in range(num_norm)] +
                         [torch.cat([torch.poisson(MORE_PED_AVG_RATES[0]*torch.ones(2)), torch.poisson(MORE_PED_AVG_RATES[1]*torch.ones(2))]) for j in range(num_ped)] +
                         [torch.cat([torch.poisson(HIGHWAY_INTERSECTION_RATES[0]*torch.ones(2)), torch.poisson(HIGHWAY_INTERSECTION_RATES[1]*torch.ones(2))]) for j in range(num_highway)] +
                         [torch.cat([torch.poisson(BIASED_RATES[0]*torch.ones(2)), torch.poisson(BIASED_RATES[1]*torch.ones(2))]) for j in range(num_biased)])

      peds = torch.stack([torch.cat([torch.poisson(NORMAL_AVG_RATES[2]*torch.ones(1)), torch.poisson(NORMAL_AVG_RATES[3]*torch.ones(1))]) for j in range(num_norm)] +
                         [torch.cat([torch.poisson(MORE_PED_AVG_RATES[2]*torch.ones(1)), torch.poisson(MORE_PED_AVG_RATES[3]*torch.ones(1))]) for j in range(num_ped)] +
                         [torch.cat([torch.poisson(HIGHWAY_INTERSECTION_RATES[2]*torch.ones(1)), torch.poisson(HIGHWAY_INTERSECTION_RATES[3]*torch.ones(1))]) for j in range(num_highway)] +
                         [torch.cat([torch.poisson(BIASED_RATES[2]*torch.ones(1)), torch.poisson(BIASED_RATES[3]*torch.ones(1))]) for j in range(num_biased)])
      out.append([cars, peds])
    return out
"""
Puts the neural network through each tick of the scenario generated
in each tick, cars and pedestrians are added to their respective matrices, then the program will check the direction of the intersection
allowing some cars to pass.

This utilises flipping the columns of a subtractor matrix depending on the direction decided upon.

All the data about the current state is then formatted and passed to the neural network, allowing it to decide wether or not to switch the direction.
Then we move onto the next tick, adding the total number of cars and pedestrains to the total complaints.
Returns the "total complaints" made
"""

def testFitnessOptmised(network, scenarios):
  scenarios = deepcopy(scenarios)
  num_complaints = 0
  num_scenarios = scenarios[0][0].shape[0]

  num_cars = torch.zeros(num_scenarios, 4)
  num_peds = torch.zeros(num_scenarios, 2)

  subtractors = torch.transpose(torch.stack([TIME_PER_TICK * torch.ones(num_scenarios),
                                             TIME_PER_TICK * torch.ones(num_scenarios),
                                             torch.zeros(num_scenarios),
                                             torch.zeros(num_scenarios)]), 0, 1)
  last_switch = torch.zeros(num_scenarios)

  for tick in scenarios:
    num_cars += tick[0]
    num_peds += tick[1]

    dir0_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[0])
    dir1_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[2])

    next_num_cars = torch.relu(num_cars - subtractors)
    cars_passed = torch.sum(num_cars - next_num_cars, 1)/TIME_PER_TICK
    num_cars = next_num_cars
    num_peds[dir0_indices, 0] = 0
    num_peds[dir1_indices, 1] = 0

    #handle getting the parameters, feeding it forward, and then flipping the subtractors

    car_ratio = torch.clone(num_cars)
    car_ratio[dir1_indices, 0] = 0
    car_ratio[dir1_indices, 1] = 0
    car_ratio[dir0_indices, 2] = 0
    car_ratio[dir0_indices, 3] = 0
    car_ratio = torch.clamp(torch.div(torch.sum(car_ratio, (1)), torch.sum(num_cars, (1)) + 0.1), min = 0, max = 1)

    other_ped = torch.tanh(torch.sum(num_peds, (1)))

    params = torch.transpose(torch.stack([car_ratio, other_ped, cars_passed, torch.tanh(last_switch)]), 0, 1)
    network_out = torch.squeeze(network(params))
    switch_indices = torch.squeeze(torch.nonzero(torch.relu(network_out)))

    subtractors[switch_indices] = torch.flip(subtractors[switch_indices], [-1])

    last_switch[switch_indices] = 0
    last_switch += 1

    dir0_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[0])
    dir1_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[2])

    time_weight = torch.clone(num_cars)
    time_weight[dir1_indices, 2] = 0
    time_weight[dir1_indices, 3] = 0
    time_weight[dir0_indices, 0] = 0
    time_weight[dir0_indices, 1] = 0
    num_complaints += (torch.sum(num_cars) + 0.75*torch.sum(num_peds) + torch.dot(torch.sum(time_weight, (1)), torch.exp(log(1.125)*last_switch) - 1)).item()

  return num_complaints

def testControlFitness(scenarios):
  num_complaints = 0
  num_scenarios = scenarios[0][0].shape[0]

  num_cars = torch.zeros(num_scenarios, 4)
  num_peds = torch.zeros(num_scenarios, 2)

  subtractors = torch.transpose(torch.stack([TIME_PER_TICK * torch.ones(num_scenarios),
                                             TIME_PER_TICK * torch.ones(num_scenarios),
                                             torch.zeros(num_scenarios),
                                             torch.zeros(num_scenarios)]), 0, 1)
  for tick in scenarios:
    num_cars += tick[0]
    num_peds += tick[1]

    dir0_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[0])
    dir1_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[2])

    num_cars = torch.relu(num_cars - subtractors)

    num_peds[dir0_indices, 0] = 0
    num_peds[dir1_indices, 1] = 0

    #handle getting the parameters, feeding it forward, and then flipping the subtractors

    subtractors = torch.flip(subtractors,[1])

    dir0_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[0])
    dir1_indices = torch.nonzero(torch.transpose(subtractors, 0, 1)[2])

    time_weight = torch.clone(num_cars)
    time_weight[dir1_indices, 2] = 0
    time_weight[dir1_indices, 3] = 0
    time_weight[dir0_indices, 0] = 0
    time_weight[dir0_indices, 1] = 0
    num_complaints += (torch.sum(num_cars) + 0.75*torch.sum(num_peds) + torch.sum(time_weight*0.125)).item()
  return num_complaints
