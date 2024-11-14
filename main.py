import test_environment
import numpy as np
from copy import deepcopy
import torch
import random
import heapq

LAYER_LENGTHS = [4,8,8,1]

HIGH_MUTATION_RATE = 0.5
MEDIUM_MUTATION_RATE = 0.05
LOW_MUTATION_RATE = 0.01

NUM_GENERATIONS = 1000
POP_SIZE = 1000
TOURNAMENT_SIZE = 50
TOUNRAMENT_TOP = 2
SWITCH_SCENARIO = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_network():
  out = torch.nn.Sequential(
    torch.nn.Linear(LAYER_LENGTHS[0], LAYER_LENGTHS[1]).requires_grad_(False).to(device),
    torch.nn.LeakyReLU(0.1).to(device),
    torch.nn.Linear(LAYER_LENGTHS[1], LAYER_LENGTHS[2]).requires_grad_(False).to(device),
    torch.nn.LeakyReLU(0.1).to(device),
    torch.nn.Linear(LAYER_LENGTHS[2], LAYER_LENGTHS[3]).requires_grad_(False).to(device)
  )
  for param in out.parameters():
    param.data = torch.rand_like(param) - 0.5
  return out

def mutate_child(child: torch.nn.Sequential, mutation_rate):
  for param in child.parameters():
    param.data += mutation_rate*torch.randn_like(param)
  return child

def random_crossover(parent1, parent2):
  child = deepcopy(parent1)
  for params1, params2 in zip(child.parameters(), parent2.parameters()):
    if bool(random.getrandbits(1)):
      params1.data = deepcopy(params2.data)
  return child

def add_mutated_children(parent1, parent2, new_gen, rate_num_arr):
  for pair in rate_num_arr:
    for i in range(pair[0]):
      child = random_crossover(parent1, parent2)
      child = mutate_child(child, pair[1])
      new_gen.append(child)

def next_gen(parents, new_gen, rate_num_arr):
  for i in range(len(parents)//2):
    add_mutated_children(parents[2*i][2], parents[2*i + 1][2], new_gen, rate_num_arr)
    new_gen.extend([deepcopy(parents[2*i][2]), deepcopy(parents[2*i + 1][2])])

def analyse(performance, control):
  print("Mean: " , np.mean(performance))
  print("Standard deviation: " , np.std(performance))
  print("Range: " , (np.max(performance) - np.min(performance)))
  print("Best performance: " , np.min(performance))
  print("Control performance: " , control)
  print("Median Relative Performance: ", (np.floor(np.median(performance) * 100 / control)) , "%")
  print("Mean Relative Performance: " , (np.floor(np.mean(performance) * 100 / control)) , "%")
  print("Best Relative Performance: " , (np.floor(np.min(performance) * 100 / control)) , "%")

def networks_equal(n1, n2):
  for p1,p2 in zip(n1.parameters(), n2.parameters()):
    if not torch.equal(p1, p2):
      return False
  return True

def train():
  generation = []
  gen_num = 0
  for i in range(POP_SIZE):
    generation.append(init_network())
  SCENARIOS = test_environment.makeTrainingSet(4,4,4,4)
  CONTROL = test_environment.testControlFitness(SCENARIOS)
  with torch.no_grad():
    while True:
        gen_num += 1
        parents = []
        perf = []
        test_perf = []
        
        if gen_num % SWITCH_SCENARIO == 0:
          SCENARIOS = test_environment.makeTrainingSet(4,4,4,4)
          CONTROL = test_environment.testControlFitness(SCENARIOS)

        for i in range(POP_SIZE):
          network = generation[i]
          fitness = test_environment.testFitnessOptmised(network, SCENARIOS)
          heapq.heappush(perf,[fitness, i, network])
          test_perf.append(fitness)
          if (i+1)%TOURNAMENT_SIZE == 0:
            parents.extend(heapq.nsmallest(TOUNRAMENT_TOP,perf))
            perf = []

        print("Generation ", gen_num)
        analyse(test_perf, CONTROL)
        print("\n")

        generation = []
        # They should add up to X = 2*TOURNAMENT_SIZE / TOURNAMENT_TOP - 2
        next_gen(parents, generation, [(16, HIGH_MUTATION_RATE), (16, MEDIUM_MUTATION_RATE), (16, LOW_MUTATION_RATE)])

train()