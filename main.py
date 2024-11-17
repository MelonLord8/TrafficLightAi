import test_environment
import numpy as np
from copy import deepcopy
import torch
import random
import heapq

LAYER_LENGTHS = [4,8,8,1]

MUTATION_RATES = [-1.5,-0.75,-0.2,0.05,0.1]

NUM_GENERATIONS = 1000
POP_SIZE = 1000
TOURNAMENT_SIZE = 50
TOURNAMENT_TOP = 2
SWITCH_SCENARIO = 10

def init_network():
  out = torch.nn.Sequential(
    torch.nn.Linear(LAYER_LENGTHS[0], LAYER_LENGTHS[1]).requires_grad_(False),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(LAYER_LENGTHS[1], LAYER_LENGTHS[2]).requires_grad_(False),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(LAYER_LENGTHS[2], LAYER_LENGTHS[3]).requires_grad_(False)
  )
  for param in out.parameters():
    param.data = torch.rand_like(param) - 0.5
  return out

def mutate_child(child: torch.nn.Sequential, mutation_rate):
  for param in child.parameters():
    param.data += np.exp([mutation_rate])[0]*torch.randn_like(param)
  return child

def random_crossover(parent1, parent2):
  child = deepcopy(parent1)
  for params1, params2 in zip(child.parameters(), parent2.parameters()):
    if bool(random.getrandbits(1)):
      params1.data = deepcopy(params2.data)
  return child

def add_mutated_children(parent1, parent2, mutations, new_gen, num_children):
  for i in range(num_children):
    child = random_crossover(parent1, parent2)
    mutation_rate = mutations[random.getrandbits(1)]
    child = mutate_child(child, mutation_rate)
    new_gen.append([child, mutation_rate])

def next_gen(parents, new_gen):
  for i in range(len(parents)//2):
    parent1 = parents[2*i]
    parent2 = parents[2*i + 1]
    add_mutated_children(parent1[2], parent2[2], [parent1[3], parent2[3]], new_gen, 2*TOURNAMENT_SIZE //TOURNAMENT_TOP - 2)
    new_gen.extend([[deepcopy(parent1[2]), parent1[3]], [deepcopy(parent2[2]), parent2[3]] ] )
  random.shuffle(new_gen)

def new_mutations(rate_num_array):
  return [np.percentile(rate_num_array, 100*(i+1)/(len(rate_num_array) + 1)) for i in range(len(rate_num_array))]

def analyse(performance, control):
  print("Mean: " , np.mean(performance))
  print("Standard deviation: " , np.std(performance))
  print("Range: " , (np.max(performance) - np.min(performance)))
  print("Mean of mutation rates: " , np.mean(MUTATION_RATES))
  print("Standard deviation of mutation rates: " , np.std(MUTATION_RATES))
  print("Range of mutation rates: " , (np.max(MUTATION_RATES) - np.min(MUTATION_RATES)))
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
    generation.append([init_network(), MUTATION_RATES[POP_SIZE%len(MUTATION_RATES)] ] )
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
          mutation_num = []
          for i in range(POP_SIZE):
            mutation_num.append(generation[i][1])
          mutation_rates = new_mutations(mutation_num)
          for i in range(POP_SIZE):
            generation[i][1] = mutation_rates[POP_SIZE%len(mutation_rates)]

        for i in range(POP_SIZE):
          network = generation[i][0]
          fitness = test_environment.testFitnessOptmised(network, SCENARIOS)
          heapq.heappush(perf,[fitness, i, network, generation[i][1]])
          test_perf.append(fitness)
          if (i+1)%TOURNAMENT_SIZE == 0:
            parents.extend(heapq.nsmallest(TOURNAMENT_TOP,perf))
            perf = []

        print("Generation ", gen_num)
        analyse(test_perf, CONTROL)
        print("\n")

        generation = []
        # They should add up to X = 2*TOURNAMENT_SIZE / TOURNAMENT_TOP - 2
        next_gen(parents, generation)

train()