import test_environment
from torch import nn, tanh, Tensor, reshape
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
from evotorch.logging import StdOutLogger

LAYER_LENGTHS = [7,8,2]
NUM_GENES = LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2]

SCENARIOS = test_environment.makeTrainingSet(0,0,0,1)

class MyNetwork (nn.Module):
    def __init__(self, values):
        super(MyNetwork, self).__init__()   

        self.fc1 = nn.Linear(LAYER_LENGTHS[0], LAYER_LENGTHS[1])
        self.fc1.weight.data = values[0]
        self.fc1.bias.data = values[1]

        self.fc2 = nn.Linear(LAYER_LENGTHS[1], LAYER_LENGTHS[2])
        self.fc2.weight.data = values[2]
        self.fc2.bias.data = values[3]


    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), 0.1)
        x = tanh(self.fc2(x))
        return x

def genoToPheno(gene_tensor : Tensor):
    weight1 = reshape(gene_tensor[0 : LAYER_LENGTHS[0]*LAYER_LENGTHS[1]],(LAYER_LENGTHS[1], LAYER_LENGTHS[0]))
    bias1 = gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1]: LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1]]
    
    weight2 = reshape(gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1]: 
                          LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2]], 
                          (LAYER_LENGTHS[2], LAYER_LENGTHS[1]))
    bias2 = gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2]:
                        LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2]]
        
    return [weight1, bias1, weight2, bias2]

def fitnessFunc(genes):
    network = MyNetwork(genoToPheno(genes))
    return test_environment.testFitness(network, SCENARIOS)
    
problem = Problem(
    "min",
    fitnessFunc,
    solution_length= NUM_GENES,
    initial_bounds= (-3, 3),
    num_actors= 2
)

searcher = GeneticAlgorithm(
    problem,
    popsize=1500,
    operators=[
        OnePointCrossOver(problem, tournament_size= 300),
        GaussianMutation(problem, stdev=5)
    ]
)
CONTROL = test_environment.testControlFitness(SCENARIOS)
_ = StdOutLogger(searcher)

searcher.step()
for i in range(1000):
    SCENARIOS = test_environment.makeTrainingSet(0,0,0,1)
    print(CONTROL)
    searcher.step()
