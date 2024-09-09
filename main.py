import test_environment
from torch import nn, tanh, Tensor, reshape
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
from evotorch.logging import StdOutLogger

LAYER_LENGTHS = [7,8,8,2]
NUM_GENES = LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2] + LAYER_LENGTHS[2]*LAYER_LENGTHS[3] + LAYER_LENGTHS[3]

SCENARIOS = test_environment.makeTrainingSet(1,1,1,2)

class MyNetwork (nn.Module):
    def __init__(self, values):
        super(MyNetwork, self).__init__()   

        self.fc1 = nn.Linear(LAYER_LENGTHS[0], LAYER_LENGTHS[1])
        self.fc1.weight.data = values[0]
        self.fc1.bias.data = values[1]

        self.fc2 = nn.Linear(LAYER_LENGTHS[1], LAYER_LENGTHS[2])
        self.fc2.weight.data = values[2]
        self.fc2.bias.data = values[3]

        self.fc3 = nn.Linear(LAYER_LENGTHS[2], LAYER_LENGTHS[3])
        self.fc3.weight.data = values[4]
        self.fc3.bias.data = values[5]

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), 0.1)
        x = nn.functional.leaky_relu(self.fc2(x))
        x = tanh(self.fc3(x))
        return x

def genoToPheno(gene_tensor : Tensor):
    weight1 = reshape(gene_tensor[0 : LAYER_LENGTHS[0]*LAYER_LENGTHS[1]],(LAYER_LENGTHS[1], LAYER_LENGTHS[0]))
    bias1 = gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1]: LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1]]
    
    weight2 = reshape(gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1]: 
                          LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2]], 
                          (LAYER_LENGTHS[2], LAYER_LENGTHS[1]))
    bias2 = gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2]:
                        LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2]]
    
    weight3 = reshape(gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2]:
                                        LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2] + LAYER_LENGTHS[2]*LAYER_LENGTHS[3]], 
                            (LAYER_LENGTHS[3], LAYER_LENGTHS[2]))
    bias3 = gene_tensor[LAYER_LENGTHS[0]*LAYER_LENGTHS[1] + LAYER_LENGTHS[1] + LAYER_LENGTHS[1]*LAYER_LENGTHS[2] + LAYER_LENGTHS[2] + LAYER_LENGTHS[2]*LAYER_LENGTHS[3]:
                        NUM_GENES]
        
    return [weight1, bias1, weight2, bias2, weight3, bias3]

def fitnessFunc(genes):
    network = MyNetwork(genoToPheno(genes))
    return test_environment.testFitness(network, SCENARIOS)
    
problem = Problem(
    "min",
    fitnessFunc,
    solution_length= NUM_GENES,
    initial_bounds= (-3, 3)
)

searcher = GeneticAlgorithm(
    problem,
    popsize=100,
    operators=[
        OnePointCrossOver(problem, tournament_size=4),
        GaussianMutation(problem, stdev=0.1),
    ],
)
CONTROL = test_environment.testControlFitness(SCENARIOS)
_ = StdOutLogger(searcher)

searcher.step()
for i in range(1000):
    print(CONTROL)
    searcher.step()
