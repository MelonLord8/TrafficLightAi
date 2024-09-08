import torch
import test_environment
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import SNES

SCENARIOS = test_environment.makeTrainingSet(5,5,5,5)

def fitnessFunc(network):
    return test_environment.testFitness(network, SCENARIOS)

class MyNetwork (torch.nn.Module):
    def __init__(self, input_size = 7, output_size = 1):
        super(MyNetwork, self).__init__()   
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
problem = NEProblem(
    "min",
    MyNetwork,
    fitnessFunc
)

searcher = SNES(problem, stdev_init = 5)
searcher.step()
for i in range(100):
    print(i*10)
    print(searcher.status["pop_best_eval"], searcher.status["mean_eval"])
    searcher.run(num_generations = 10)
