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