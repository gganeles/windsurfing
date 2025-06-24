from goProPumpingParser import main as parsePumps
from PumpDetector.labeledpumps010 import pumps
import pprint
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


paramTestList = [
    # {
    #     "cutOff": 40,
    #     "passBand": [0.001, 1],
    #     "fs": 201,
    #     "kernel_n": 1000,
    #     "pumpThresh": 0.45,
    #     "plots": True
    # },
    # {
    #     "cutOff": 50,
    #     "passBand": [0.001, 0.08],
    #     "fs": 201,
    #     "kernel_n": 3000,
    #     "pumpThresh": 0.30,
    #     "plots": True
    # },
    # {
    #     "cutOff": 40,
    #     "passBand": [0.001, 0.07],
    #     "fs": 201,
    #     "kernel_n": 2000,
    #     "pumpThresh": 0.37,
    #     "plots": True
    # },
    # {
    #     "cutOff": 40,
    #     "passBand": [0.001, 0.5],
    #     "fs": 201,
    #     "kernel_n": 3000,
    #     "pumpThresh": 0.30,
    #     "plots": True
    # },
    {
        "cutOff": 35,
        "passBand": [0.001, 0.5],
        "fs": 201,
        "kernel_n": 5000,
        "pumpThresh": 0.3,
        "plots": True
    },
]

for params in paramTestList:
    print("Params:")
    pprint.pprint(params)
    
    pumpGuess = parsePumps(params)
    
    for x in range(len(pumps)):
        x_1=datetime.fromisoformat(pumps[x]["start"])
        x_2=datetime.fromisoformat(pumps[x]["end"])
        plt.plot(np.array([x_1,x_1]),np.array([0,1]))
        plt.plot(np.array([x_2,x_2]),np.array([0,1]))
        


    plt.show()
    
    if (pumpGuess.shape[0]==len(pumps)+1):
        for i in range(len(pumps)):
            print("   Pump #",i+1)
            print("Labeled:  ",[pumps[i][x] for x in pumps[i]])
            print("Predicted:",[x.time().strftime('%H:%M:%S') for x in list(pumpGuess[["start","end"]].iloc[i+1])])
        print()
    else:
        print("Error: Pump count mismatch")
        print(len(pumps), pumpGuess.shape[0])