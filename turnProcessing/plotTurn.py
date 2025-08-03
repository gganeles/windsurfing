from turnAnalysis import turnEffectiveness
import pandas as pd
import numpy as np
import sys
import os

if not os.path.exists("paramsForTurnAnalysis.txt"):
    print("Error: paramsForTurnAnalysis.txt file not found")
    exit()

with open("paramsForTurnAnalysis.txt", "r", encoding="utf-16") as file:
    for line in file:
        if "TurnPeaksFileName" in line:
            turn_peaks_file = line.split(": ")[1].strip()
        if "SailmonCsvFileName" in line:
            sailmon_file = line.split(": ")[1].strip()
    if not turn_peaks_file or not sailmon_file:
        print("Error: Missing required files in paramsForTurnAnalysis.txt")

if len(sys.argv) < 2:
    print("Usage: python plotTurn.py <turn_numbers or all>")
    exit(1)

data = pd.read_csv(turn_peaks_file,index_col=0)
sailmonData = pd.read_csv(sailmon_file)

argument = sys.argv[1]

if argument == "all":
    turnEffectiveness(data, sailmonData, debug=True)
else:
    turns_to_plot = np.array(list(map(int, ",".join(sys.argv[1:]).split(","))))
    turnEffectiveness(data.iloc[turns_to_plot], sailmonData, debug=True)