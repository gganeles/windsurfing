import pandas as pd

from mat4py import loadmat

data = loadmat('detected_turnPeaks1.mat')
# Convert the variable to a DataFrame if it's a 2D array or list
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('detected_turnPeaks.csv', index=False)