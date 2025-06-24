import matplotlib.pyplot as plt
import pandas as pd


# Read the data
df1 = pd.read_csv('GL_New_0010.csv')
df2 = pd.read_csv('GL0010.csv')


for x in df1.columns:
    if x == "unix-time [utc] (date)":
        continue
    try:
        print("Diff in ", x, ":", df1[x].sum() - df2[x].sum())
        print("Abs Diff in ", x, ":", abs(df1[x] - df2[x]).sum())
        print("Squared Diff in ", x, ":", (df1[x] - df2[x]).pow(2).sum())
        print()
        df2[x].plot(label='Old')
        df1[x].plot(label='New')
        plt.legend()
        plt.title(x)
        plt.show()
    except:
        pass
