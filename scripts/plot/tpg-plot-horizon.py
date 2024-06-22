import pandas as pd
import matplotlib.pyplot as plt
import sys
print ('argument list', sys.argv)
file = sys.argv[1]

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# headers = ['Step','Target', 'Prediction']

df = pd.read_csv(file,header=None)

# df.set_index('Step').plot()
df.plot()

plt.show()
