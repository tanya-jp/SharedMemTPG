import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

file = sys.argv[1]
p = pathlib.Path(file)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

df = pd.read_csv(p)

# df.plot(plt.plot(df.Name, df.Marks))
df.plot(x='Time', y=['Target', 'Prediction'])
# plt.xticks(x, labels, rotation=45)
plt.legend()#['Target', 'Prediction'])
# plt.ylim(0,1)
plt.ylim(0,128)
plt.savefig(p.with_name(p.name.split('.')[0]).with_suffix('.png'))
