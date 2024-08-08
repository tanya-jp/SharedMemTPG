import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys


def clip_generations(df):
  # Find the index of the first row with any NaN values
  first_nan_index = df[df.isna().any(axis=1)].index.min()

  if pd.isna(first_nan_index):  # Check if no NaN values were found
    truncated_df = df
  else:
    # Truncate the DataFrame
    truncated_df = df.loc[:first_nan_index-1]
  return truncated_df

def plot_experiment(arg, col, alph):
  file = sys.argv[arg]
  p = pathlib.Path(file)
  df = pd.read_csv(p, header=None, sep="\s+",names=range(max_gen))
  df = df.T
  # df = clip_generations(df)
  medians = df.median(axis='columns')
  plt.plot(df, color=col, alpha=alph)
#   plt.plot(medians.index, medians.values, linestyle='-', color=col)
  

max_gen = 100000
plot_name = sys.argv[1]
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plot_experiment(2,'b', 1.0)
plot_experiment(3,'r', 0.5)

# plt.show()
plt.savefig(plot_name)