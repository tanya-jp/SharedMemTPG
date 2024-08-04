import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

max_gen = 100000
file = sys.argv[1]
p = pathlib.Path(file)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# plt.figure(figsize=(16, 8), dpi=150) 

df = pd.read_csv(p, header=None, sep="\s+",names=range(max_gen))
# print(df.to_string())

df = df.T

# Find the index of the first row with any NaN values
first_nan_index = df[df.isna().any(axis=1)].index.min()

if pd.isna(first_nan_index):  # Check if no NaN values were found
    truncated_df = df
else:
    # Truncate the DataFrame
    truncated_df = df.loc[:first_nan_index-1]

medians = truncated_df.median(axis='columns')
plt.plot(truncated_df, color='blue')
plt.plot(medians.index, medians.values, linestyle='-', color='red')
plt.show()
medians
# plt.savefig(p.with_name(p.name.split('.')[0]).with_suffix('.pdf'))