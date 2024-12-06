# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def AddToPlot(reps):
    reps_mean = np.median(reps, 0)
    reps_std = np.std(reps, 0)
    plt.fill_between(range(len(reps[0])), reps_mean+0.5*reps_std, 
      reps_mean-0.5*reps_std, alpha=0.5)
    plt.plot(range(len(reps[0])), reps_mean)

# %%

path_1=("/home/skelly/experiments/mujoco-InvertedDoublePendulum/"
"mujoco-InvertedDoublePendulum-2024-12-04-11-57-40-b4b2daf/"
"mujoco-InvertedDoublePendulum-2024-12-04-11-57-40-b4b2daf_aux_0_ST_0_p0.rslt")

path_2 = ("/home/skelly/experiments/mujoco-InvertedDoublePendulum/"
"mujoco-InvertedDoublePendulum-2024-12-04-11-57-41-b4b2daf/"
"mujoco-InvertedDoublePendulum-2024-12-04-11-57-41-b4b2daf_aux_0_ST_0_p0.rslt")

max_generation=1000
df1 = pd.read_csv(path_1, sep='\s+', header=None, usecols=range(0,max_generation))
df2 = pd.read_csv(path_2, sep='\s+', header=None, usecols=range(0,max_generation))

fig = plt.figure(figsize=(6, 4))
AddToPlot(df1.to_numpy())
AddToPlot(df2.to_numpy())
plt.show()