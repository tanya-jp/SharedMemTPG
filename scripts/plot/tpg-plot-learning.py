# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def AddToPlot(reps, lab):
    reps_min = np.min(reps, 0)
    reps_max = np.max(reps, 0)
    reps_mean = np.mean(reps, 0)
    reps_std = np.std(reps, 0)
    plt.fill_between(range(len(reps[0])), reps_mean+0.5*reps_std, 
      reps_mean-0.5*reps_std, alpha=0.5)
    # plt.fill_between(range(len(reps[0])), reps_min, 
    #   reps_max, alpha=0.5)
    plt.plot(range(len(reps[0])), reps_max, label=lab)
    plt.legend()

# %%
# path_1="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-21-20-29-03-cc759d5"
path_1="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-22-18-27-11-b0181ad" #TPG
path_2="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-22-09-42-06-b0181ad" #TPGp
path_3="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-22-09-42-11-b0181ad" #TPGd



result_to_compare="aux_0_ST_0_p0.csv"
max_generations=500
df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
df2 = pd.read_csv(path_2 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
df3 = pd.read_csv(path_3 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df4 = pd.read_csv(path_4 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))


fig = plt.figure(figsize=(6, 4))
AddToPlot(df1.to_numpy(),"TPG")
AddToPlot(df2.to_numpy(), "TPGp")
AddToPlot(df3.to_numpy(), "TPGd")
# AddToPlot(df4.to_numpy(), "TPGp")

plt.show()