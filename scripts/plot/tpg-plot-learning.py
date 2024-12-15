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
# path_1="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-14-23-05-58-7427007"
# path_2="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-14-23-06-05-7427007"
path_3="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-14-23-06-09-7427007"
# path_4="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-14-19-47-40-af5d32c"



result_to_compare="aux_0_ST_0_p0.csv"
max_generations=100
# df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df2 = pd.read_csv(path_2 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
df3 = pd.read_csv(path_3 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df4 = pd.read_csv(path_4 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))


fig = plt.figure(figsize=(6, 4))
# AddToPlot(df1.to_numpy(),"LGP")
# AddToPlot(df2.to_numpy(), "SBB")
AddToPlot(df3.to_numpy(), "TPG")
# AddToPlot(df4.to_numpy(), "TPGmi")

# df1.T.plot()
# df2.T.plot()
plt.show()