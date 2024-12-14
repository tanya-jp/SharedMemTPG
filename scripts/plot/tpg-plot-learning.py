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

# path_1="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-20-47-56-ded0204"
# path_2="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-20-48-01-ded0204"
path_3="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-20-48-06-ded0204"
# path_4="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-20-58-14-ded0204"
# path_5="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-21-00-04-ded0204"
path_6="~/experiments/mujoco-Reacher/mujoco-Reacher-2024-12-13/mujoco-Reacher-2024-12-12-21-01-36-ded0204"

result_to_compare="aux_0_ST_0_p0.csv"
max_generations=1000
# df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df2 = pd.read_csv(path_2 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
df3 = pd.read_csv(path_3 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df4 = pd.read_csv(path_4 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
# df5 = pd.read_csv(path_5 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))
df6 = pd.read_csv(path_6 + "/" + result_to_compare, sep='\s+', header=None, usecols=range(0,max_generations))

fig = plt.figure(figsize=(6, 4))
# AddToPlot(df1.to_numpy(),"1")
# AddToPlot(df2.to_numpy(), "2")
AddToPlot(df3.to_numpy(), "3")
# AddToPlot(df4.to_numpy(), "4")
# AddToPlot(df5.to_numpy(), "5")
AddToPlot(df6.to_numpy(), "6")
# df1.T.plot()
# df2.T.plot()
plt.show()