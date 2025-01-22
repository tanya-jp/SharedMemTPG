# %%
import numpy as np
import pandas as pd
import csv
import os
import sys
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
import matplotlib.pyplot as plt


# %%
def AddToPlot(reps, l):
    reps_mean = np.mean(reps, 0)
    reps_std = np.std(reps, 0)
    plt.fill_between(range(len(reps[0])), reps_mean+0.5*reps_std, 
      reps_mean-0.5*reps_std, alpha=0.5)
    plt.plot(range(len(reps[0])), reps_mean, label=l)

# %%
def process_csv_file(input_file, output_file):
    """
    Process a CSV file to truncate rows to the minimum row length and save the output.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the processed CSV file.
    """
    # Increase field size limit to handle large fields
    csv.field_size_limit(sys.maxsize)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the input file and process rows
    rows = []
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            # Split numbers by space and store them as lists
            rows.append(row[0].split())

    # Find the minimum row length
    min_length = min(len(row) for row in rows)

    # Truncate each row to the minimum length and join with spaces
    processed_rows = [" ".join(row[:min_length]) for row in rows]

    # Create and write to the output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for row in processed_rows:
            writer.writerow([row])

    print(f"Processed file saved as {output_file}. Minimum length of rows was {min_length}.")



# %%

path_1="/home/tanya/mcmaster/research/tpg/experiment_directories/mujoco-Hopper-mem"
path_2="/home/tanya/mcmaster/research/tpg/experiment_directories/mujoco-Hopper"

result_to_compare_raw="aux_2_ST_0_p0.csv"
result_to_compare="aux_2_ST_0_p0_new.csv"


process_csv_file(path_1 + "/" + result_to_compare_raw, path_1 + "/" + result_to_compare)
process_csv_file(path_2 + "/" + result_to_compare_raw, path_2 + "/" + result_to_compare)

# df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep='\s+', header=None, on_bad_lines='warn')
df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep=r'\s+', header=None, on_bad_lines='warn')
df2 = pd.read_csv(path_2 + "/" + result_to_compare, sep=r'\s+', header=None, on_bad_lines='warn')

fig = plt.figure(figsize=(6, 4))
AddToPlot(df1.to_numpy(), "with shared memory")
AddToPlot(df2.to_numpy(), "without shared memory")
plt.legend(loc="best")
plt.xlabel("Generations")
plt.ylabel("Complexity")
plt.show()

# %%
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # Or 'Qt5Agg'
# import matplotlib.pyplot as plt


# # %%
# def truncate_to_min_length(data):
#     min_length = min(len(row) for row in data)
#     return np.array([row[:min_length] for row in data])

# def AddToPlot(reps, label):
#     reps_min = np.min(reps, axis=0)
#     reps_max = np.max(reps, axis=0)
#     reps_mean = np.mean(reps, axis=0)
#     plt.fill_between(range(reps.shape[1]), reps_max, reps_min, alpha=0.5, label=label)
#     plt.plot(range(reps.shape[1]), reps_mean)

# # %%

# path_1="/home/tanya/mcmaster/research/tpg/experiment_directories/classic-control-all"
# #path_2 = "/root/tpg/experiment_directories/mujoco-Ant_deter_stateful_6_hebbian_50"
# # path_3 = "/root/tpg/experiment_directories/mujoco-Ant_deter_stateful_6_hebbian_100"

# result_to_compare = "aux_0_ST_0_p0.csv"

# df1 = pd.read_csv(path_1 + "/" + result_to_compare, sep='\s+', header=None, on_bad_lines='warn')
# data1 = truncate_to_min_length(df1.to_numpy())

# #

# # df3 = pd.read_csv(path_3 + "/" + result_to_compare, sep='\s+', header=None, on_bad_lines='warn')
# # data3 = truncate_to_min_length(df3.to_numpy())



# # Plotting
# fig = plt.figure(figsize=(8, 6))
# AddToPlot(data1, "Hebbian 0 ")
# #AddToPlot(data2, "Hebbian 25 ")
# # AddToPlot(data3, "Hebbian 50 ")

# plt.legend(loc="best")
# plt.title("Mujoco No Leg Breaking")
# plt.xlabel("Steps")
# plt.ylabel("Values")
# plt.show()
