import math
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

display_start = 10

def calculate_avg_reward(series, start_point, end_point):
    pass

def calculate_avg_rewards(df, column, series_name):
    if column == "Step":
        return None

    if "MIN" in column or "MAX" in column:
        return None

    print("-------------------------------")
    print(series_name)
    series = pd.Series(df[column]).dropna().reset_index(drop=True)

    avg_reward_points = [0, 0.25*last_cutoff, 0.5 * last_cutoff, 0.75 * last_cutoff, last_cutoff]
    episode_range_values = "episode ranges & "
    avg_reward_string = str(series_name) + " & "

    for i in range(len(avg_reward_points) - 1):
        start_point = math.floor(avg_reward_points[i])
        end_point = math.floor(avg_reward_points[i+1])
        episode_range_values += str(start_point) + "-" + str(end_point) + " & "

        avg_reward = series[start_point:end_point].mean() * batch_size
        avg_reward_string += f'{avg_reward:.2f}' + " & "

        #print("x_1:", str(start_point), "x_2:", str(end_point), ":", str(avg_reward))

    avg_reward = series[avg_reward_points[0]:avg_reward_points[-1]].mean() * batch_size
    episode_range_values += str(0) + "-" + str(last_cutoff) + " \\\\"
    avg_reward_string += f'{avg_reward:.2f}' + "  \\\\"
    print(avg_reward_string)
    print(episode_range_values)

    #print("x_1:", str(avg_reward_points[0]), "x_2:", str(avg_reward_points[-1]), ":", str(avg_reward))

def smooth_data(df, column):
    if column == "Step":
        return None

    if "MIN" in column or "MAX" in column:
        return None
    running_average = []
    series = pd.Series(df[column]).dropna().reset_index(drop=True)
    results = []
    counter = 0

    for index, row in series.items():
        if counter > max_generation_display:
            break
        counter += 1
        if counter < start_point:
            continue
        running_average.append(row*batch_size)
        if len(running_average) >= running_average_max:
            running_average.pop(0)
        if counter >= display_start:
            results.append(sum(running_average) / len(running_average))

    print(len(results))
    new_series = pd.Series(results)
    return new_series




#dirt_no_wait_step = pd.read_csv("dirt_models/AC/no-wait-step-1e4/rewards.csv")
#dirt_wait_step_1 = pd.read_csv("dirt_models/AC/wait-step-1-1e4/rewards.csv")
#dirt_wait_step_2 = pd.read_csv("dirt_models/AC/wait-step-2-1e4/rewards.csv")

#dirt_acdfp = pd.read_csv("dirt_models/AC/acdfp.csv")
#dirt_eacdfp = pd.read_csv("dirt_models/AC/eacdfp.csv")
#dirt_no_wait_mean_beta = pd.read_csv("dirt_models/AC/no-wait-mean-beta09466-alpha09044-new-init.csv")

#dirt_wait_mean_beta_new_init = pd.read_csv("dirt_models/AC/wait-mean-1-1e3-beta09466-alpha09044.csv")
#dirt_mean_alpha_09_1 = pd.read_csv("dirt_models/AC/wait-mean-beta09-alpha09-new-init-1.csv")
#dirt_mean_alpha_095_1 = pd.read_csv("dirt_models/AC/wait-mean-beta095-alpha095-new-init-1.csv")
#dirt_mean_alpha_09_2 = pd.read_csv("dirt_models/AC/wait-mean-beta09-alpha09-new-init-2.csv")
#dirt_mean_alpha_095_2 = pd.read_csv("dirt_models/AC/wait-mean-beta095-alpha095-new-init-2.csv")
#dirt_wait_mean_beta_2 = pd.read_csv("dirt_models/AC/wait-mean-2-1e3-beta09466-alpha09044.csv")


#dirt_mean_ddfp = pd.read_csv("dirt_models/DFP/ddfp-corrected.csv")
#dirt_dfp_1 = pd.read_csv("dirt_models/DFP/dfp-1-max.csv")
#dirt_dfp_2 = pd.read_csv("dirt_models/DFP/dfp-2-max.csv")
#dirt_dfp_3 = pd.read_csv("dirt_models/DFP/dfp-mean.csv")
#dirt_lehddfp = pd.read_csv("dirt_models/DFP/lehddfp.csv")
#dirt_max_ddfp = pd.read_csv("dirt_models/DFP/max-ddfp.csv")



#wood_ac_1 = pd.read_csv("wood_models/AC/ac-1.csv")
#wood_ac_2 = pd.read_csv("wood_models/AC/ac-2.csv")
#wood_acdfp = pd.read_csv("wood_models/AC/acdfp.csv")
#wood_eacdfp = pd.read_csv("wood_models/AC/eacdfp.csv")



#wood_acdfp = pd.read_csv("wood_models/AC/acdfp.csv")

running_average_max = 1100

"""
# create step sync vs. async, dirt, rewards
file_paths = ["dirt_models/AC/no-wait-step-1e4/rewards.csv", "dirt_models/AC/wait-step-1-1e4/rewards.csv",
              "dirt_models/AC/wait-step-2-1e4/rewards.csv"]
line_styles = ["-", "-", "--"]
series_names = ["A3C", "A2C (1)", "A2C (2)"]
colors = ["red", "blue", "blue"]
title = "Synced vs un-synced Actor Critic with step update"
save_path = "images/sync-vs-unsync.png"
"""

"""
# create new vs. old init
file_paths = ["dirt_models/AC/wait-mean-beta09466-alpha09044-old-init/rewards.csv",
              "dirt_models/AC/wait-mean-beta09466-alpha09044-new-init/rewards.csv"]
line_styles = ["-", "-"]
series_names = ["old-init", "new-init"]
colors = ["red", "blue"]
title = "new init vs old init with accumulated update"
save_path = "images/old-vs-new-init.png"
"""

"""
# create new vs. old init gradients
file_paths = ["dirt_models/AC/wait-mean-beta09466-alpha09044-old-init/gradient_size.csv",
              "dirt_models/AC/wait-mean-beta09466-alpha09044-new-init/gradient_size.csv"]
line_styles = ["-", "-"]
series_names = ["old-init", "new-init"]
colors = ["red", "blue"]
title = "new init vs old init with accumulated update gradient size"
save_path = "images/old-vs-new-init-gradient-size.png"
"""

"""
# sync vs async mean method
file_paths = ["dirt_models/AC/no-wait-mean-beta09466-alpha09044-new-init/rewards.csv",
              "dirt_models/AC/wait-mean-beta09466-alpha09044-old-init/rewards.csv"]
line_styles = ["-", "-"]
series_names = ["A3C", "A2C"]
colors = ["red", "blue"]
title = "sync vs async with accumulated update"
save_path = "images/batched-sync-vs-un-sync.png"
"""

"""
# different hyperparameters settings
file_paths = ["dirt_models/AC/wait-mean-beta09-alpha09-new-init-1/rewards.csv",
               "dirt_models/AC/wait-mean-beta09-alpha09-new-init-2/rewards.csv",
               "dirt_models/AC/wait-mean-beta095-alpha095-new-init-1/rewards.csv",
               "dirt_models/AC/wait-mean-beta095-alpha095-new-init-2/rewards.csv"]
line_styles = ["-", "--", "-", "--"]
series_names = ["alpha=0.9 (1)", "alpha=0.9 (2)", "alpha=0.95 (1)", "alpha=0.95 (2)"]
colors = ["blue", "blue", "red", "red"]
title = "Average Gradient size decay of RMSProp"
save_path = "images/alpha-09-vs-095.png"
"""

"""
# different actor models, dirt, rewards
file_paths = ["dirt_models/AC/wait-mean-beta095-alpha095-new-init-1/rewards.csv",
              "dirt_models/AC/wait-mean-beta095-alpha095-new-init-2/rewards.csv",
              "dirt_models/AC/acdfp/rewards.csv", "dirt_models/AC/eacdfp/rewards.csv"]
line_styles = ["-","--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "ACDFP", "EACDFP"]
colors = ["red", "red", "blue", "green"]
title = "Actor critic models with same hyperparameters"
save_path = "images/ac-vs-acdfp-vs-eacdfp.png"
"""


# different dfp models, dirt, rewards

file_paths = ["dirt_models/DFP/dfp-1-max/rewards.csv",
               "dirt_models/DFP/dfp-2-max/rewards.csv",
               "dirt_models/DFP/lehddfp/rewards.csv",
               "dirt_models/DFP/max-ddfp/rewards.csv",
               "dirt_models/DFP/mean-ddfp/rewards.csv"]
line_styles = ["-", "--", "-", "-", "--"]
series_names = ["DFP (1)", "DFP (2)", "LEHDDFP", "Max-DDFP", "Mean-DDFP"]
colors = ["red", "red", "green", "orange", "blue"]
title = "Direct Future Prediction models with same hyperparameters"
save_path = "images/dfp-vs-ddfp-vs-lehddfp.png"



"""
# wood actor models, combined, rewards

file_paths = ["wood_models/AC/ac-1/combined/rewards.csv",
              "wood_models/AC/ac-2/combined/rewards.csv",
              "wood_models/AC/acdfp/combined/rewards.csv",
              "wood_models/AC/eacdfp/combined/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "ACDFP", "EACDFP"]
colors = ["red", "red", "blue", "green"]
title = "Wood + dirt"
save_path = "images/wood/AC/combined.png"
running_average_max = 2200
"""

"""
# wood actor models, dirt, rewards

file_paths = ["wood_models/AC/ac-1/dirt/rewards.csv",
              "wood_models/AC/ac-2/dirt/rewards.csv",
              "wood_models/AC/acdfp/dirt/rewards.csv",
              "wood_models/AC/eacdfp/dirt/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "ACDFP", "EACDFP"]
colors = ["red", "red", "blue", "green"]
title = "Dirt"
save_path = "images/wood/AC/dirt.png"
running_average_max = 1100
"""

"""
# wood actor models, wood, rewards
file_paths = ["wood_models/AC/ac-1/wood/rewards.csv",
              "wood_models/AC/ac-2/wood/rewards.csv",
              "wood_models/AC/acdfp/wood/rewards.csv",
              "wood_models/AC/eacdfp/wood/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "ACDFP", "EACDFP"]
colors = ["red", "red", "blue", "green"]
title = "Wood"
save_path = "images/wood/AC/wood.png"
running_average_max = 1100
"""

"""
# wood actor models, zero-shot, rewards
file_paths = ["wood_models/AC/ac-1/zero_shot/rewards.csv",
              "wood_models/AC/ac-2/zero_shot/rewards.csv",
              "wood_models/AC/acdfp/zero_shot/rewards.csv",
              "wood_models/AC/eacdfp/zero_shot/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "ACDFP", "EACDFP"]
colors = ["red", "red", "blue", "green"]
title = "Zero shot"
save_path = "images/wood/AC/zero_shot.png"
running_average_max = 220
"""

"""
# wood staggered actor models, zero-shot, rewards
file_paths = ["wood_models/AC/ac-1/combined/rewards.csv",
              "wood_models/AC/ac-2/combined/rewards.csv",
              "wood_models/AC/staggered-AC-1/combined/rewards.csv",
              "wood_models/AC/staggered-AC-2/combined/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "Staggered A2C (1)", "Staggered A2C (2)"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/staggered_rewards.png"
running_average_max = 2200
"""

"""
# wood staggered actor models, zero-shot, rewards
file_paths = ["wood_models/AC/ac-1/advantage/rewards.csv",
              "wood_models/AC/ac-2/advantage/rewards.csv",
              "wood_models/AC/staggered-AC-1/advantage/rewards.csv",
              "wood_models/AC/staggered-AC-2/advantage/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "Staggered A2C (1)", "Staggered A2C (2)"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/staggered_advantage.png"
running_average_max = 2200
"""

"""
# wood staggered actor models, action 8 logits
file_paths = ["wood_models/AC/ac-1/action_8_loggits/rewards.csv",
              "wood_models/AC/ac-2/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-1/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-2/action_8_loggits/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "Staggered A2C (1)", "Staggered A2C (2)"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/staggered_action_8_logits.png"
running_average_max = 2200
"""

"""
# wood staggered actor models, large action 8 logits
file_paths = ["wood_models/AC/low-gamma-ac-1/action_8_loggits/rewards.csv",
              "wood_models/AC/low-gamma-ac-1/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-1/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-2/action_8_loggits/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = [r"$𝛾=0.9655 \, (1)$", r"$𝛾=0.9655 \, (2)$", r"$𝛾=0.99 \, (1)$", r"$𝛾=0.99 \, (2)$"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/staggered_action_8_large_logits.png"
running_average_max = 2200
"""

"""
# wood staggered actor models, action 8 logits
file_paths = ["wood_models/AC/ac-1/action_8_loggits/rewards.csv",
              "wood_models/AC/ac-2/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-1/action_8_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-2/action_8_loggits/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "Staggered A2C (1)", "Staggered A2C (2)"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/staggered_action_8_logits.png"
running_average_max = 2200
"""


"""
# wood staggered actor models, zero-shot, rewards
file_paths = ["wood_models/AC/ac-1/action_0_loggits/rewards.csv",
              "wood_models/AC/ac-2/action_0_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-1/action_0_loggits/rewards.csv",
              "wood_models/AC/staggered-AC-2/action_0_loggits/rewards.csv"]
line_styles = ["-", "--", "-", "-"]
series_names = ["A2C (1)", "A2C (2)", "Staggered A2C (1)", "Staggered A2C (2)"]
colors = ["red", "red", "blue", "blue"]
title = "Zero shot"
save_path = "images/wood/AC/action_0_loggits.png"
running_average_max = 2200
"""

"""
# wood DFP models, combined, rewards
file_paths = ["wood_models/DFP/dfp-1/combined/rewards.csv",
              "wood_models/DFP/lehddfp/combined/rewards.csv",
              "wood_models/DFP/max-ddfp/combined/rewards.csv",
              "wood_models/DFP/mean-ddfp/combined/rewards.csv"]
line_styles = ["-", "-", "-", "-"]
series_names = ["DFP", "LEHDDFP", "Max-DDFP", "Mean-DDFP"]
colors = ["red", "green", "orange", "blue"]
title = "Wood + dirt"
save_path = "images/wood/DFP/combined.png"
running_average_max = 2200
"""

"""
# wood DFP models, dirt, rewards
file_paths = ["wood_models/DFP/dfp-1/dirt/rewards.csv",
              "wood_models/DFP/lehddfp/dirt/rewards.csv",
              "wood_models/DFP/max-ddfp/dirt/rewards.csv",
              "wood_models/DFP/mean-ddfp/dirt/rewards.csv"]
line_styles = ["-", "-", "-", "-"]
series_names = ["DFP", "LEHDDFP", "Max-DDFP", "Mean-DDFP"]
colors = ["red", "green", "orange", "blue"]
title = "Dirt"
save_path = "images/wood/DFP/dirt.png"
running_average_max = 1100
"""

"""
# wood DFP models, wood, rewards
file_paths = ["wood_models/DFP/dfp-1/wood/rewards.csv",
              "wood_models/DFP/lehddfp/wood/rewards.csv",
              "wood_models/DFP/max-ddfp/wood/rewards.csv",
              "wood_models/DFP/mean-ddfp/wood/rewards.csv"]
line_styles = ["-", "-", "-", "-"]
series_names = ["DFP", "LEHDDFP", "Max-DDFP", "Mean-DDFP"]
colors = ["red", "green", "orange", "blue"]
title = "Wood"
save_path = "images/wood/DFP/wood.png"
running_average_max = 1100
"""

"""
# wood DFP models, zero-shot, rewards
file_paths = ["wood_models/DFP/dfp-1/zero_shot/rewards.csv",
              "wood_models/DFP/lehddfp/zero_shot/rewards.csv",
              "wood_models/DFP/max-ddfp/zero_shot/rewards.csv",
              "wood_models/DFP/mean-ddfp/zero_shot/rewards.csv"]
line_styles = ["-", "-", "-", "-"]
series_names = ["DFP", "LEHDDFP", "Max-DDFP", "Mean-DDFP"]
colors = ["red", "green", "orange", "blue"]
title = "Zero shot"
save_path = "images/wood/DFP/zero_shot.png"
running_average_max = 220
"""


#data_frames = [df1, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18]
#data_frames = [df13, df14, df15, df16, df17, df18]


max_generation_display = 2000000
series_counter = 0
start_point = 0

upper_dirt_y_lim_ac = 30
upper_dirt_y_lim_dfp = 30

upper_wood_y_lim_dfp = 35
upper_wood_y_lim_ac = 30

batch_size = 256
upper_y_lim = upper_wood_y_lim_ac


last_cutoff = 7078



for file_path in file_paths:
    df = pd.read_csv(file_path)
    for column in df.columns:
        calculate_avg_rewards(df, column, series_names[series_counter])
        new_series = smooth_data(df, column)
        if new_series is not None:

            plt.plot(new_series, label=series_names[series_counter], linestyle=line_styles[series_counter], color=colors[series_counter])
            #plt.plot(new_series)
            series_counter += 1
        pass

#df = pd.DataFrame(reduced_data).T

#df.plot(legend=True)
plt.grid(visible=True)
plt.ylim(0.0, upper_y_lim)
#plt.ylim(-0.5, 0.5)
plt.xlim(0.0)
plt.xlabel("Episode")
plt.ylabel("Reward per episode")
#plt.yscale("log")
plt.legend()
#plt.title(title)
#plt.show()
plt.savefig(save_path)
plt.show()

