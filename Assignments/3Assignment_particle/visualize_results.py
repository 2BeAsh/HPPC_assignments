import numpy as np
import matplotlib.pyplot as plt
import general_functions as gf  # My file with functions
from matplotlib.lines import Line2D

# Load data
data = np.genfromtxt("slurm_time_output.txt", delimiter=",", skip_header=1)
Ntasks = data[:, 0]
time_task = data[:, 1]
time_total = data[:, 2]

# Sort data
sort_idx = np.argsort(Ntasks)
Ntasks = Ntasks[sort_idx]
time_task = time_task[sort_idx]
time_total = time_total[sort_idx]

# Subtask A
# Plot time_task vs Ntasks
CPU_time_per_task = time_task * (Ntasks - 1)  # -1 because of Master

fig, ax = plt.subplots()
ax_color = "rebeccapurple"
ax.plot(Ntasks, time_task, ".-", c=ax_color, markersize=8, label="Single Task")
ax.plot(Ntasks, CPU_time_per_task, ls="dashdot", marker=".", markersize=8, c=ax_color, label="CPU time/task")
ax.set(xlabel="Number of cores")
ax.set_ylabel(r"Time ($\mu$s)", color=ax_color)
ax.tick_params(axis="y", labelcolor=ax_color)


# Twin axis
ax_twin = ax.twinx()
ax_twin_color = "red"
ax_twin.set_ylabel(r"Time (s)", color=ax_twin_color)
ax_twin.plot(Ntasks, time_total, ls="--", marker="x", markersize=8, c=ax_twin_color)
ax_twin.tick_params(axis="y", labelcolor=ax_twin_color)

legend_elements = [Line2D([], [], color=ax_color, ls="-", marker=".", label="Single task"),
                   Line2D([], [], color=ax_color, ls="dashdot", marker=".", label="CPU time/task"),
                   Line2D([], [], color=ax_twin_color, ls="--", marker="x", label="Total time")]
ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)


figname = "ex3a_time_per_task.png"
plt.savefig(figname)
plt.show()
plt.close()

