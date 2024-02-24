import numpy as np
import matplotlib.pyplot as plt
import general_functions as gf  # My file with functions
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


# Load data
data = np.genfromtxt("slurm_time_output.txt", delimiter=",", skip_header=1)
Ntasks = data[:, 0]
time_task = data[:, 1]
time_total = data[:, 2]

# Multiple datapoints for ntasks = 2 (i.e 1 worker). 
# Combine into one and get uncertainty
one_worker_idx = np.where(Ntasks==2)[0]
not_one_worker_idx = np.where(Ntasks!=2)[0]
time_task_one_worker = time_task[one_worker_idx]
time_total_one_worker = time_total[one_worker_idx]

mean_task = np.mean(time_task_one_worker)
std_task = np.std(time_task_one_worker) / np.sqrt(time_total_one_worker.size)
mean_total = np.mean(time_total_one_worker)
std_total = np.std(time_total_one_worker) / np.sqrt(time_total_one_worker.size)

# Remove the one worker points and insert the mean instead
Ntasks = np.append(Ntasks[not_one_worker_idx], 2) - 1  # -1 to remove master
time_task = np.append(time_task[not_one_worker_idx], mean_task)
time_total = np.append(time_total[not_one_worker_idx], mean_total)

# Sort data
sort_idx = np.argsort(Ntasks)
Ntasks = Ntasks[sort_idx]
time_task = time_task[sort_idx]
time_total = time_total[sort_idx]

CPU_time_per_task = time_task * Ntasks 


# -- SUBTASK A --
# Plot time_task vs Ntasks
def subtask_A():
    fig, (ax, ax1) = plt.subplots(nrows=2)
    ax_color = "rebeccapurple"
    ax.plot(Ntasks, time_task, ".-", c=ax_color, markersize=8, label="Single Task")
    ax.plot(Ntasks, CPU_time_per_task, ls="dashdot", marker=".", markersize=8, c=ax_color, label="CPU time/task")
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
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=3)

    # Ax 1 - relative 
    N_worker = Ntasks
    time_task_rel = time_task[0] / time_task  # Ignore the first point as it 
    ax1.plot(N_worker, N_worker, c="deepskyblue", label="Ideal")  
    ax1.plot(N_worker, time_task_rel, ls="--", marker=".", c="darkorange", label="Data")  # -1 to remove master
    ax1.set(xlabel="Number of cores", ylabel="Speedup")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2)

    figname = "ex3a_time_per_task.png"
    plt.savefig(figname)
    plt.show()
    plt.close()


# -- SUBTASK B --
def subtask_B():

    # Vurder hvilke dele af koden som er parallel og serial. Brug derefter profiler til at sige hvor stor en del de fylder af tiden?
    # Fit vores data for at finde serial og parallel fraction

    fig, ax = plt.subplots()
    def amdahl_law_theoretical(N_processor, parallel_fraction):
        """Amdahl's law given the number of processors and the fraction of the program that is parallel."""
        S_latency = 1 / ((1 - parallel_fraction) + parallel_fraction / N_processor)
        return S_latency

    N_worker = Ntasks
    time_task_rel = time_task[0] / time_task 
    std_task_rel = std_task * time_task_rel

    par, cov = curve_fit(amdahl_law_theoretical, N_worker, time_task_rel, sigma=std_task_rel, p0=(0.5, ))
    err = np.sqrt(np.diag(cov))
    x_fit = np.linspace(N_worker.min(), N_worker.max(), 200)
    y_fit = amdahl_law_theoretical(x_fit, *par)
    ax.plot(N_worker, N_worker, c="deepskyblue", label="Ideal")  
    ax.errorbar(N_worker, time_task_rel, yerr=std_task_rel, ls="--", fmt=".", c="darkorange", label="Data") 
    ax.plot(x_fit, y_fit, "-", label="Fit")
    ax.set(xlabel="Number of cores", ylabel="Speedup")
    ax.axhline(0, c="black", lw=1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=3)
    str_result = r"$Par. frac = $" + f"{par[0]:.3f}" + r"$\pm$" + f"{err[0]:.3f}"
    ax.text(20, 100, s=str_result)

    figname = "ex3b_amdahl.png"
    plt.savefig(figname)
    plt.show()
    plt.close()

    
    
if __name__ == "__main__":
    subtask_A()
    subtask_B()