import numpy as np
import matplotlib.pyplot as plt

# Get data
data = np.genfromtxt("sir_output.txt", delimiter=",", skip_header=1)
S = data[:, 0]
I = data[:, 1]
R = data[:, 2]
dt = 0.01
time = np.arange(S.size) * dt

# Visualize
def plot_sir(vaccinated=False):
    fig, ax = plt.subplots()    
    fname = "sir_model.png"
    if vaccinated:
        ax.axvline(670, ls="--", color="grey", alpha=0.7, label="dS/dt=0")
        ax.axhline(510, ls="--", color="grey", alpha=0.7)
        fname = "sir_vaccinated.png"

    ax.plot(time, S, "-,", label="S")
    ax.plot(time, I, "-.", label="I")
    ax.plot(time, R, "-", label="R")
    ax.set(xlabel="Time", ylabel="Number of people", title=fr"SIR model")
    ax.set_xticks(np.linspace(time.min(), time.max(), 10))
    ax.legend()
    plt.savefig(fname, dpi=200)
    
plot_sir()
plot_sir(vaccinated=True)
