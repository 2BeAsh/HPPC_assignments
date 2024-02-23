import enum
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import csv
import sympy as sp
from IPython.display import display
from iminuit import Minuit
from scipy.optimize import curve_fit, minimize


# Customizing matplotlib settings
default_cycler = (cycler(color=["rebeccapurple", "red", "darkorange", "seagreen", "deepskyblue", "black"])
                  + cycler(linestyle = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))])
                  + cycler(marker=[".", "x", "v", "1", "p", ">"]))
plt.rc('axes', prop_cycle=default_cycler)
plt.style.use("presentation.mplstyle")


# -- DATA IMPORTING -- 
def load_data_with_subdatasets(fname, delimiter, skip_header, dtype):
    """Load a datafile which contains multiple datasets. Does not work if last element in datafile is whitespace."""

    # Read datafile and get length of datafile
    datafile = open(fname, "r")
    datareader = csv.reader(datafile)
    datafile_for_length = open(fname, "r") 
    row_count = sum(1 for _ in csv.reader(datafile_for_length))
    
    # Skip header rows
    for _ in range(skip_header):
        next(datareader, None)        
    
    # Empty lists to append subdatasets to 
    data_sets_combined = []
    
    # Loop
    line_number = 0
    while row_count > line_number:
        sub_data_set = []
            
        for row in datareader:
            line_number = datareader.line_num     
            # Skip empty rows
            if len(row) == 0:
                continue
            
            # Need to seperate row values and convert them to numbers
            # See if can string convert to float. If cannot, then we have encountered header and thus end of subdataset.
            # At last line need to append data and break. OBS could alternatively add a string to last line.
            row = row[0]  # "Convert" list to string
            try:
                row_value = row.split(delimiter)
                row_float = np.array(row_value, dtype=dtype)
                sub_data_set.append(row_float)
                
                if line_number == row_count: # Last line 
                    print("Not more data")
                    data_sets_combined.append(sub_data_set)
                    break
                                    
            except:  # header cannot convert to float
                data_sets_combined.append(sub_data_set)
                break
            
    return data_sets_combined


# -- BAYSIAN STATISTICS --
def find_normalization_const(f_expr, var, lower_lim, upper_lim, display_result=False):
    """Calculate normalization constant of a function by finding the definite integral using Sympy. 

    Args:
        f_expr (sympy expression): Function whose normalization constant is needed, as a Sympy expression
        var (sympy symbol): Integration variable.
        lower_lim (float): Lower integration limit
        upper_lim (float): Upper integration limit
        display_result (bool, optional): Display function and definite integral. Defaults to False.

    Returns:
        float: Normalization constant of f_expr
    """
    def_integral = sp.integrate(f_expr, (var, lower_lim, upper_lim)).doit()
    norm_const = 1 / def_integral
    if display_result:
        print("Function")
        display(f_expr)
        print("Definite Integral")
        display(sp.simplify(def_integral))
        print("Norm Constant")
        display(norm_const)
    return norm_const


def bayesian_posterior(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal):
    return f_likelihood(x, par_lh) * f_prior(x, par_prior) / f_marginal(x, par_marginal)


def visualiuze_bayesian_distributions(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal, xlabel="", ylabel="", title=""):
    prior = f_prior(x, par_prior)
    lh = f_likelihood(x, par_lh)
    posterior = bayesian_posterior(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal)
    
    fig, ax = plt.subplots()
    ax.plot(x, posterior, label="Posterior")
    ax.plot(x, prior, label="Prior")
    ax.plot(x, lh, label="Likelihood")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    plt.show()


def loglikelihood(par, x, pdf, sum_axis=0):
    """Calculate negative log likelihood of the pdf evaluated in x given parameters par. Assumes par is a tupple.

    Args:
        par (tupple): Parameter values for the pdf
        x (1darray): Points where the pdf is evaluated
        pdf (func): Probability density function
        sum_axis (int, optional): Which axis to sum over. Defaults to 0.

    Returns:
        1darray: llh values for each pararameter. Size par. 
    """
    return -np.sum(np.log(pdf(x, *par)), axis=sum_axis)


def minimize_llh(f_pdf, x, p0):
    res = minimize(loglikelihood, x0=p0, args=(x, f_pdf), method="L-BFGS-B")
    par = res.x
    err = np.sqrt(np.diag(res.hess_inv.todense()))
    return par, err
    

def parameter_bootstrapping(f_pdf, fit_par, x_bound, bootstrap_steps, MC_steps):
    par_vals = np.empty((fit_par.size, bootstrap_steps))
    
    for i in range(bootstrap_steps):
        # Get samples from pdf
        x_pdf = monte_carlo_sample_from_pdf(f_pdf, fit_par, x_bound, MC_steps)
        # Fit the sampels to get the fit par
        par, err = minimize_llh(f_pdf, x_pdf, p0=fit_par)
        par_vals[:, i] = par
        
    return par_vals





def llh_raster1d(f_pdf, x, par_vals, plot=False):
    """1d Raster LLH scan. 

    Args:
        f_pdf (func): The pdf for which the llh is calculated.
        x (1darray): x-data for the pdf
        par (1darray): Array of parameter values
        plot (bool, optional): Visualize the llh values and MLE. Defaults to False.

    Returns:
        (MLE_idx, MLE, par_MLE): The index of the MLE value, the MLE value, and the paramter evaluated at the MLE idx
    """
    llh_vals = np.empty(np.size(par_vals))
    llh_vals = loglikelihood(par_vals[None, :], x[:, None], f_pdf)
            
    MLE_idx = np.argmin(llh_vals)
    MLE = llh_vals[MLE_idx]
    par_MLE = par_vals[MLE_idx]
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(par_vals, llh_vals, ".", label="LLH")
        ax.plot(par_MLE, MLE, "x", label="Max LLH")
        ax.set(xlabel="Parameter value", ylabel="LLH")
        ax.legend()
        plt.show()
    
    return MLE_idx, MLE, par_MLE, llh_vals




# def llh_raster1d(llh_func, x, par_range, other_par=(), N_raster_vals=200, plot=False):
#     """Currently suboptimal/Non-working"""
#     par_vals = np.linspace(*par_range, N_raster_vals)
#     llh_vals = llh_func(x, par_vals, *other_par)
#     idx_max = np.argmax(llh_vals)
#     MLE = llh_vals[idx_max]  # Maximum Likelihood Estimate
    
#     if plot:
#         fig, ax = plt.subplots()
#         ax.plot(par_vals, llh_vals, label="LLH")
#         ax.axvline(par_vals[idx_max], ls="dashed", alpha=0.9)
#         return MLE, fig, ax
#     return MLE


def llh_2d_confidence_region(f_pdf, x, par1, par2, plot=False):
    # 2d raster    
    par1_mesh, par2_mesh = np.meshgrid(par1, par2)
    pp1 = par1_mesh[None, :, :]
    pp2 = par2_mesh[None, :, :]
    xx = x[:, None, None]

    llh_vals = loglikelihood((pp1, pp2), xx, f_pdf)
    
    # MLE
    MLE_idx = np.unravel_index(np.argmin(llh_vals, axis=None), llh_vals.shape)
    MLE = llh_vals[MLE_idx]
        
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(llh_vals, extent=(par1.min(), par1.max(), par2.min(), par2.max()), origin="upper", vmin=MLE, vmax=1060)
        # Confidence region
        ax.contour(llh_vals, levels=[MLE+1.15, MLE+3.09, MLE+5.92], extent=(par1.min(), par1.max(), par2.min(), par2.max()), origin="upper")
        ax.set(xlabel="alpha", ylabel="beta")
        plt.colorbar(im)
        plt.show()


# -- FITTING --

def fit_unbinned_data(fit_pdf, x, y, p0, plot=False):
    par, cov = curve_fit(fit_pdf, x, y, p0)
    err = np.sqrt(np.diag(cov))
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, fmt=".", label="data")        
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = fit_pdf(x_fit, *par)
        ax.plot(x_fit, y_fit, label="Fit")
        ax.legend()
        plt.show()
    
    return par, err
        

# --MONTE CARLO --
def monte_carlo_sample_from_pdf(f_pdf, par, x_bound, MC_steps):
    x_pdf = []
    rng = np.random.default_rng()
    while len(x_pdf) < MC_steps:
        x = rng.uniform(low=x_bound[0], high=x_bound[1])
        u = rng.uniform(low=0, high=1)
        if f_pdf(x, *par) > u:
            x_pdf.append(x)
    return np.array(x_pdf)


# -- GENERAL -- 
def find_nearest(x, val):
    """Find the value in the array x that is closest to the given value val.

    Args:
        x (1darray): Data values
        val (float): Value we want an index in x close to.

    Returns:
        int: Index of the value in x closest to val
    """
    idx = np.abs(x - val).argmin()
    return idx
