import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from errors import *

def solution_plot(u_h):
    # plot of solution
    plt.figure()
    c = plot(u_h, title="Recovered state u_h", mode="color")
    plt.colorbar(c)
    plt.show()

def error_distribution_plot(u_h, g0):
    # error distribution plot
    V = u_h.function_space()
    mesh = V.mesh()
    degree_u = u_h.function_space().ufl_element().degree()
    W = FunctionSpace(mesh, "CG", degree_u + 3)
    e_h = project(u_h - g0, W)
    plt.figure(figsize=(4, 3))
    p = plot(e_h, title="error e_h = u_h - ue", mode="color", cmap="coolwarm")
    plt.colorbar(p)
    plt.tight_layout()
    plt.show()

def error_plot(hs, errs, label=r"$\|u_h - ue\|_{H^1}$"):
    # error evolution log-log plot
    plt.figure()
    plt.loglog(hs, errs, "-o", label=label,markerfacecolor='none')

    # optional slope guide — least-squares fit  log(err) = p*log(h) + c
    logh = np.log(hs)
    loge = np.log(errs)
    p_fit, c_fit = np.polyfit(logh, loge, 1)  # slope p_fit, intercept c_fit
    e_fit = np.exp(c_fit)
    guide = e_fit * hs ** p_fit
    plt.loglog(hs, guide, "--", color="gray", lw=1.2,
               label=rf"$\mathcal{{O}}\!\left(h^{{{p_fit:.2f}}}\right)$")

    # plt.gca().invert_xaxis()  # finest mesh on the right
    plt.xlabel(r"log mesh size  $h = 1/N$")
    plt.ylabel(r"log relative  $H^1$  error")
    plt.title("Convergence of the Cauchy solver")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def parameter_plot(alphas, errs_L2, errs_H1):
    # for heuristic choice of regularisation parameter
    plt.figure()
    plt.loglog(alphas, errs_L2, "-o", label="L2", markerfacecolor='none')
    plt.loglog(alphas, errs_H1, "-o", label="H1", markerfacecolor='none')
    plt.gca().invert_xaxis()
    plt.xlabel(r"log alpha_T")
    plt.ylabel(r"log relative error")
    plt.title("Heuristic choice of parameter")
    plt.grid(True)
    plt.tight_layout()
    plt.show()