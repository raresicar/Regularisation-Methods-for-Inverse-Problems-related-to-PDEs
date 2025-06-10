import matplotlib.pyplot as plt
from errors import *
from methods import *
from plots import *
from data import load

# ue, g0, g1, f = load("1")              # -> burman and pinns polynomial data
# ue, g0, g1, f = load("2")              # -> fenics documentation-book data
ue, g0, g1, f = load("3", n=2.0)       # -> Hadamard data
# ue, g0, g1, f = load("4")              # -> poly and oscillations sum data

k = 0.0 # for Helmholtz equation
nele = 128 # mesh is nele x nele
alpha = 1e-7 # Tikhonov regularisation parameter
gamma = 10.0 # penalty weight in Nitsche method
weak = True # for choosing the method
noise = False # for noisy experiment

if weak:
    u_h, _ = tikhonovfem(g0, g1, f, k, nele, alpha, noise, gamma=gamma)
else:
    u_h, _ = tikhonovfem(g0, g1, f, k, nele, alpha, noise)

# errors
e_L2, e_H1, e_H01 = errors(ue, u_h)
print("Relative L2 norm error: ", e_L2)
print("Relative H1 norm error: ", e_H1)
print("Relative H01 seminorm error: ", e_H01)

# error distribution plot
error_distribution_plot(u_h, ue)

# solution plot
solution_plot(u_h)

# error evolution loglog-plot
mesh_sizes = [8, 16, 32, 64, 128, 256]
errs_H1    = []
errs_L2    = []
hs         = []

for N in mesh_sizes:
    print(f"Solving on {N} × {N} mesh …")
    if weak:
        u_h, _ = tikhonovfem(g0, g1, f, k, N, alpha, noise, gamma=gamma)
    else:
        u_h, _ = tikhonovfem(g0, g1, f, k, N, alpha, noise)
    err_L2, err_H1, _ = errors(ue, u_h)
    errs_H1.append(err_H1)
    errs_L2.append(err_L2)
    hs.append(1.0 / N)

error_plot(hs, errs_H1)