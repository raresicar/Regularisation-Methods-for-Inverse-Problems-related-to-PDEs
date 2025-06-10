from errors import *
from methods import *
from plots import *
from data import load

# ue, g0, g1, f = load("1")
ue, g0, g1, f = load("2")
# ue, g0, g1, f = load("3", n=2.0)
# ue, g0, g1, f = load("4")

k = 0.0
nele = 128
gamma = 10.0
weak = True
noise = False

errs_H1    = []
errs_L2    = []
alphas     = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for p in alphas:
    print(f"Solving with parameter {p}")
    if weak:
        u_h, _ = tikhonovfem(g0, g1, f, k, nele, p, noise, gamma=gamma)
    else:
        u_h, _ = tikhonovfem(g0, g1, f, k, nele, p, noise)
    err_L2, err_H1, _ = errors(ue, u_h)
    errs_H1.append(err_H1)
    errs_L2.append(err_L2)

parameter_plot(alphas, errs_L2, errs_H1)