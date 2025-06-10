from domains import *
from methods import *
from errors import *
import matplotlib.pyplot as plt
import numpy as np

ind_omega = ind_omega_nonconv(degree=0)
ind_omega.set_values(1, 0)
ind_B = ind_B_nonconv(degree=0)
ind_B.set_values(1, 0)

# ue = Expression("1 + x[0] * x[0] + 2 * x[1] * x[1]",degree=5)
# f = Expression("-6", degree=5)

# ue = Expression("30 * x[0] * x[1] * (1 - x[0]) * (1- x[1])", degree=5)
# f = Expression("60 * (x[0] - x[0] * x[0] + x[1] - x[1] * x[1])", degree=5)

# Hadamard-type example
# n = 2.0
# ue = Expression("(0.5 / (n * n)) * sin(n * x[0]) * "
#                     "(exp(n * x[1]) - exp(-n * x[1]))",
#                     degree=5, n=n)
# f = Expression("0.0", degree=5)

# Gaussian Bump for Hemholtz
sigmax = 0.01
sigmay = 0.1
k      = 10.0
f = Expression(
    "-exp( -pow(x[0]-0.5,2)/(2*sigmax)"
          " -pow(x[1]-1.0,2)/(2*sigmay) )"
    " * ( pow(x[0]-0.5,2)/(sigmax*sigmax) - 1.0/sigmax"
    "   + pow(x[1]-1.0,2)/(sigmay*sigmay) - 1.0/sigmay"
    "   + k*k )",
    degree=5, sigmax=sigmax, sigmay=sigmay, k=k)
ue = Expression(
    "exp( -pow(x[0]-0.5,2)/(2*sigmax)"
         " -pow(x[1]-1.0,2)/(2*sigmay) )",
    degree=5, sigmax=sigmax, sigmay=sigmay)

# ue = Expression("1 + exp(x[0]) * sin(x[0]) + x[1] * x[1]", degree=5)
# f = Expression("-2 * (exp(x[0]) * cos(x[0]) + 1)", degree=5)

k = 10.0
nele = 128
alpha = 1e-6

u_h, z_h = tikhonov_fem(ue, f, k, ind_omega, nele, alpha)

e_L2, e_H1, e_H01 = errors(ue, u_h, ind_B)
print("Relative L2 norm error: ", e_L2)
print("Relative H1 norm error: ", e_H1)
print("Relative H01 seminorm error: ", e_H01)

# error distribution plot
V = u_h.function_space()
mesh = V.mesh()
degree_u = u_h.function_space().ufl_element().degree()
W = FunctionSpace(mesh, "CG", degree_u + 3)
e_h = project(u_h - ue, W)
plt.figure(figsize=(4,3))
p = plot(e_h, title="error e_h = u_h - ue", mode="color", cmap="coolwarm")
plt.colorbar(p)
plt.tight_layout()
plt.show()

c = plot(u_h, mode="color")
plt.colorbar(c)
plt.show()
