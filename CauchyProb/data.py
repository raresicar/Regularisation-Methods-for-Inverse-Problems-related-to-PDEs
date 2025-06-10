from dolfin import Expression
import numpy as np

# ----------------------------------------------------------------------
# polynomial example (PINNs + Burman paper) for top or bottom edge
# ----------------------------------------------------------------------
def poly_case():
    ue = Expression("30 * x[0] * x[1] * (1 - x[0]) * (1 - x[1])",
                    degree=5)
    g1 = Expression("-30 * x[0] * (1 - x[0])",
                    degree=5)
    f  = Expression("60 * (x[0] - x[0] * x[0] + "
                    "x[1] - x[1] * x[1])",
                    degree=5)
    g0 = Expression("0.0", degree=5)
    return ue, g0, g1, f

# ----------------------------------------------------------------------
# FEniCS documentation book example for top or bottom edge y = 0, 1
# ----------------------------------------------------------------------
def doc_case():
    ue = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]",
                    degree=5)
    g1 = Expression("4 * x[1]",
                    degree=5)
    f  = Expression("-6",
                    degree=5)
    g0 = ue
    return ue, g0, g1, f

# ----------------------------------------------------------------------
# Hadamard family   u = (1/n) sin(nx) sinh(ny) for bottom edge y = 0
# switch mesh to Rectangle and add data parameter for better results
# ----------------------------------------------------------------------
def hadamard_case(n=2.0):
    ue = Expression("(0.5 / (n * n)) * sin(n * x[0]) * "
                    "(exp(n * x[1]) - exp(-n * x[1]))",
                    degree=5, n=n)
    g1 = Expression("-(1 / n) * sin(n * x[0])",
                    degree=5, n=n)
    f  = Expression("0.0", degree=5)
    g0 = Expression("0.0", degree=5)
    return ue, g0, g1, f

# ----------------------------------------------------------------------
# poly in y but oscillating in x : for left edge x = 0
# switch k to 10.0 -> Helmholtz
# ----------------------------------------------------------------------
def combination_case():
    ue = Expression("1 + exp(x[0]) * sin(x[0]) + x[1] * x[1]", degree=5)
    g1 = Expression("-1", degree=5)
    f = Expression("-2 * (exp(x[0]) * cos(x[0]) + 1) - 100 - 100 * "
                   "exp(x[0]) * sin(x[0]) - 100 * x[1] * x[1]", degree=5)
    g0 = Expression("1 + x[1] * x[1]", degree=5)
    return ue, g0, g1, f

# ----------------------------------------------------------------------
# Registry for easy lookup
# ----------------------------------------------------------------------
_cases = {
    "1":     poly_case,
    "2":      doc_case,
    "3": hadamard_case,
    "4": combination_case,
}

def load(name, **kwargs):
    """
    Convenience loader:
        ue, g0, g1, f = load("hadamard", n=5.0)
    """
    if name not in _cases:
        raise KeyError(f"Unknown case '{name}'. "
                       f"Available: {list(_cases)}")
    return _cases[name](**kwargs)