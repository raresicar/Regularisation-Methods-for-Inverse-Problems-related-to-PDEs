from dolfin import *
from boundaries import *
import numpy as np


def tikhonovfem(g0, g1, f, k, nele, alpha, noise, gamma=None, rel_noise=0.01, random_seed=None):
    k2 = Constant(k * k)
    mesh = UnitSquareMesh(nele, nele)

    # mesh for Hadamard's example (0, pi) x (0, 1)
    # nx, ny = 2 * nele, nele
    # p0 = Point(0.0, 0.0)
    # p1 = Point(np.pi, 1.0)
    # mesh = RectangleMesh(p0, p1, nx, ny, "left/right")

    n = FacetNormal(mesh)  # used for weakly imposition of boundary condition on u
    h = CellDiameter(mesh)  # used for penalty term in nitsche method

    facet_markers = MeshFunction("size_t",
                                 mesh,
                                 mesh.topology().dim() - 1,
                                 0)
    accessible = Accessible()
    inaccesible = Inaccessible()
    accessible.mark(facet_markers, 1)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    V = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FiniteElement("CG", mesh.ufl_cell(), 1)
    VW = FunctionSpace(mesh, V * W)
    (u, z) = TrialFunctions(VW)
    (v, w) = TestFunctions(VW)
    boundary_condition = DirichletBC(VW.sub(1), Constant(0.0), inaccesible)

    V_scalar = FunctionSpace(mesh, "CG", 5)  # ≥ degree of expressions
    g0_n = noisy_data(g0, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 0)
    g1_n = noisy_data(g1, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 1)
    f_n = noisy_data(f, V_scalar, dx, rel_noise, seed=None if random_seed is None else random_seed + 2)

    if gamma is None:
        a = (
                u * v * ds(1)
                + alpha * dot(grad(u), grad(v)) * dx
                -alpha * dot(grad(z), grad(w)) * dx  # regularisation for dual
                + dot(grad(v), grad(z)) * dx
                + dot(grad(u), grad(w)) * dx
                - k2 * v * z * dx
                - k2 * u * w * dx
        )
        L = (
                g0 * v * ds(1)
                + f * w * dx
                + g1 * w * ds(1)
        )
        L_n = (
                g0_n * v * ds(1)
                + f_n * w * dx
                + g1_n * w * ds(1)
        )
    else:
        a = (
                alpha * dot(grad(u), grad(v)) * dx
                - alpha * dot(grad(z), grad(w)) * dx  # regularisation for dual
                + dot(grad(v), grad(z)) * dx
                + dot(grad(u), grad(w)) * dx
                - k2 * v * z * dx
                - k2 * u * w * dx
        )
        L = (
                f * w * dx
                + g1 * w * ds(1)
        )
        # add Nitsche terms (no penalty)
        a += u * dot(grad(v), n) * ds(1) # +⟨u, ∇v·n⟩

        L += g0 * dot(grad(v), n) * ds(1)  # ⟨g, ∇v·n⟩
        # penalty term for coercivity
        a += gamma * (1.0 / h) * u * v * ds(1)
        L += gamma * (1.0 / h) * g0 * v * ds(1)
        L_n = (
                f_n * w * dx
                + g1_n * w * ds(1)
                + g0_n * dot(grad(v), n) * ds(1)
                + gamma * (1.0 / h) * g0_n * v * ds(1)
        )

    if noise:
        sol = Function(VW)
        solve(a == L_n, sol, boundary_condition,
              solver_parameters={"linear_solver": "mumps"})
        u_h, z_h = sol.split()
    else:
        sol = Function(VW)
        solve(a == L, sol, boundary_condition,
              solver_parameters={"linear_solver": "mumps"})
        u_h, z_h = sol.split()

    return [u_h, z_h]


def noisy_data(expr, V, measure, rel_std=0.01, tag=None, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    f = interpolate(expr, V)
    if tag is None:
        L2sq = assemble(f**2 * measure)
    else:
        L2sq = assemble(f**2 * measure(tag))
    sigma = rel_std * np.sqrt(L2sq)
    noise = Function(V)
    noise.vector()[:] = sigma * rng.standard_normal(V.dim())
    f_noisy = Function(V)
    f_noisy.vector()[:] = f.vector() + noise.vector()
    return f_noisy


# def tikhonov_fem(g0, g1, f, k, nele, alpha, noise, rel_noise=0.01, random_seed=None):
#     k2 = Constant(k * k)
#     mesh = UnitSquareMesh(nele, nele)
#
#     facet_markers = MeshFunction("size_t",
#                                  mesh,
#                                  mesh.topology().dim() - 1,
#                                  0)
#     accessible = Accessible()
#     inaccesible = Inaccessible()
#     accessible.mark(facet_markers, 1)
#     ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)
#
#     V = FiniteElement("CG", mesh.ufl_cell(), 1)
#     W = FiniteElement("CG", mesh.ufl_cell(), 1)
#     VW = FunctionSpace(mesh, V * W)
#     (u, z) = TrialFunctions(VW)
#     (v, w) = TestFunctions(VW)
#     boundary_condition = DirichletBC(VW.sub(1), Constant(0.0), inaccesible)
#
#     V_scalar = FunctionSpace(mesh, "CG", 5)   # ≥ degree of expressions
#     g0_n = noisy_data(g0, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 0)
#     g1_n = noisy_data(g1, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 1)
#     f_n = noisy_data(f, V_scalar, dx, rel_noise, seed=None if random_seed is None else random_seed + 2)
#
#     a = (
#             u * v * ds(1)
#             + alpha * dot(grad(u), grad(v)) * dx
#             + dot(grad(v), grad(z)) * dx
#             + dot(grad(u), grad(w)) * dx
#             - k2 * v * z * dx
#             - k2 * u * w * dx
#     )
#
#     L = (
#             g0 * v * ds(1)
#             + f * w * dx
#             + g1 * w * ds(1)
#     )
#
#     L_n = (
#             g0_n * v * ds(1)
#             + f_n * w * dx
#             + g1_n * w * ds(1)
#     )
#
#     if noise:
#         sol = Function(VW)
#         solve(a == L_n, sol, boundary_condition,
#               solver_parameters={"linear_solver": "mumps"})
#         u_h, z_h = sol.split()
#     else:
#         sol = Function(VW)
#         solve(a == L, sol, boundary_condition,
#               solver_parameters={"linear_solver": "mumps"})
#         u_h, z_h = sol.split()
#
#     return [u_h, z_h]
#
#
# def weak_tikhonov_fem(g0, g1, f, k, nele, alpha, gamma, noise, rel_noise=0.01, random_seed=None):
#     k2 = Constant(k * k)
#     mesh = UnitSquareMesh(nele, nele)
#
#     n = FacetNormal(mesh) # used for weakly imposition of boundary condition on u
#     h = CellDiameter(mesh) # used for penalty term in nitsche method
#
#     facet_markers = MeshFunction("size_t",
#                                  mesh,
#                                  mesh.topology().dim() - 1,
#                                  0)
#     accessible = Accessible()
#     inaccesible = Inaccessible()
#     accessible.mark(facet_markers, 1)
#     ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)
#
#     V = FiniteElement("CG", mesh.ufl_cell(), 1)
#     W = FiniteElement("CG", mesh.ufl_cell(), 1)
#     VW = FunctionSpace(mesh, V * W)
#     (u, z) = TrialFunctions(VW)
#     (v, w) = TestFunctions(VW)
#     boundary_condition = DirichletBC(VW.sub(1), Constant(0.0), inaccesible)
#
#     V_scalar = FunctionSpace(mesh, "CG", 5)  # ≥ degree of expressions
#     g0_n = noisy_data(g0, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 0)
#     g1_n = noisy_data(g1, V_scalar, ds, rel_noise, tag=1, seed=None if random_seed is None else random_seed + 1)
#     f_n = noisy_data(f, V_scalar, dx, rel_noise, seed=None if random_seed is None else random_seed + 2)
#
#     a = (
#             alpha * dot(grad(u), grad(v)) * dx
#             + alpha * dot(grad(z), grad(w)) * dx # regularisation for dual
#             + dot(grad(v), grad(z)) * dx
#             + dot(grad(u), grad(w)) * dx
#             - k2 * v * z * dx
#             - k2 * u * w * dx
#     )
#
#     L = (
#             f * w * dx
#             + g1 * w * ds(1)
#     )
#
#     # add Nitsche terms (no penalty)
#     a += (- dot(grad(u), n) * v  # –⟨∇u·n, v⟩
#           + u * dot(grad(v), n)  # +⟨u, ∇v·n⟩
#           ) * ds(1)
#
#     L += g0 * dot(grad(v), n) * ds(1)  # ⟨g, ∇v·n⟩
#
#     # penalty term for coercivity
#     a += gamma * (1.0 / h)  * u * v * ds(1)
#     L += gamma * (1.0 / h)  * g0 * v * ds(1)
#
#     L_n = (
#             f_n * w * dx
#             + g1_n * w * ds(1)
#             + g0_n * dot(grad(v), n) * ds(1)
#             + gamma * (1.0 / h)  * g0_n * v * ds(1)
#     )
#
#     if noise:
#         sol = Function(VW)
#         solve(a == L_n, sol, boundary_condition,
#               solver_parameters={"linear_solver": "mumps"})
#         u_h, z_h = sol.split()
#     else:
#         sol = Function(VW)
#         solve(a == L, sol, boundary_condition,
#               solver_parameters={"linear_solver": "mumps"})
#         u_h, z_h = sol.split()
#
#     return [u_h, z_h]
