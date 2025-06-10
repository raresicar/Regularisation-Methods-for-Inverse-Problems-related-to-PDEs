from dolfin import *

def boundary(x, on_boundary):
    return on_boundary

def tikhonov_fem(ue, f, k, ind_omega, nele, alpha):
    k2 = Constant(k*k)
    mesh = UnitSquareMesh(nele, nele)

    V = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FiniteElement("CG", mesh.ufl_cell(), 1)
    VW = FunctionSpace(mesh, V*W)
    (u, z) = TrialFunction(VW)
    (v, w) = TestFunction(VW)
    boundary_condition = DirichletBC(VW.sub(1), Constant(0.0), boundary)

    a = (
            u * v * ind_omega * dx
        + alpha * dot(grad(u), grad(v)) * dx
        - alpha * dot(grad(z), grad(w)) * dx # regularisation for dual
        + dot(grad(v), grad(z)) * dx
        - k2 * v * z * dx
        + dot(grad(u), grad(w)) * dx
        - k2 * u * w * dx
    )

    L = (
            ue * v * ind_omega * dx
            + f * w * dx
    )

    sol = Function(VW)
    solve(a == L, sol, boundary_condition,
          solver_parameters={'linear_solver': 'mumps'})

    (u_h, z_h) = sol.split()

    return [u_h, z_h]

