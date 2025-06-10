from dolfin import *
import numpy as np

def error_H1(g0, u, mesh, degree):
    # 1) high-order space for the difference
    W = FunctionSpace(mesh, 'P', degree + 3)

    # 2) interpolate both exact and numeric solutions into W
    ue_W = interpolate(g0, W)
    ũ_W = interpolate(u, W)

    # 3) form the difference function
    e_W = Function(W)
    # get the underlying NumPy arrays
    ue_loc = ue_W.vector().get_local()
    u_loc = ũ_W.vector().get_local()
    # set the local values of e_W to the pointwise difference
    e_W.vector().set_local(ue_loc - u_loc)
    # push those values into the PETSc vector
    e_W.vector().apply("insert")

    # 4) assemble the H^1-norm
    a = (inner(e_W, e_W) + dot(grad(e_W), grad(e_W))) * dx
    return sqrt(abs(assemble(a)))

def error_H01(g0, u, mesh, degree):
    W = FunctionSpace(mesh, 'P', degree + 3)
    e_W = project(g0 - u, W)
    a = inner(grad(e_W), grad(e_W)) * dx
    return sqrt(abs(assemble(a)))

def errors(g0, u):
    fspace = u.function_space()
    mesh = fspace.mesh()
    degree = fspace.ufl_element().degree()

    err_L2 = sqrt(assemble(dot(g0-u, g0-u) * dx)) / sqrt(assemble(dot(g0, g0) *dx(mesh)))
    err_H1 = error_H1(g0, u, mesh, degree) / error_H1(g0, Constant(0), mesh, degree)
    err_H01 = error_H01(g0, u, mesh, degree) / error_H01(g0, Constant(0), mesh, degree)

    return [err_L2, err_H1, err_H01]
