from dolfin import *

def error_H1(ue, u, ind_B, mesh, degree):
    W = FunctionSpace(mesh, 'P', degree + 3)
    ue_W = interpolate(ue, W)
    ũ_W = interpolate(u, W)
    e_W = Function(W)

    ue_loc = ue_W.vector().get_local()
    u_loc = ũ_W.vector().get_local()

    e_W.vector().set_local(ue_loc - u_loc)
    e_W.vector().apply("insert")
    a = ind_B * (inner(e_W, e_W) + dot(grad(e_W), grad(e_W))) * dx
    return sqrt(abs(assemble(a)))

def error_H01(ue, u, ind_B, mesh, degree):
    W = FunctionSpace(mesh, 'P', degree + 3)
    e_W = project(ue - u, W)
    a = ind_B * inner(grad(e_W), grad(e_W)) * dx
    return sqrt(abs(assemble(a)))

def errors(ue, u, ind_B):
    fspace = u.function_space()
    mesh = fspace.mesh()
    degree = fspace.ufl_element().degree()

    err_B_L2 = sqrt(assemble(ind_B * dot(ue-u, ue-u) * dx)) / sqrt(assemble(ind_B * dot(ue, ue) *dx(mesh)))
    err_B_H1 = error_H1(ue, u, ind_B, mesh, degree) / error_H1(ue, Constant(0), ind_B, mesh, degree)
    err_B_H01 = error_H01(ue, u, ind_B, mesh, degree) / error_H01(ue, Constant(0), ind_B, mesh, degree)

    return [err_B_L2, err_B_H1, err_B_H01]