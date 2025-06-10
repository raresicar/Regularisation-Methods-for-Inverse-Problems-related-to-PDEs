from dolfin import *

# Unit Square Example with top-bottom accessible
class Accessible(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14

        # Accessible boundary Γ: y = 0 or y = 1 (top & bottom edges)
        # return on_boundary and (abs(x[1]) < tol or abs(x[1] - 1) < tol)

        # Accesible boundary Γ: y = 0 (bottom edge)
        return on_boundary and abs(x[1]) < tol

        # Accesible boundary Γ: y = 1 (top edge)
        # return on_boundary and abs(x[1] - 1) < tol

        # Accesible boundary Γ: y = 0 and |x - 0.5| < 0.1 (partial bottom edge)
        # return on_boundary and abs(x[1]) < tol and abs(x[0] - 0.5) < 0.1 + tol

        # Accesible boundary Γ: x = 0 (left edge)
        # return on_boundary and abs(x[0]) < tol


class Inaccessible(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not Accessible().inside(x, on_boundary)

