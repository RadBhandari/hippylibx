import dolfinx as dlx
import ufl
from petsc4py import PETSc


def projection(v, target_func, bcs=None):
    """
    Project expression ``v`` onto the function space of ``target_func``.

    The result is written in-place into ``target_func``.
    """
    if bcs is None:
        bcs = []

    V = target_func.function_space
    dx = ufl.dx(domain=V.mesh)

    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)

    a = dlx.fem.form(ufl.inner(Pv, w) * dx)
    L = dlx.fem.form(ufl.inner(v, w) * dx)

    A = None
    b = None
    solver = None

    try:
        A = dlx.fem.petsc.assemble_matrix(a, bcs=bcs)
        A.assemble()

        b = dlx.fem.petsc.assemble_vector(L)

        dlx.fem.petsc.apply_lifting(
            b,
            [a],
            bcs=[bcs],
        )

        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

        dlx.fem.petsc.set_bc(b, bcs)

        solver = PETSc.KSP().create(V.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setFromOptions()

        solver.solve(
            b,
            target_func.x.petsc_vec,
        )

        target_func.x.scatter_forward()

    finally:
        if solver is not None:
            solver.destroy()

        if b is not None:
            b.destroy()

        if A is not None:
            A.destroy()