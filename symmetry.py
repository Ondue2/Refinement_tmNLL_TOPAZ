import numpy as np
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
import sympy as sp

def get_operators_sp(name):

    group = MagneticSpaceGroup(name)

    operators = []

    for op in group.symmetry_ops:

        R = sp.Matrix(op.rotation_matrix).applyfunc(
            lambda x: sp.Rational(str(x))
        )

        t = sp.Matrix(op.translation_vector).applyfunc(
            lambda x: sp.Rational(str(x)).limit_denominator(48)
        )

        epsilon = op.time_reversal

        operators.append((R, t, epsilon))

    return operators

def get_operators_np(name):

    group = MagneticSpaceGroup(name)

    operators = []

    for op in group.symmetry_ops:

        R = op.rotation_matrix
        t = op.translation_vector
        epsilon = op.time_reversal

        operators.append((R, t, epsilon))

    return operators

def get_op_functions_sp(name, magnetic=False):

    ops = get_operators_sp(name)
    
    if magnetic == False:

        nonmag_op_functions = [lambda x, op=op: op[0]*x + op[1] for op in ops]

        return nonmag_op_functions

    else:

        mag_op_functions = [lambda x, op=op: (op[0].det()*op[0])*x*op[2] for op in ops]

        return mag_op_functions


def get_op_functions_np(name, magnetic=False):

    ops = get_operators_np(name)

    if magnetic == False:

        nonmag_op_functions = [lambda x, op=op: op[0]@x + op[1] for op in ops]

        return nonmag_op_functions

    else:

        mag_op_functions = [lambda x, op=op:np.linalg.det(op[0])*op[0]@x*op[2] for op in ops]

        return mag_op_functions

def get_U_transform_matrix_sp(R):

    R = sp.Matrix(R)

    indices = [
        (0, 0), (1, 1), (2, 2),
        (1, 2), (0, 2), (0, 1)
    ]

    basis = []

    for i, j in indices:

        E = sp.zeros(3, 3)
        E[i, j] = 1
        E[j, i] = 1

        basis.append(E)

    T = sp.zeros(6, 6)

    for n, E in enumerate(basis):

        E_new = R * E * R.T

        for m, (i, j) in enumerate(indices):
            T[m, n] = E_new[i, j]

    return T

def get_U_transform_matrix_np(R):

    R = np.asarray(R)

    indices = [
        (0, 0), (1, 1), (2, 2),
        (1, 2), (0, 2), (0, 1)
    ]

    basis = []

    for i, j in indices:

        E = np.zeros((3, 3))
        E[i, j] = 1
        E[j, i] = 1

        basis.append(E)

    T = np.zeros((6, 6))

    for n, E in enumerate(basis):

        E_new = R @ E @ R.T

        for m, (i, j) in enumerate(indices):
            T[m, n] = E_new[i, j]

    return T

def get_minimal_operators(position, MSG, lattice_constants, epsilon = 1e-3, magnetic = False):

    r0 = np.mod(np.array(position) + epsilon, 1) - epsilon
    generated = [r0]
    Identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    a = lattice_constants[0]
    b = lattice_constants[1]
    c = lattice_constants[2]
    
    S = np.diag([a, b, c])
    invS = np.diag([1/a, 1/b, 1/c])

    if magnetic == False:

        nonmag_minimal_operators = []

        nonmag_op_functions = get_op_functions_np(MSG, magnetic=magnetic)

        for nonmag_op_function in nonmag_op_functions:

            current_r = np.mod(nonmag_op_function(r0) + epsilon, 1) - epsilon

            exist = False

            for r in generated:

                if np.linalg.norm(current_r - r) < epsilon:

                    exist = True
                    break

            if exist == False:

                generated.append(current_r)
                nonmag_minimal_operators.append(nonmag_op_function)

        nonmag_minimal_operators.insert(0, lambda x: Identity@x)

        return nonmag_minimal_operators

    if magnetic == True:

        mag_minimal_operators = []

        nonmag_op_functions = get_op_functions_np(MSG, magnetic=False)
        mag_op_functions = get_op_functions_np(MSG, magnetic=True)

        for nonmag_op_function, mag_op_function in zip(nonmag_op_functions, mag_op_functions):

            current_r = np.mod(nonmag_op_function(r0) + epsilon, 1) - epsilon

            exist = False
    
            for r in generated:
    
                if np.linalg.norm(current_r - r) < epsilon:
    
                    exist = True
                    break
    
            if exist == False:
    
                generated.append(current_r)
                R = mag_op_function(Identity)
                M_axis = S @ R @ invS
                mag_minimal_operators.append(lambda x, M=M_axis: M @ x)

        M_axis = S @ invS
        mag_minimal_operators.insert(0, lambda x, M=M_axis: M @ x)

        return mag_minimal_operators

def get_minimal_matrices_U(position, MSG, lattice_constants, epsilon = 1e-3):

    r0 = np.mod(np.array(position) + epsilon, 1) - epsilon
    generated = [r0]

    a = lattice_constants[0]
    b = lattice_constants[1]
    c = lattice_constants[2]
    
    S = np.diag([a, b, c])
    invS = np.diag([1/a, 1/b, 1/c])

    U_matrices = []

    ops = get_operators_np(MSG)

    for op in ops:

        current_r = np.mod(op[0]@r0 + op[1] + epsilon, 1) - epsilon

        exist = False

        for r in generated:

            if np.linalg.norm(current_r - r) < epsilon:

                exist = True
                break

        if exist == False:

            generated.append(current_r)
            R_axis = S @ op[0] @ invS
            T = get_U_transform_matrix_np(R_axis)
            
            U_matrices.append(T)

    U_matrices.insert(0, np.eye(6))

    return U_matrices

def get_symmetric_vectors(vector, operators, epsilon = 1e-3, magnetic = False):

    symmetric_vectors = []
    for operator in operators:

        symmetric_vector = operator(vector)

        if magnetic == False:
            symmetric_vector = np.mod(symmetric_vector + epsilon, 1) - epsilon
        symmetric_vectors.append(symmetric_vector)

    return symmetric_vectors

def get_site_symmetry_operators(position, MSG, epsilon = 1e-3):

    ops = get_operators_sp(MSG)
    site_symmetry_operators = []

    r0 = sp.Matrix(position).applyfunc(lambda x: sp.Mod(x + epsilon, 1) - epsilon)

    for op in ops:

        r = op[0]*r0 + op[1]
        r = r.applyfunc(lambda x: sp.Mod(x + epsilon, 1) - epsilon)

        if (r-r0).norm() < epsilon:

            site_symmetry_operators.append(op[0])

    return site_symmetry_operators

def get_magnetic_site_symmetry_operators(position, MSG, lattice_constants, epsilon = 1e-3):

    ops = get_operators_sp(MSG)
    magnetic_site_symmetry_operators = []

    a = sp.Rational(lattice_constants[0])
    b = sp.Rational(lattice_constants[1])
    c = sp.Rational(lattice_constants[2])

    S = sp.diag(a, b, c)
    invS = sp.diag(sp.Rational(1,a), sp.Rational(1,b), sp.Rational(1,c))

    r0 = sp.Matrix(position).applyfunc(lambda x: sp.Mod(x + epsilon, 1) - epsilon)

    for op in ops:

        r = op[0]*r0 + op[1]
        r = r.applyfunc(lambda x: sp.Mod(x + epsilon, 1) - epsilon)

        if (r-r0).norm() < epsilon:

            magnetic_site_symmetry_operators.append(S*op[0]*invS*op[0].det()*op[2])

    return magnetic_site_symmetry_operators

def solve_free_moments(position, MSG, lattice_constants, epsilon=1e-3):

    ops = get_magnetic_site_symmetry_operators(
        position, MSG, lattice_constants, epsilon=epsilon
    )

    I = sp.eye(3)

    constraints = sp.Matrix.vstack(
        *(R - I for R in ops)
    )

    basis = constraints.nullspace()
    basis_np = [
    np.array(v, dtype=float).flatten()
    for v in basis
    ]

    return basis_np

def solve_free_positions(position, MSG, epsilon=1e-3):

    ops = get_site_symmetry_operators(
        position, MSG, epsilon=epsilon
    )

    I = sp.eye(3)

    constraints = sp.Matrix.vstack(
        *(R - I for R in ops)
    )

    basis = constraints.nullspace()
    basis_np = [
    np.array(v, dtype=float).flatten()
    for v in basis
    ]

    return basis_np

def solve_free_U(position, MSG, lattice_constants, epsilon=1e-3):
    
    position = np.asarray(position, dtype=float)

    U11, U22, U33, U23, U13, U12 = sp.symbols("U11 U22 U33 U23 U13 U12")
    U0 = sp.Matrix([U11, U22, U33, U23, U13, U12])

    ops = get_site_symmetry_operators(position, MSG, epsilon = epsilon)

    Ts = []

    a, b, c = [
        sp.Rational(str(v))
        for v in lattice_constants
    ]
    
    S = sp.diag(a, b, c)
    invS = sp.diag(1/a, 1/b, 1/c)
    
    for R in ops:
    
        R_axis = S * R * invS
        T = get_U_transform_matrix_sp(R_axis)
        Ts.append(T)

    I = sp.eye(6)

    constraints = sp.Matrix.vstack(
        *(T - I for T in Ts)
    )

    basis = constraints.nullspace()
    basis_np = [
    np.array(v, dtype=float).flatten()
    for v in basis
    ]

    return basis_np