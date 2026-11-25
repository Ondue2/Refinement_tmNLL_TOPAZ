import numpy as np
import symmetry as sym

def position_generator(posits, nonmag_minimal_operators):

    sym_positions = {}
    atoms = list(posits.keys())

    for atom in atoms:

        position = posits[atom]
        operators = nonmag_minimal_operators[atom]
        positions = sym.get_symmetric_vectors(position, operators)
        sym_positions[atom] = np.array(positions)

    return sym_positions

def moment_generator(magnetic_moments, mag_minimal_operators):

    sym_moments = {}
    magnetic_atoms = list(magnetic_moments.keys())

    for atom in magnetic_atoms:

        moment = magnetic_moments[atom]
        operators = mag_minimal_operators[atom]
        moments = sym.get_symmetric_vectors(moment, operators, magnetic = True)
        sym_moments[atom] = np.array(moments)

    return sym_moments

def Q_vector(h,k,l, lattice_constants, unit_vectors):

    h = np.array(h)
    k = np.array(k)
    l = np.array(l)
    
    a = lattice_constants[0]
    b = lattice_constants[1]
    c = lattice_constants[2]

    a_vec = a*np.array(unit_vectors[0])
    b_vec = b*np.array(unit_vectors[1])
    c_vec = c*np.array(unit_vectors[2])

    denom = np.dot(a_vec, np.cross(b_vec, c_vec))
    
    b1 = 2*np.pi*np.cross(b_vec, c_vec)/denom
    b2 = 2*np.pi*np.cross(c_vec, a_vec)/denom
    b3 = 2*np.pi*np.cross(a_vec, b_vec)/denom
    
    q_vectors = h[:,None]*b1[None, :] + k[:,None]*b2[None, :] + l[:,None]*b3[None, :]
    
    return q_vectors

def DWF(q_vectors, U, unit_vectors):

    U_mat = np.array([[U[0], U[5], U[4]], [U[5], U[1], U[3]], [U[4], U[3], U[2]]])
    U_mat = unit_vectors.T @ U_mat @ unit_vectors
    
    DWF = np.exp(-(1/2)*(np.matmul(q_vectors, U_mat)*q_vectors).sum(axis = 1))
                 
    return DWF 

def F_square(h,k,l, lattice_constants, posits, occs, Us, nonmag_minimal_operators, U_matricies, unit_vectors, scatt_lengths):
    
    posits_sym = position_generator(posits, nonmag_minimal_operators)
    
    F = 0
    q_vectors = Q_vector(h,k,l, lattice_constants, unit_vectors)
    hkl_array = np.array([h, k, l]).T

    atoms = list(posits.keys())
    
    for atom in atoms:
        
        atom_num = len(posits_sym[atom])
        
        for i in range(atom_num):
            
            coords = np.array(posits_sym[atom][i])

            T = U_matricies[atom][i]
            U = T @ np.asarray(Us[atom])
            
            F += DWF(q_vectors, U, unit_vectors)*occs[atom]*scatt_lengths[atom]*np.exp(-2*np.pi*1j*np.sum(hkl_array*coords[None, :], axis = 1))
            
    F_square = np.abs(F)**2
            
    return F_square

def magnetic_form_factor(q_vectors, magnetic_form_coefs):
    
    Q = np.linalg.norm(q_vectors, axis = 1)
    
    magnetic_form = 0
    
    for i in range(3):
        
        magnetic_form = magnetic_form_coefs[i]*np.exp(-magnetic_form_coefs[3+i]*(Q/(4*np.pi))**2) + magnetic_form
    magnetic_form = magnetic_form_coefs[6] + magnetic_form
        
    return magnetic_form

def projection_factor(q_vectors, m_vector, unit_vectors):
    
    Q = np.linalg.norm(q_vectors, axis = 1)
    q_norm_vecs = q_vectors/Q[:, None]

    m_vector = unit_vectors.T @ m_vector
    
    projection_factor = m_vector[None, :] - q_norm_vecs*np.sum(q_norm_vecs*m_vector[None, :], axis = 1)[:,None]
    
    return projection_factor
    
def magnetic_F_square(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, magnetic_form_coefs):

    posits_sym = position_generator(posits, nonmag_minimal_operators)  
    magnetic_moments_sym = moment_generator(magnetic_moments, mag_minimal_operators)

    F = 0
    q_vectors = Q_vector(h,k,l, lattice_constants, unit_vectors)
    hkl_array = np.array([h, k, l]).T

    magnetic_atoms = list(magnetic_moments.keys())
    
    for atom in magnetic_atoms:
         
        atom_num = len(posits_sym[atom])
        
        for i in range(atom_num):
            
            coords = np.array(posits_sym[atom][i])
            
            m_vector = magnetic_moments_sym[atom][i]

            T = U_matricies[atom][i]
            U = T @ np.asarray(Us[atom])
            
            F += 2.695*projection_factor(q_vectors, m_vector, unit_vectors)*DWF(q_vectors, U, unit_vectors)[:, None]*occs[atom]*magnetic_form_factor(q_vectors, magnetic_form_coefs[atom])[:, None]*np.exp(-2*np.pi*1j*np.sum(hkl_array*coords[None, :], axis = 1))[:,None]
            
    F_square = np.linalg.norm(F, axis = 1)**2
            
    return F_square

def intensity(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, A, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, scatt_lengths, magnetic_form_coefs):

    nonmag = F_square(h,k,l, lattice_constants, posits, occs, Us, nonmag_minimal_operators, U_matricies, unit_vectors, scatt_lengths)
    mag = magnetic_F_square(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, magnetic_form_coefs)

    return A*(nonmag + mag)

def intensity_nonmag(h,k,l, lattice_constants, posits, occs, Us, A, nonmag_minimal_operators, U_matricies, unit_vectors, scatt_lengths):

    nonmag = F_square(h,k,l, lattice_constants, posits, occs, Us, nonmag_minimal_operators, U_matricies, unit_vectors, scatt_lengths)

    return A*nonmag

def intensity_mag(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, A, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, magnetic_form_coefs):

    mag = magnetic_F_square(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, magnetic_form_coefs)

    return A*mag

def intensity_hkl(hkl, lattice_constants, posits, occs, Us, magnetic_moments, A, groups, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, scatt_lengths, magnetic_form_coefs):

    h = hkl["h"]
    k = hkl["k"]
    l = hkl["l"]

    a = A[groups]

    scaled_fsqr = a*intensity(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, 1, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, scatt_lengths, magnetic_form_coefs)

    return scaled_fsqr

def intensity_hkl_extinction(hkl, lattice_constants, posits, occs, Us, magnetic_moments, A, groups, wavelength, ttheta, tbar, g, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, scatt_lengths, magnetic_form_coefs, V_cell):

    h = hkl["h"]
    k = hkl["k"]
    l = hkl["l"]

    a = A[groups]

    Fsqr_nm = intensity_nonmag(h,k,l, lattice_constants, posits, occs, Us, 1, nonmag_minimal_operators, U_matricies, unit_vectors, scatt_lengths)
    Fsqr_m = intensity_mag(h,k,l, lattice_constants, posits, occs, Us, magnetic_moments, 1, nonmag_minimal_operators, U_matricies, mag_minimal_operators, unit_vectors, magnetic_form_coefs)

    Q_nm = 0.01*Fsqr_nm*wavelength**3/(V_cell**2*np.sin(ttheta))
    Q_m = 0.01*Fsqr_m*wavelength**3/(V_cell**2*np.sin(ttheta))

    A_ex = 0.58 + 0.48*np.cos(ttheta) + 0.24*np.cos(ttheta)**2
    B = 0.02 - 0.025*np.cos(ttheta)

    x_nm = (2*np.sqrt(2)/3)*Q_nm*g*tbar
    x_m = (2*np.sqrt(2)/3)*Q_m*g*tbar

    y_nm = 1/np.sqrt(1 + 2.12*x_nm + A_ex*x_nm**2/(1 + B*x_nm))
    y_m = 1/np.sqrt(1 + 2.12*x_m + A_ex*x_m**2/(1 + B*x_m))

    Fsqr_corrected = y_nm*Fsqr_nm + y_m*Fsqr_m

    scaled_fsqr = a*Fsqr_corrected

    return scaled_fsqr