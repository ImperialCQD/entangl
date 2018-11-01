# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

@author0: MIUK
@author1: FS

Purpose: Deal with the operations on pure bipartite 
quantum states

Bipartite states are represented using kronecker product 
i.e. with Psi_A = [a,b] Psi_b = [c, d] 
-> Psi_AB = Psi_A x Psi_B = [ac, ad, bc, bd] (x is a tensor prod)

Conventions
-----------
+ Computational basis
    |0> = [1,0] |1> = [0,1]
    then |00> = [1,0,0,0], |01> = [0,1,0,0], |10> = [0,0,1,0] 
    and |11> = [0,0,0,1] 

+ Pauli matrices
  X =  0  1  //  Y = 0 -i  //  Z = 1  0
       1  0          i  0          0 -1
d: refers to the dimension of the hilbert space

Caredull with the dimensions and shapes

"""
import numpy as np
from scipy import linalg, stats, optimize



# ===================================================================================== #
# Manipulation of pure bi-partite states (sometimes slightly more general)
# ===================================================================================== #
qb_0 = np.array([1., 0], dtype='complex128')
qb_1 = np.array([0, 1.], dtype='complex128')
qb_basis = np.stack([qb_0, qb_1], 1)
qbs_00 = np.kron(qb_0, qb_0)
qbs_01 = np.kron(qb_0, qb_1)
qbs_10 = np.kron(qb_1, qb_0)
qbs_11 = np.kron(qb_1, qb_1)
# Maybe change the convention such that it matches the convention for the qubits
# representation
X = np.array([[0, 1.], [1., 0]], dtype='complex128') 
Y = np.array([[0, -1.j], [1.j, 0]], dtype='complex128') 
Z = np.array([[1., 0], [0 ,-1]], dtype='complex128')  
I = np.array([[1., 0], [0 ,1.]], dtype='complex128')
bell0 = 1/np.sqrt(2) * np.array([1., 0, 0, 1.], dtype='complex128')
bell1 = 1/np.sqrt(2) * np.array([1., 0, 0, -1.], dtype='complex128')
bell2 = 1/np.sqrt(2) * np.array([0, 1., 1., 0], dtype='complex128')
bell3 = 1/np.sqrt(2) * np.array([0, 1., -1., 0], dtype='complex128')

def state_to_dm(states):
    """ Compute density matrices from a list of states
    Only deals with bipartite state

    Arguments
    ---------
    state_vectors: numy 
        1d shape = (4) single state
        3d shape = (nb_states, 4)        
    """
    n_dim = np.ndim(states)
    if(n_dim == 2):   
        ket = np.reshape(states, (len(states), 4, 1))
        bra = np.conjugate(np.reshape(states, (len(states), 1, 4)))
    elif(n_dim == 1):
        ket = np.reshape(states, (4, 1))
        bra = np.conjugate(np.reshape(states, (1, 4)))
    rho = ket * bra
    return rho    

def ip(state0, state1):
    """ Inner product between two states (or list)
    
    Arguments
    ---------
    state: np.array
        1d shape (d) i.e. only one state
        2d shape (n_states, d)
    
    Output
    ------
        2-d np.array with dimension (nb_states0, nb_states_1)
    """
    state0 = state0[np.newaxis, :] if np.ndim(state0) == 1 else state0
    state1 = state1[np.newaxis, :] if np.ndim(state1) == 1 else state1
    return np.dot(np.conjugate(state0), np.transpose(state1))

def ip_square(state0, state1):
    """ square module of the inner product 
        Same output dimensions as teh output of ip()"""
    return np.square(np.abs(ip(state0, state1)))

def norm(states):
    """ Norm of state (or collection of states)
    Same output dimensions as teh output of ip()"""
    return np.sqrt(np.real(ip(states, states)))

def partial_trace(rho, subsystem='A'):
    """ Compute the partial trace of a bipartite system. It relies
    on einstein summation

    Arguments
    ---------
    rho: np.array
        density matrix
        shape = (4 x 4) one density matrix 
        shape = (N x 4 x 4) a collection of density matrices
    """
    n_dim = np.ndim(rho)
    if(n_dim == 3):   
        rho_rshp = np.reshape(rho, (len(rho), 2, 2, 2, 2))
        if(subsystem == 'A'):
            rho_partial = np.einsum('aijik->ajk', rho_rshp)
        else:
            rho_partial = np.einsum('aijkj->aik', rho_rshp)        
    elif(n_dim == 2):
        rho_rshp = np.reshape(rho, (2, 2, 2, 2))
        if(subsystem == 'A'):
            rho_partial = np.einsum('ijik->jk', rho_rshp)
        else:
            rho_partial = np.einsum('ijkj->ik', rho_rshp)        
    return np.array(rho_partial);

def meas_one_sub(states, nb_measures=np.inf, n = [1,0,0], subsystem = 'A'):
    """simulates projective measurements on only one of the subsystem.
    Works only for two-qubits states
    
    Arguments
    ---------
    states:
         a list of state vectors
    nb_measures: int
        number of measurements to perform
        by default infinity, i.e. return the true expectation value of the 
        operator, else if it is a finite number will return empirical value
    n: list<int>
        Encode the mesurement hermitian operator 
        [a, b, c] -> O = aX + bY + cZ
    subsystem: <str>
        'A' or 'B' on which subsystem do we perform the measurement

    Output
    ------
        res 1d np.array with the same length as states
        Estimation of the expected value <O>
    """
    O =  n[0] * X + n[1] * Y + n[2] * Z
    eigenvals, eigenvecs = np.linalg.eig(O)
    assert _is_real_enough(eigenvals), "O has complex (i.e not real) eigenvalues"
    eigenvals = np.real(eigenvals)
    if(subsystem == 'A'):
        proj_basis = [np.stack([np.kron(e, qb_0), np.kron(e, qb_1)]) for e in 
                     np.transpose(eigenvecs)]    
    else:
        proj_basis = [np.stack([np.kron(qb_0, e), np.kron(qb_1, e)]) for e in 
             np.transpose(eigenvecs)]    
    
    probs = proj_proba(states, basis = proj_basis)
    assert np.allclose(np.sum(probs, 1), 1.), "Measurement probas don't sum to 1"
    if(nb_measures == np.inf):
        res = np.dot(probs, eigenvals)
    else:
        freq = np.random.binomial(nb_measures, probs[:,0]) / nb_measures
        res = eigenvals[0] * freq + eigenvals[1] * (1-freq) 
    return res


def proj_proba(states, basis = None):
    """ proba of being projected on a list of vectors (subspaces)

    Arguments
    ---------
        states: np.array
            1d shape = (d)
            2d shape = (nb_states, d)
        basis: list
            In which basis do we perofrm the measurements. 
            basis[i] is a np.array 
                1d shape = (d) specify the projection onto a vector
                2d shape = (n_sub, d) specify the projection onto a subspace
            By default if the basis is not specified it will be the 
            computational basis

    Output
    ------
        probs: np.array 
            shape = (nb_states,nb_subspaces_proj)
            where proba[n, i] is the proba of the n-th state 
            to be projected on the i-th basis vector
    """
    states = np.array(states)[np.newaxis, :] if np.ndim(states) == 1 else np.array(states)
    dim_H = states.shape[1]
    basis = np.eye(dim_H) if basis is None else basis
    probs = [np.sum(ip_square(states, b), 1) for b in basis]
    return np.transpose(probs)

def samples_from_proba(probs, nb_samples):
    """ to do later"""
    pass

# ===================================================================================== #
# Entanglement 
# Mostly based on the Von Newman Entropy of the partial trace of a density matrix
# It could be extended to incorporate other measures
# ===================================================================================== #
def entangl_of_states(states, subsystem = 'A'):
    """ get a measure of entanglement (based on the Von Newman entropy of 
    the partial trace) of a list of states.
    For entangled (separable) verify it retuns log(2) (0)
    
    Arguments
    ---------
    states: nd-array
        shape = ()
        shape = ()
    subsystem: str
        On which subsystem do we trace out

    TODO: verif that it returns the same result whatever the subsystem is
    """
    return vne_of_dm(partial_trace(state_to_dm(states), subsystem = subsystem))

def vne_of_dm(rhos):
    """ Von Newman entropy of a list of density matrices
    vne_of_dm(rhos) = - tr[rho * log(pho)] 
    		= - sum_i e_i * log(e_i) where e_i's are the eigenvalues of rhos  
    
    Arguments
    ---------
    rhos: nd-array
        shape = (d,d) one density matrix (d is the dim of the Hilbert space)
        shape = (nb_rho, d, d) collection of density matrices
    """
    if np.ndim(rhos) == 2:
        e = linalg.eigvals(rhos)
        assert _is_real_enough(e), "Imaginary values of the eigenvalues are not null"
        e = np.real(e)
        vn_entrop = - np.dot(e[e!=0], np.log(e[e!=0]))
    elif(np.ndim(rhos) == 3):
        vn_entrop = np.array([vne_of_dm(r) for r in rhos])
    return vn_entrop
    
def concurrence(psi): 
    """ bipartite concurrence
    Another measure of entanglement. C = abs(alpha*delta - beta*gamma). C = 0 if seperable 
    and 1 if maximally entangled. C would be between 0 and 1 otherwise suggesting entanglement.
    
    TODO: verify/test
    """
    conarray = np.zeros((len(psi),1,1)) + 0j*np.zeros((len(psi),1,1))
    for index in range(len(psi)):
        conarray[index] = 2*np.sqrt((psi[index][0]*psi[index][3] - 
                          psi[index][1]*psi[index][2])*np.conjugate(psi[index][0]*psi[index][3] - 
                          psi[index][1]*psi[index][2]))
    return conarray
    

# ===================================================================================== #
# Use of Schmidt decomposition to generate random (two qubits) statesbn with a certain
# amount of entanglement
# |psi> = sqrt(lambda) |i>_A|i>_B + sqrt(1-lambda) |j>_A|j>_B
# Then Entanglement(|psi>) = - [lambda * log(lambda) + (1-lambda) * log(1-lambda)
# 
# TODO: test vne_to_lambda
# ===================================================================================== #
def grid_lambda_gen(res):
    """ Generates lambda values s.t. entanglements lie on a regularly spaced grid 
    where the resolution is res. 
    *** Not really in use ***
    """
    ent_space = np.linspace(0, np.log(2),res)
    np.random.shuffle(ent_space)
    la = [ent_to_lambda(v) for v in ent_space]
    la = np.clip(la, 0, 1)
    return la

def rdm_lambda_gen(nb):
    """ Generates lambda values (cf. Schmidt decomposition) s.t. the associated 
    Von Newman entropies have an uniform distribution on [0, ln(2))
    vne = - [la * log(la) + (1 - la) * log(1 - la)] 
    """
    vne = np.log(2) * np.random.random(nb)
    la = [ent_to_lambda(v) for v in vne]
    la = np.clip(la, 0, 1)
    return la 

def ent_to_lambda(ent):
    """ invert (numerically) lambda_to_vne """ 
    if(np.ndim(ent) == 0):
        if(ent == 0):
            lambd = 0
        else:
            func_root = lambda la: lambda_to_ent(la) - ent
            lambd = optimize.newton(func_root, 10e-12)
    else:
        tmp = [ent_to_lambda(e) for e in np.nditer(ent)]
        lambd = np.reshape(tmp, np.shape(ent))
    return lambd

def lambda_to_ent(la):
    """ entanglement from a schmidt coeff lambda
    ent = - [la * log(la) + (1 - la) * log(1 - la)]
    where la (lambda) is the Schmidt coefficient
    """
    return - np.nan_to_num((1-la)*np.log(1-la) + la*np.log(la))

def rdm_states_from_lambda(lambdas):
    """ For a collection of lambdas generate a collection of 
    random states. Build based on the Schmidt decomposition
    of a pure bipartite state:
    |psi> = sqrt(lambda) |i_A>|i_B> + sqrt(1 - lambda) |j_A>|j_B>
    where {|i_A>, |i_B>} ({|j_A>, |j_B>}) are a basis of the system A (B)
    """
    nb_states = 1 if np.ndim(lambdas) == 0  else len(lambdas)
    basis_A = gen_rdm_basis(nb_states, 2)
    basis_B = gen_rdm_basis(nb_states, 2)
    if(nb_states > 1):
        states = [np.sqrt(l) * np.kron(i_A[:,0], i_B[:,0]) + np.sqrt(1 - l) * np.kron(i_A[:,1], i_B[:,1]) 
            for l, i_A, i_B in zip(lambdas, basis_A, basis_B)]
    else:
        states = np.sqrt(lambdas) * np.kron(basis_A[:,0], basis_B[:,0]) 
        states += np.sqrt(1 - lambdas) * np.kron(basis_A[:,1], basis_B[:,1])
    return states

def gen_rdm_basis(nb, dim):
    """ Generate a random basis. Based on the generation of random
    unitaries U(N) from scipy.stats.unitary_group.rvs.

    Argument
    --------
        nb: int
            number of states to generate
        dim: int
            dimension of the Hilbert space
    Output
    ------
        2d or 3-d np.array
        shape = ((nb), dim, dim)
        where output[(n), :, i] is the i-th state of the n-th basis 
    """
    u = stats.unitary_group.rvs(dim, size=nb)
    return u

def _is_real_enough(e):
    """ test if the imaginary part is null (or really close to 0)"""
    return np.allclose(np.imag(e), 0.)
    

if __name__ == '__main__':
	# For sake of convenience some testing are done here
	# should be moved somewhere else later on
    test_entanglement = False
    test_measurements = False
    test_gen_rdm_states = False
    test_ent_to_lambdas = False
    states_test = [qbs_00, bell0]    
    
    if(test_entanglement):
        # Verify we find the expected values for two known states
        # qbs_00 is a pure separable state (i.e. entanglement should be 0), 
        # while bell0 is a maximally entangled pure state (i.e. entanglement 
        # should be log(2))
        print("Entanglement of |00> and (|00> + |11>)/sqrt(2):")
        entangl_of_states(states_test)
	
    if(test_measurements):
        # Infinite number of measurements (i.e. perfect measurement)
        print("Perfect measurements of |00>, Bell0 in the X basis")
        print(meas_one_sub(states_test, np.inf, [1, 0, 0]))
        print("Perfect measurements of |00>, Bell0 in the Y basis")
        print(meas_one_sub(states_test, np.inf, [0, 1, 0]))
        print("Perfect measurements of |00>, Bell0 in the Z basis")
        print(meas_one_sub(states_test, np.inf, [0, 0, 1]))
        
        # One measurement 
        print("1 measurement of |00>, Bell0 in the X basis")
        print(meas_one_sub(states_test, 1, [1, 0, 0]))
        print("1 measurement of |00>, Bell0 in the Y basis")
        print(meas_one_sub(states_test, 1, [0, 1, 0]))
        print("1 measurement of |00>, Bell0 in the Z basis")
        print(meas_one_sub(states_test, 1, [0, 0, 1]))
    
        # Ten measurement 
        print("10 measurement of |00>, Bell0 in the X basis")
        print(meas_one_sub(states_test, 10, [1, 0, 0]))
        print("10 measurement of |00>, Bell0 in the Y basis")
        print(meas_one_sub(states_test, 10, [0, 1, 0]))
        print("10 measurement of |00>, Bell0 in the Z basis")
        print(meas_one_sub(states_test, 10, [0, 0, 1]))
        
        # Ten thousands measurements (should be really close to the real values)
        print("10000 measurement of |00>, Bell0 in the X basis")
        print(meas_one_sub(states_test, 10000, [1, 0, 0]))
        print("10000 measurement of |00>, Bell0 in the Y basis")
        print(meas_one_sub(states_test, 10000, [0, 1, 0]))
        print("10000 measurement of |00>, Bell0 in the Z basis")
        print(meas_one_sub(states_test, 10000, [0, 0, 1]))
    
    if(test_ent_to_lambdas):
        # Test of going back and forth between lambdas and entanglements
        ent_test =  np.random.uniform(0, np.log(2), 1000)
        l_test = ent_to_lambda(ent_test)
        ent_final = lambda_to_ent(l_test)
        assert np.allclose(ent_test, ent_final), "Problem in test_ent_to_lambdas "
    
    if(test_gen_rdm_states):
        ent_test = np.random.uniform(0, np.log(2), 100)
        l_test = ent_to_lambda(ent_test)
        states = rdm_states_from_lambda(l_test)
        ent_final = entangl_of_states(states)
        assert np.allclose(ent_test, ent_final), "Problem in test_random_states "
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    