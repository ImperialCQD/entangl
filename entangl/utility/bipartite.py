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
"""
import numpy as np
from scipy import linalg, stats, optimize



# ===================================================================================== #
# Manipulation of pure bi-partite states (some times it's more general)
# 
# 
# ===================================================================================== #
qb_0 = np.array([1., 0], dtype='complex128')
qb_1 = np.array([0, 1.], dtype='complex128')
qbs_00 = np.kron(qb_0, qb_0)
qbs_01 = np.kron(qb_0, qb_1)
qbs_10 = np.kron(qb_1, qb_0)
qbs_11 = np.kron(qb_1, qb_1)
X = np.array([[0, 1.], [1., 0]], dtype='complex128') 
Y = np.array([[0, -1.j], [1., 0]], dtype='complex128') 
Z = np.array([[0, 1.], [1. ,0]], dtype='complex128')  
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
    n_dim = np.ndim(rho)
    if(n_dim == 2):   
        ket = states.reshape((len(state_vectors), 4, 1))
        bra = np.conjugate(states.reshape((len(states), 1, 4)))
    elif(n_dim == 1):
        ket = states.reshape(4, 1)
        bra = np.conjugate(states.reshape(1, 4))
    rho = ket * bra
    return rho    

def ip(state0, state1):
    """ Inner product between two states (or list) <state0 | state1>"""
    if(np.ndim(state0) == 1):
        ip = np.dot(np.conjugate(state0), state1)
    if(np.ndim(state0) == 2):
        ip = np.sun(np.conjugate(state0) * state1, 1)
    return ip

def norm(states):
    """ Norm of state (or collection of states)"""
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
        rho_rshp = rho.reshape([len(rho), 2, 2, 2, 2])
        if(subsystem == 'A'):
            rho_partial = np.einsum('aijik->ajk', rho_rshp)
        else:
            rho_partial = np.einsum('aijkj->aik', rho_rshp)        
    elif(n_dim == 2):
        rho_rshp = rho.reshape([2, 2, 2, 2])
        if(subsystem == 'A'):
            rho_partial = np.einsum('ijik->jk', rho_rshp)
        else:
            rho_partial = np.einsum('ijkj->ik', rho_rshp)        
    return np.array(rho_partial);


def measurements(states, nb_measures, n = [1,0,0], system = 'A'):
    """simulates projective measurements on one of the subsystem.
    Works only for two-qubits states
    
    Arguments
    ---------
    states:
         a list of state vectors
    nb_measures: int
        number of measurements to perform
    n: list<int>
        Encode the mesurement hermitian operator 
        [a, b, c] -> O = aX + bY + cZ
    system: <str>
        'A' or 'B' on which system do we perform the measurement

    Output
    ------

    TODO: Probably could be simplified
    TODO: When nb.measures = np.inf return expectation
    """
    O =  n[0] * X + n[1] * Y + n[2] * Z
    eigenvals, eigenvecs = np.linalg.eig(O)
    probs = np.array([e * proba_one_subsystem(states, system) for e in np.transpose(eigenvecs)])
    assert np.allclose(np.sum(probs, 1), 1.)
    if(nb_measures == np.inf):
        res = np.matmul(probs, eigenvals)
    else:
        freq = np.random.binomial(nb_measures, probs[:,0]) / nb_measures
        res = eigenvals[0] * freq + eigenvals[1] * (1-freq) 
    return res


def proba_one_subsystem(states, system = 'A'):
    """ proba of a bipartite state to be projected on the 
    computational basis on one of the system. Relies on the
    conventions used. Only work for two-qubits states
    E.g. two qubit states -> probas of the first(second) qubit
    to be observed in 0 or 1

    Arguments
    ---------
        states: nb-array
            shape = (2)
            shape = (nb_states, 2)
        system: str
            Which subsystem are we observing 

    Output
    ------
        proba: np.array 
            shape = (nb_states,2) if 2-d input
            shape = (2) if 1-d input
            where proba[n, i] is the proba of the n-th state 
            to be projected on the i-th basis vector
    """
    states = states[np.newaxis, :] if np.ndim(states) == 1          
    if(system == 'A'):
        p = np.array(states[:, 0] + states[:, 1], states[:, 2] + states[:, 3])
    else(system == 'B'):
        p = np.array(states[:, 0] + states[:, 2], states[:, 1] + states[:, 3])
    return np.squeeze(p)


# ===================================================================================== #
# Entanglement 
# Mostly based on the Von Newman Entropy of the partial trace of a density matrix
# It could be extended to incorporate other measures
# ===================================================================================== #
def entangl_of_states(states, system = 'A'):
    """ get a measure of entanglement (based on the Von Newman entropy of 
    the partial trace) of a list of states.
    
    Arguments
    ---------
    states: nd-array
        shape = ()
        shape = ()
    system: str
        On which subsystem do we trace out

    TODO: it should return the same result whatever the subsystem is
    TODO: for entangled (separable) verify it retuns log(2) (0)
    """
    return vne_of_dm(partial_trace(state_to_dm(states), system = system))

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
    if np.ndim(rhos):
        e = linalg.eigvals(rhos)
        vn_entrop = - np.dot(e[e!=0], np.log(e[e!=0]))
    elif(np.ndim(rhos) == 3):
        vn_entrop = np.array([vne_of_dm(r) for r in rho])
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
    la = [vne_to_lambda(v) for v in vne]
    la = np.clip(la, 0, 1)
    return la

def rdm_lambda_gen(nb):
    """ Generates lambda values (cf. Schmidt decomposition) s.t. the associated 
    Von Newman entropies have an uniform distribution on [0, ln(2))
    vne = - [la * log(la) + (1 - la) * log(1 - la)] 
    """
    vne = np.log(2) * np.random.random(nb)
    la = [vne_to_lambda(v) for v in vne]
    la = np.clip(la, 0, 1)
    return la 

def ent_to_lambda(vne):
    """ invert (numerically) lambda_to_vne """ 
    func_root = lambda la: lambda_to_vne(la) - ent
    lambd = optimize.newton(func_root, 0.25)
    return lambd

def lambda_to_ent(la):
    """ entanglement from a schmidt coeff lambda
    ent = - [la * log(la) + (1 - la) * log(1 - la)]
    where la (lambda) is the Schmidt coefficient
    """
    return - np.nan_to_num((1-la)*np.log(1-la) - la*np.log(la))

def rdm_states_from_lambda(lamb = 0.5):
    """ For a collection of lambdas generate a collection of 
    random states
    """
    basis_A = gen_rdm_states(nb_states, 2)
    basis_B = gen_rdm_states(nb_states, 2)
    vect = np.kron(np.sqrt(lamb) * basis_A, np.sqrt(1 - lamb) * basis_B)
    return vect

def gen_rdm_states(nb, dim):
    """ Generate random states. Based on the generation of random
    unitaries U(N) from scipy.stats.unitary_group.rvs.
    state = U x |i> ehere U is the random unitary and i is a fixed 
    basis vector

    Argument
    --------
        nb: int
            number of states to generate
        dim: int
            dimension of the Hilbert space
    """
    u = stats.unitary_group.rvs(dim, size=nb)
    one = np.zeros((nb, dim, 1))
    one[:,-1,:] = 1
    return np.matmul(u, one)

if __name__ == '__main__':
	# For sake of convenience some testing are done here
	# should be moved somewhere else
	test_random_states = False
	test_vne = False
    test_vne_to 

	
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    