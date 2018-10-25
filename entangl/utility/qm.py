# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

@author: MIUK

Purpose: Deal with the operations on quantum states. So far it is 
limited to pure bi-partite states

TODO:
TODO:
TODO:


"""
import numpy as np
from scipy import linalg, stats, optimize

def dm(state_vectors):
    """ Compute density matrices from a list of states

    Arguments
    ---------
    state_vectors: 2D arrays shape = (nb_vectors, 4)

	To do
	-----
	make more general: only deals with 2 qubits pure states

    """
    ket = state_vectors.reshape((len(state_vectors), 4, 1))
    bra = state_vectors.reshape((len(state_vectors), 1, 4))
    bra = np.conjugate(bra)
    rho = ket*bra
    return rho    
    
def partial_trace(rho, subsystem='A'):
    """ Compute the partial trace of a bipartite system. It relies
    on einstein summation and cope with two type of shapes
    shape = (4 x 4) 
    shape = (N x 4 x 4)
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


def measurements(state_vector, nb_measures, n = [1,0,0], system = 'A'):#, statmode='off', truth='off'):
     """simulates projective measurements on a subsystem.

    Arguments
    ---------
    state_vector:
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
     O = np.array([[n[2], (n[0]- 1j * n[1])], [n[0] +1j * n[1], -1 * n[2]]], dtype = 'complex128').reshape((2,2))
     eigenvals, eigenvecs = np.linalg.eig(O)
     pro_1 = np.matmul(eigenvecs[0].reshape((2,1)), np.conjugate(eigenvecs[0]).reshape((1,2)))
     pro_2 = np.matmul(eigenvecs[1].reshape((2,1)), np.conjugate(eigenvecs[1]).reshape((1,2)))
     
     if system == 'A':
         measure_op1 = np.array([np.kron(pro_1, np.identity(2)),]*len(state_vector)) 
         measure_op2 = np.array([np.kron(pro_2, np.identity(2)),]*len(state_vector))
         
     if system == 'B':
         measure_op1 = np.array([np.kron(np.identity(2), pro_1),]*len(state_vector))
         measure_op2 = np.array([np.kron(np.identity(2), pro_2),]*len(state_vector))
         
     prob_1 = np.matmul(np.conjugate(state_vector).reshape((len(state_vector),1,4)),
                        np.matmul(measure_op1, state_vector.reshape((len(state_vector),4,1))))
     prob_1 = prob_1.reshape((len(state_vector)))
 
     prob_2 = np.matmul(np.conjugate(state_vector).reshape((len(state_vector),1,4)),np.matmul(measure_op2, 
                        state_vector.reshape((len(state_vector),4,1))))
     prob_2 = prob_2.reshape((len(state_vector)))
     
     p_vec = np.array([np.real(prob_1), np.real(prob_2)]).reshape((2,len(state_vector),))
     results_sim = np.zeros((len(state_vector),2,)) # simulated results
     expecs = np.zeros((len(state_vector),))
     
     for j in xrange(len(state_vector)):
         for i in xrange(nb_measures):
             prob_item = np.array([p_vec[0][j], p_vec[1][j]]).reshape((2,))
             outcome = np.random.choice([0,1], p = prob_item)
             results_sim[j][outcome] += 1
         expecs[j] += float((results_sim[j][0] - results_sim[j][1])) / nb_measures

     results_sim = np.concatenate((results_sim, expecs[:,None]), axis=1)
     return results_sim

def vne_of_states(states):
    """ get a measure of entanglement (based on the Von Newman entropy of 
    a list of states) """
    return vne_of_dm(dm(states))

def vne_of_dm(rho):
    """ get the measure of entanglement (based on the Von Newman entropy) of 
    a list of density matrices
    vn_e(rho) = - tr[rho * log(pho)] 
    		  = - sum_i e_i * log(e_i) where e_i's are the eigenvalues of rhos  
    before vn_entropy
    """
    eigvals = [linalg.eigvals(r) for r in rho]
    entropies = [-1 * np.dot(e[e!=0], np.log(e[e!=0])) for e in eigvals]
    return vn_entropy_m
    

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
    
def omop(psi_i, rho): 
    """
    Creates a orthogonal projective operator in the form |psi_i><psi_i|. Can specifiy psi_i
    in the usual notation: zz_oo -> |00><11| etc.
    TODO: verify/test
    """
    psi = str(psi_i)
    z = np.array([1.,0.], dtype = 'complex128').reshape((2,1))
    o = np.array([0.,1.], dtype = 'complex128').reshape((2,1))   
    
    if(psi[0] == 'z'):
        if (psi[1] == 'z'):
            return np.array([np.kron(z, z)*np.transpose(np.kron(z, z)),]*len(rho))
        if( psi[1] == 'o'):
            return np.array([np.kron(z, o)*np.transpose(np.kron(z, o)),]*len(rho))
    if (psi[0] == 'o'):
        if (psi[1] == 'z'):
            return np.array([np.kron(o, z)*np.transpose(np.kron(o, z)),]*len(rho))
        if (psi[1] == 'o'):
            return np.array([np.kron(o, o)*np.transpose(np.kron(o, o)),]*len(rho))


def gen_mop(rho, op = np.array([[1,0],[0,1]], dtype = 'complex128')): # create a list of custom mop that acts on rho
    """
    single system custom operator. So only applies on one subsytem. Must act on the reduced
    density matrix.
    """
    return np.array([op,]*len(rho))
    
def meas_probs(mop, rho): # something wrong with this. Need to work this out
    return np.einsum('ikk',np.matmul(mop, np.matmul(
            np.conjugate(mop.reshape((len(mop), np.shape(mop)[-1], np.shape(mop)[-2]))), rho)))


mop = gen_mop(rho, op = np.kron(np.array([[0,0.5],[0.5,0]], dtype = 'complex128').reshape((2,2)), np.array([[1,0],[0,1]], dtype = 'complex128').reshape((2,2))))
# mpp = X_a [x] I_b
prob = meas_probs(mop, rho) #list of prob values


def dm_after_m(mop, rho): # need to check. not functional 
    "returns density matrix after measurement"
    return np.matmul(mop, np.matmul(rho, 
           np.conjugate(mop.reshape((len(mop), np.shape(mop)[2], np.shape(mop)[1])))))/ meas_probs(mop, rho).reshape((len(rho), 1, 1))

print(dm_after_m(mop, rho))

     
def grid_lambda_gen(res):
    """ Generates lambda values s.t. Von Neuman entropies lie on a regularly spaced grid 
    VNentropy = -(1 - lambda) log(1 - lambda) - lambda log(lambda)
    """
    lambdas = np.zeros((res))
    ent_space = np.linspace(0, np.log(2),res)
    np.random.shuffle(ent_space)
    counter = 0
    for i in ent_space:
        def s(arg, vne = i):
            yum  = -(1-arg)*np.log(1-arg) -arg*np.log(arg) - vne
            yum = np.nan_to_num(yum)
            return yum
        lamb = (op.newton(s, 0.000001))
        if lamb < 0:
            lamb = 0
        la[counter] += lamb  # reverse engineering lambdas from linear vnes by solving non-linear eqn
        counter += 1  
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

def vne_to_lambda(vne):
    """ invert (numerically) lambda_to_vne """ 
    func_root = lambda la: lambda_to_vne(la) - vne
    vne = optimize.newton(func_root, 0.25)
    return vne

def lambda_to_vne(la):
    """ Von Newman entropy from a schmidt coeff lambda
    vne = - [la * log(la) + (1 - la) * log(1 - la)]
    where la (lambda) is the schmidt coefficient
    """
    return - np.nan_to_num((1-la)*np.log(1-la) - la*np.log(la))

def rdm_states_from_lambda(nb_states, lamb = 0.5):
    """ For a fixed Lamb 
    psi = lambd 

    Previoulsy sep_ent_deterministic
    """
    a = np.array([1,0], dtype = 'complex128').reshape((2,1))    # define fixed basis for A and B
    a = np.array([a,]*nb_states)                                    # create copies for vectorized generation
    b = np.array([0, 1], dtype = 'complex128').reshape((2,1)) 
    b = np.array([b,]*items)
    
    basis1 = np.kron(a,b)      # take the tensor product to create bipartite basis
    basis2 = np.kron(b,a)
    
    # Ua [x] Ub using scipy.stats.unitary_group random matrix generator
    
    ua = stats.unitary_group.rvs(2, size=items) # random unitaries from group U(2)
    detua = np.sqrt(np.linalg.det(ua))    # normalization by determinant to get unitaries from group SU(2) corresponding to spinor rotations
    ua = ua / np.repeat(detua,4).reshape(np.shape(ua))
    
    ub = stats.unitary_group.rvs(2, size=items)
    detub = np.sqrt(np.linalg.det(ub))    
    ub = ub / np.repeat(detub,4).reshape(np.shape(ub))
    
    unitaries = np.kron(ua, ub) 
    state_vectors = np.matmul(unitaries, np.array(np.sqrt(lamb)*basis1 + np.sqrt(1-lamb)*basis2))
    
    return state_vectors


def gen_dataset(state_vectors, nb_measures):
    """ Generate data_set (both input and ouputs) based on a list of state vectors.
    data_set[i] = (X_A, Y_A, Z_A, X_B, Y_B, Z_B, E) """
    x1 = measurements(state_vectors, nb_measures, [1,0,0], 'A')
    y1 = measurements(state_vectors, nb_measures, [0,1,0], 'A')
    z1 = measurements(state_vectors, nb_measures, [0,0,1], 'A')
    x2 = measurements(state_vectors, nb_measures, [1,0,0], 'B')
    y2 = measurements(state_vectors, nb_measures, [0,1,0], 'B')
    z2 = measurements(state_vectors, nb_measures, [0,0,1], 'B')
    
    ents = vn_ent(partial_trace(dm(state_vectors)))
    ents = ents.reshape((len(state_vectors), 1))
    ents = np.pad(ents, (0,2), mode = 'constant')[:-2] # add zeros to make shape compatible with measurements x1,x2...
    
    data_set = np.stack((x1,y1,z1,x2,y2,z2,ents), axis=1)
    
    return data_set0
    




if __name__ == '__main__':
	# For sake of convenience some testing are done here
	# should be moved somewhere else
	test_creating = False
	test_loading = False

	if(test_creating):
	# script to generate data
	    list_nb_measures = [1, 10, np.inf] #Nb of measurements 
		nb_examples = 1000000  # stepping in entropy values for which we generate states deterministically
		lambdas = rdm_lambda_gen(entropy_spacing)
		vecs = [rdm_states_from_lambda(1, lamb=l)[0] for l in lambdas]
		np.random.shuffle(state_vectors)  # shuffle to discard the order of creation of these states

		for nb_m in list_nb_measures:
		    data_set = gen_dataset(state_vectors, nb_m) 
		  
		    with file(('dataset5_%smeas.txt' % nb_m), 'w+') as outfile:
		        # Writing a header here for the sake of readability
		        # Any line starting with "#" will be ignored by numpy.loadtxt
		        outfile.write('# Array shape: {0}\n'.format(data_set.shape))
		        outfile.write('# Number of measuremets: {0} \n'.format(nb_m) )
		        outfile.write('# Number of state vectors with particular entropy value: {0} \n'.format(resolution*resolution))
		        outfile.write('# VN Entropy spacing from 0 to 0.5: {0} \n'.format(entropy_spacing))
		        outfile.write('# Third column calculates expectation value of sigma using bin_dist. Extra zeros for vne for numpy array compatibility \n')
		    
		        for data_slice in data_set:
		            np.savetxt(outfile, data_slice, fmt='%5.12g') #'%-7.2f' '%.4e'
		            outfile.write('# New vector with column format (xa, ya, za, xb, yb, zb, vne) \n')

	with file('states5.txt', 'w+') as outfile:
	    outfile.write('# State vectors for dataset3 shaped: {0}\n'.format(state_vectors.shape))	    
	    for state in state_vectors:
	        np.savetxt(outfile, state, fmt = '%5.12g')
	        outfile.write('# State vector \n')
	
    if(test_loading):
	# script to load data and inspect them
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    