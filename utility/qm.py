# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

Expanded to complex and considerably increased efficiency.
@author: MIUK


Purpose: 
Deal with the operations on quantum states. So far it is limited
to pure bi-partite state
"""

import numpy as np
import scipy.linalg as sc
import scipy.stats


def state_vectors(items):
    "creates complex random state vectors (array type = 1r x 4c) in the half-open interval [-1,1)"
    
    states = 2*(np.random.random([items, 4]))-1 # fetches 4 random numbers
    states = states + (2j*np.random.random([items, 4])-1j) # adds 4 complex numbers
    norms = np.sqrt(np.sum(states*np.conjugate(states), 1))
    norms = np.repeat(norms, 4).reshape(items, 4)
    states = states / norms # normalized
    
    return states

#==============================================================================

def sep_ent(items, lamda = 0.5):
     """
     Using Schmidt decomposition, generates maximally entangled or seperable states based 
     on lamda value.if lambda = 0 then all seperables are generated and if lamda = 0.5 maximally 
     entangled ones are generated.
     """
     alpha = 2*(np.random.random([items]))-1  # random numbers in [-1,1)
     alpha = alpha + (2j*np.random.random([items])-1j) # made complex
     for index in range(items):
         if alpha[index]*np.conjugate(alpha[index]) > 1:
             alpha[index] = alpha[index] / np.linalg.norm(alpha[index]) # checked for normalization
     beta = 2*(np.random.random([items]))-1
     beta = beta + (2j*np.random.random([items])-1j)    
     for index in range(items):
         if beta[index]*np.conjugate(beta[index]) > 1:  # so that the sqrt is not complex
             beta[index] = beta[index] / np.linalg.norm(beta[index])
     
     basis1 = np.array([[alpha*beta],[alpha*np.sqrt(1-np.conjugate(beta)*beta)], 
                        [np.sqrt(1-np.conjugate(alpha)*alpha)*beta],   # creating an orthogonal iA [x] iB
                         [np.sqrt(1-np.conjugate(alpha)*alpha)*np.sqrt(1-
                          np.conjugate(beta)*beta)]]
                          , dtype = 'complex128')
     basis1 = np.transpose(basis1)
     basis2 = np.array([[np.sqrt(1-np.conjugate(alpha)*alpha)*np.sqrt(1-np.conjugate(beta)*beta)],
                         [-1*np.sqrt(1-np.conjugate(alpha)*alpha)*np.conjugate(beta)],
                         [-1*np.conjugate(alpha)*np.sqrt(1-np.conjugate(beta)*beta)], 
                         [np.conjugate(alpha)*np.conjugate(beta)]]
                          , dtype = 'complex128')     # creating an orthogonal jA [x] jB
     basis2 = np.transpose(basis2)
     state_vectors = np.array(np.sqrt(lamda)*basis1 + np.sqrt(1-lamda)*basis2)
     state_vectors = state_vectors.reshape((items, 4, 1))
     return state_vectors

def sep_ent_deterministic(resolution, lamb = 0.5):
    "generating resolution^2 seps or max ents using unitary transformation U_a [x] U_b"
    
    m1 = 1  #arbitrary fixed basis with non zero values
    m2 = 2  
    m3= 3
    m4= 4
    
    k = m1+1j*m2
    j = m3+1j*m4
    
    a = np.array([k,j], dtype = 'complex128').reshape((2,1))
    a = a / np.linalg.norm(a)
    b = np.array([-np.conjugate(j), np.conjugate(k)], dtype = 'complex128').reshape((2,1)) 
    b = b / np.linalg.norm(b)
    
    basis1 = np.array([np.kron(a,b),]*resolution*resolution) # list of untransformed Schmidt bases
    basis2 = np.array([np.kron(b,a),]*resolution*resolution)
    
    x = np.zeros((resolution*resolution, 4, 4), dtype = 'complex128')
    counter = 0    # creating a unitary applying on one system with a range of (theta, phi) values
    for theta in np.linspace(0+np.pi/resolution, np.pi-(np.pi/resolution), resolution):
        for phi in np.linspace(0, (2*np.pi-(np.pi/resolution)), resolution):
            unitary_a = np.array([[np.cos(theta), -1*np.sin(theta)*np.exp(-1j*phi)], [np.sin(theta)*np.exp(1j*phi), np.cos(theta)]], 
                                   dtype = 'complex128').reshape((2,2))  # unitary on A
            unitary_b = np.array([[1., 0.], [0., 1.]], 
                                   dtype = 'complex128').reshape((2,2))
            x[counter] = np.kron(unitary_a, unitary_b)
            counter += 1
    unitaries = x # generates a list of unitary operators Ua [x] Ub
    state_vectors = np.matmul(unitaries, np.array(np.sqrt(lamb)*basis1 + np.sqrt(1-lamb)*basis2))
    # basis transformation above
    return state_vectors
#==============================================================================
    
def dm(state_vectors):
    """ Compute density matrices for a list of states

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
    

def red_dm(rho_ab, subsystem="A"):
    """compute the reduced density matrix: partial trace over one of the subsystem
    """
    rho = rho_ab
    up = np.array([1.,0.]).reshape((2,1)) + 0j*np.array([1.,0.]).reshape((2,1))
    down = np.array([0.,1.]).reshape((2,1)) + 0j*np.array([1.,0.]).reshape((2,1))
    
    l11 = np.array([up*up.reshape((1,2)),]*len(rho)) # 2 x 2 basis for red rho
    l12 = np.array([up*down.reshape((1,2)),]*len(rho))
    l21 = np.array([down*up.reshape((1,2)),]*len(rho))
    l22 = np.array([down*down.reshape((1,2)),]*len(rho))
    
    zz_zz = np.kron(up, up)*np.kron(up, up).reshape((1,4)) # 4 x 4 basis to extract rho values
    zz_zo = np.kron(up, up)*np.kron(up, down).reshape((1,4))
    zz_oz = np.kron(up, up)*np.kron(down, up).reshape((1,4))  # manually computes reduced density matrix values
    zo_zo = np.kron(up, down)*np.kron(up, down).reshape((1,4))
    zo_oo = np.kron(up, down)*np.kron(down, down).reshape((1,4))
    oz_oz = np.kron(down, up)*np.kron(down, up).reshape((1,4))
    oz_oo = np.kron(down, up)*np.kron(down, down).reshape((1,4))
    
    if subsystem == "A":  # calculates partial  trace over B
        a11 = rho*(zz_zz + zo_zo)
        a11 = np.sum(np.sum(a11,1),1).reshape((len(rho),1,1))
        a22 = ((1.+0.j)-a11)*l22
        a11 = a11*l11
        a12 = rho*(zz_oz+zo_oo)
        a12 = np.sum(np.sum(a12,1),1).reshape((len(rho),1,1))
        a21 = np.conjugate(a12*l21)
        a12 = a12*l12
        return a11+a12+a21+a22
    
    if subsystem == "B": # calculates partial  trace over A 
        a11 = rho*(zz_zz + oz_oz)
        a11 = np.sum(np.sum(a11,1),1).reshape((len(rho),1,1))
        a22 = ((1.+0.j)-a11)*l22
        a11 = a11*l11
        a12 = rho*(zz_zo+oz_oo)
        a12 = np.sum(np.sum(a12,1),1).reshape((len(rho),1,1))
        a21 = np.conjugate(a12*l21)
        a12 = a12*l12
        return a11+a12+a21+a22


def vn_entropy(rho):
    """Compute the Von neuman entropy over a list of density matrices
    vn_e(rho) = - tr[rho * log(pho)] 
    		  = - sum_i e_i * log(e_i) where e_i's are the eigenvalues of rhos  
    """
    eigvals = [sc.eigvals(r) for r in rho]
    entropies = [-1 * np.dot(e[e!=0], np.log(e[e!=0])) for e in eigvals]
    return vn_entropy_m
    
    

def concurrence(psi): # bipartite concurrence
    """
    another measure of entanglement. C = abs(alpha*delta - beta*gamma). C = 0 if seperable 
    and 1 if maximally entangled. C would be between 0 and 1 otherwise suggesting entanglement.
    """
    conarray = np.zeros((len(psi),1,1)) + 0j*np.zeros((len(psi),1,1))
    for index in range(len(psi)):
        conarray[index] = 2*np.sqrt((psi[index][0]*psi[index][3] - 
                          psi[index][1]*psi[index][2])*np.conjugate(psi[index][0]*psi[index][3] - 
                          psi[index][1]*psi[index][2]))
    return conarray
    
def omop(psi_i, rho): # this acts on the whole density matrix
    """
    Creates a orthogonal projective operator in the form |psi_i><psi_i|. Can specifiy psi_i
    in the usual notation: zz_oo -> |00><11| etc.
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

    #==============================================================================
def wrapper(func, *args, **kwargs):
     def wrapped():
         return func(*args, **kwargs)
     return wrapped      
#==============================================================================
from functools import wraps

def pt(rho, subsystem='A'):
    n_dim = np.ndim(rho)
    #if it's an array of density matrices
    if(n_dim == 3):
        #    
        rho_rshp = rho.reshape([len(rho), 2, 2, 2, 2])
        if(subsystem == 'A'):
            rho_partial = np.einsum('aijik->ajk', rho_rshp)
        else:
            rho_partial = np.einsum('aijkj->aik', rho_rshp)        
    #If it is one dm
    elif(n_dim == 2):
        # reshape s.t. rho_rshp[i,j,k,l] is the element associated to
        # |i>|j><k|<l|
        rho_rshp = rho.reshape([2, 2, 2, 2])
        if(subsystem == 'A'):
            # einstein summation
            rho_partial = np.einsum('ijik->jk', rho_rshp)
        else:
            rho_partial = np.einsum('ijkj->ik', rho_rshp)        
    return np.array(rho_partial);









# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:55:40 2018

@author: Irtaza

psim -> measurements
"""
import numpy as np
import scipy.optimize as op
from scipy.stats import unitary_group as u
     
def measurements(state_vector, measurements, n = [1,0,0], system = 'A'):#, statmode='off', truth='off'):
     """simulates projective measurements on a subsystem.

	Arguments
	---------
	state_vector:
		 a list of state vectors

	meaasurements: int
		number of measurements to perform

	n: list<int>
		Encode the mesurement hermitian operator 
		[a, b, c] -> O = aX + bY + cZ

	system: <str>
		'A' or 'B' on which system do we perform the measurement


	Output
	------
	

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
         for i in xrange(measurements):
             prob_item = np.array([p_vec[0][j], p_vec[1][j]]).reshape((2,))
             outcome = np.random.choice([0,1], p = prob_item)
             results_sim[j][outcome] += 1
         expecs[j] += float((results_sim[j][0] - results_sim[j][1])) / measurements

     results_sim = np.concatenate((results_sim, expecs[:,None]), axis=1)
     return results_sim
     


def grid_lambda_gen(res):
    """ Generates lambda values s.t. Von Neuman entropies lie on a regularly spaced grid 
    VNentropy = -(1 - lambda) log(1 - lambda) - lambda log(lambda)
    """
    lambdaas = np.zeros((res))
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

def stochastic_lambda_gen(items):
    """ Generates lambda values s.t. s.t. Von Neuman entropies have an uniform distribution
    """
    la = np.zeros((items))  # empty array that will be filled with lambda values
    counter = 0
    for i in np.log(2)*np.random.random(items): # randomly draw from [0, ln2)
        def s(arg, vne = i):  # define vn entropy function
            yum  = -(1-arg)*np.log(1-arg) -arg*np.log(arg) - vne
            yum = np.nan_to_num(yum)   # make sure no non number is returned
            return yum
        ent_val = (op.newton(s, 0.000001))   # solve vn entropy function using newton rhapson
        if ent_val < 0:
            ent_val = 0
        la[counter] += ent_val  # reverse engineering lambdas from linear vnes by solving non-linear eqn
        counter += 1
    return la

def sep_ent_deterministic(items, lamb = 0.5):
    "generating resolution^2 seps or max ents using unitary transformation U_a [x] U_b"
    

    a = np.array([1,0], dtype = 'complex128').reshape((2,1))    # define fixed basis for A and B
    a = np.array([a,]*items)                                    # create copies for vectorized generation
    b = np.array([0, 1], dtype = 'complex128').reshape((2,1)) 
    b = np.array([b,]*items)
    
    basis1 = np.kron(a,b)      # take the tensor product to create bipartite basis
    basis2 = np.kron(b,a)
    
    # Ua [x] Ub using scipy.stats.unitary_group random matrix generator
    
    ua = u.rvs(2, size=items) # random unitaries from group U(2)
    detua = np.sqrt(np.linalg.det(ua))    # normalization by determinant to get unitaries from group SU(2) corresponding to spinor rotations
    ua = ua / np.repeat(detua,4).reshape(np.shape(ua))
    
    ub = u.rvs(2, size=items)
    detub = np.sqrt(np.linalg.det(ub))    
    ub = ub / np.repeat(detub,4).reshape(np.shape(ub))
    
    unitaries = np.kron(ua, ub) 
    state_vectors = np.matmul(unitaries, np.array(np.sqrt(lamb)*basis1 + np.sqrt(1-lamb)*basis2))
    
    return state_vectors




def dataset(state_vectors, measurements):
    """ Generate data_set (both input and ouputs) based on a list of state vectors.
    data_set[i] = (X_A, Y_A, Z_A, X_B, Y_B, Z_B, E) """
    x1 = measurements(state_vectors, measurements, [1,0,0], 'A')
    y1 = measurements(state_vectors, measurements, [0,1,0], 'A')
    z1 = measurements(state_vectors, measurements, [0,0,1], 'A')
    
    x2 = measurements(state_vectors, measurements, [1,0,0], 'B')
    y2 = measurements(state_vectors, measurements, [0,1,0], 'B')
    z2 = measurements(state_vectors, measurements, [0,0,1], 'B')   # all 6 measurements on psi_AB
    
    ents = vn_ent(pt(density_matrix(state_vectors))) #vnes
    ents = ents.reshape((len(state_vectors), 1))
    ents = np.pad(ents, (0,2), mode = 'constant')[:-2] # add zeros to make shape compatible with measurements x1,x2...
    
    data_set0 = np.stack((x1,y1,z1,x2,y2,z2,ents), axis=1)
    #data_set1 = np.stack((x1[1],y1[1],z1[1],x2[1],y2[1],z2[1]), axis=1)
    #data_set = np.stack((data_set0, data_set1))
    
    return data_set0
    




if __name__ == '__main__':
	# For sake of convenience some testing are put her
	# should be moved somewhere else
	test_creating = False
	test_loading = False

	if(test_creating):
	# script to generate data
	    measurements = 
		entropy_spacing = 1000000  # stepping in entropy values for which we generate states deterministically
		resolution = 1  # squared this is number of states with specific entropy values. 
		state_vectors = np.zeros((entropy_spacing*resolution*resolution, 4, 1), dtype = 'complex128')
		lambdas = stochastic_lambda_gen(entropy_spacing)
		vecs = [sep_ent_deterministic(resolution, lamb=l)[0] for l in lambdas]
		np.random.shuffle(state_vectors)  # shuffle to discard the order of creation of these states

		for measurement in measurements:
		    data_set = dataset(state_vectors, measurement) 
		  
		    with file(('dataset5_%smeas.txt' % measurement), 'w+') as outfile:
		        # Writing a header here for the sake of readability
		        # Any line starting with "#" will be ignored by numpy.loadtxt
		        outfile.write('# Array shape: {0}\n'.format(data_set.shape))
		        outfile.write('# Number of measuremets: {0} \n'.format(measurement) )
		        outfile.write('# Number of state vectors with particular entropy value: {0} \n'.format(resolution*resolution))
		        outfile.write('# VN Entropy spacing from 0 to 0.5: {0} \n'.format(entropy_spacing))
		        outfile.write('# Third column calculates expectation value of sigma using bin_dist. Extra zeros for vne for numpy array compatibility \n')
		    
		        # Iterating through a ndimensional array produces slices along
		        # the last axis. This is equivalent to data[i,:,:] in this case
		        for data_slice in data_set:
		    
		            # The formatting string indicates that I'm writing out
		            # fmt type to make sure the right decimal accuracy is preserved
		            np.savetxt(outfile, data_slice, fmt='%5.12g') #'%-7.2f' '%.4e'
		    
		            # Writing out a break to indicate different slices...
		            outfile.write('# New vector with column format (xa, ya, za, xb, yb, zb, vne) \n')

	with file('states5.txt', 'w+') as outfile:
	    outfile.write('# State vectors for dataset3 shaped: {0}\n'.format(state_vectors.shape))
	    
	    for state in state_vectors:
	        
	        np.savetxt(outfile, state, fmt = '%5.12g')
	        outfile.write('# State vector \n')
	



    
    if(test_loading):
	# script to generate data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    