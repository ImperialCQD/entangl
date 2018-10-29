# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

@author: MIUK

Generate datasets for ANN training testing.

TODO:
TODO:
TODO:
"""


def gen_dataset(states, nb_examples, nb_measures):
    """ Generate data_set (both input and ouputs) based on a list of state vectors.
    data_set[i] = (X_A, Y_A, Z_A, X_B, Y_B, Z_B, E) 

	Arguments
	---------
		nb_examples: int
			how many examples do we want to generate
		nb_measures: int
			How many measurements are doing

	Output
	------

    """
    lambdas = rdm_lambda_gen(nb_examples)

    x1 = measurements(states, nb_measures, [1,0,0], 'A')
    y1 = measurements(states, nb_measures, [0,1,0], 'A')
    z1 = measurements(states, nb_measures, [0,0,1], 'A')
    x2 = measurements(states, nb_measures, [1,0,0], 'B')
    y2 = measurements(states, nb_measures, [0,1,0], 'B')
    z2 = measurements(states, nb_measures, [0,0,1], 'B')
    
    ents = vn_ent(partial_trace(dm(state_vectors)))
    ents = ents.reshape((len(state_vectors), 1))
    ents = np.pad(ents, (0,2), mode = 'constant')[:-2] # add zeros to make shape compatible with measurements x1,x2...
    
    data_set = np.stack((x1,y1,z1,x2,y2,z2,ents), axis=1)
    
    return data_set0


if __name__ == '__main__':
	test_creating = False
	test_loading

	if(test_creating):
	# script to generate data
	    list_nb_measures = [1, 10, np.inf] #Nb of measurements 
		nb_examples = 1000000  # stepping in entropy values for which we generate states deterministically
		
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