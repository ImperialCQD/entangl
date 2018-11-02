# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

@author: MIUK

Generate datasets for ANN training testing.

TODO: Ensure compatibility with ANN scripts
TODO: Look at shape of Nuke's input file

"""
from ..utility import bipartite

def gen_bipartite_dataset(nb_train, nb_test, nb_m, e_min = 0, e_max = np.log(2),
                          systems = ['A', 'B'], check_entanglement = False):
    """ Generate data_set of measures based on random states of uniformly 
    distributed entropies
    data_set[i] = (X_A, Y_A, Z_A, X_B, Y_B, Z_B, E, ) 

	Arguments
	---------
		nb_training: int
			how many training examples do we want
		nb_testing: int
            how many testing examples do we want
         nb_m: int
			How many measurements are we doing
         e_min: float
             Min entropy
         e_max: float
             Max entropy
         check_entanglement:
            Verify that entanglement required is the same as the one obtained

	Output
	------
        data_set: (data_train, data_test, infos)
            data_train: (nb_train, ) 
            data_test: (nb_test, )
            info_fileds: str 
                Provides information about the columns and parameters used to 
                generate the data
    """
    info = []
    nb_total = nb_train + nb_test
    ent = np.random.uniform(e_min, e_max, nb_total)
    lambd = bipartite.ent_to_lambda(ent)
    states = bipartite.rdm_states_from_lambda(lambd)
    if check_entanglement:
        ent_final = bipartite.entangl_of_states(states)
        assert np.allclose(ent_final, ent), "Entanglement produced don't match"
    x_A = measurements(states, nb_m, [1,0,0], 'A')
    y_A = measurements(states, nb_m, [0,1,0], 'A')
    z_A = measurements(states, nb_m, [0,0,1], 'A')
    x_B = measurements(states, nb_m, [1,0,0], 'B')
    y_B = measurements(states, nb_m, [0,1,0], 'B')
    z_B = measurements(states, nb_m, [0,0,1], 'B')
    
    data_set = np.stack((x_A, y_A, z_A, x_B, y_B, z_B, ent, states), axis=1)
    i = np.random.shuffle(np.arange(nb_total)) 
    i_train = i[:nb_train]
    i_test = i[nb_train:nb_total]
    info = []
    info.append("# Array shape: {0}".format(data_set.shape))
    info.append("# Number of measuremets: {0} ".format(nb_m))
    info.append("# Ent min/max {0}/{1}".format(e_min, e_max))
    info.append("# Columns: (0-2) X, Y, Z measurements on A")
    info.append("#(3-5) X, Y, Z measurements on B")
    info.append("#(6-10) amplitudes of states (in the computational basis)")

    return data_set[i_train, :], data_set[i_testing, :], info

def write_data_set(data_set, file_name, folder_name=None):
    """ Write a data_set as a csv file
    Input
    -----
        data_set
        
    Output
    ------
        csv file, with some comments
        
    """
    data_train, data_test, info = data_set
    
    with file(file_name, 'w+') as outfile:
        outfile.write(infos)
        for data_slice in data_set:
            np.savetxt(outfile, data_slice, fmt='%5.12g') #'%-7.2f' '%.4e'
            #outfile.write('# New vector with column format (xa, ya, za, xb, yb, zb, vne) \n')


def load_data_set(data_set, file_name, folder_name=None):
    """ Write a data_set as a csv file
    Input
    -----
    
    Output
    ------
        data_set: (data_train, data_test, infos)
            With the same characteristics as the output of 
    """
    
    

def extend_dataset_symmetries(dataset):
    """ Create an extended dataset using the symmetries of the system:
    
    """
    pass

if __name__ == '__main__':
	test_creating = False
	test_loading = False

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