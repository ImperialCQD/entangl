# -*- coding: utf-8 -*-
"""
Created on Sat Aug 04 22:18:20 2018

@author0: MIUK
@author1: FS

Generate datasets for ANN training testing.

TODO: Ensure compatibility with ANN scripts
TODO: Look at shape of Nuke's input file
"""
from ..utility import bipartite

def gen_dataset(nb_train, nb_test, nb_m, e_min = 0, e_max = np.log(2),
                systems = ['A', 'B'], check_e = False, extra = True, 
                info_out = False):
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
         check_e: bool
            Verify that entanglement required is the same as the one obtained
         extra: bool

	Output
	------
        data_set: (X_train, Y_train), (X_test, Y_test), info
            X_train: (nb_train, nb_features) 
            Y_train: 1D numpy-array (nb_train)
            X_train: (nb_test, nb_features) 
            Y_train: 1D numpy-array (nb_test)
            info: str (optional depends on info_out) 
                Provides information about the columns and parameters used to 
                generate the data
    """
    info = []
    nb_total = nb_train + nb_test
    ent = np.random.uniform(e_min, e_max, nb_total)
    lambd = bipartite.ent_to_lambda(ent)
    states = bipartite.rdm_states_from_lambda(lambd)
    if check_e:
        ent_final = bipartite.entangl_of_states(states)
        assert np.allclose(ent_final, ent), "Entanglement produced don't match"
    x_A = measurements(states, nb_m, [1,0,0], 'A')
    y_A = measurements(states, nb_m, [0,1,0], 'A')
    z_A = measurements(states, nb_m, [0,0,1], 'A')
    x_B = measurements(states, nb_m, [1,0,0], 'B')
    y_B = measurements(states, nb_m, [0,1,0], 'B')
    z_B = measurements(states, nb_m, [0,0,1], 'B')
    if extra:
        data_set = np.stack((ent, x_A, y_A, z_A, x_B, y_B, z_B, states), axis=1)
    else:
        data_set = np.stack((ent, x_A, y_A, z_A, x_B, y_B, z_B), axis=1)
    np.random.shuffle(data_set) 
    Y_train = data_set[:nb_train, 0]
    X_train = data_set[:nb_train, 1:]
    Y_test = data_set[nb_train:, 0]
    X_test = data_set[nb_train, 1:]
    
    if(info_out):
        info = []
        info.append("Array shape: {0}".format(data_set.shape))
        info.append("Number of measuremets: {0} ".format(nb_m))
        info.append("Ent min/max required {0}/{1}".format(e_min, e_max))
        info.append("Columns: (0-2) X, Y, Z measurements on A")
        info.append("(3-5) X, Y, Z measurements on B")
        info.append("(6-10) amplitudes of states (in the computational basis)")
        res = (X_train, Y_train), (X_Test, Y_test), info
    else:
        res = (X_train, Y_train), (X_Test, Y_test)          
    return res


def write_data_set(data_set, name_data_set, info = '', folder_name=None):
    """ Write a data_set as a csv file
    Input
    -----
        data_set = (X, Y)
        
    Output
    ------
        csv file, with some comments
        
    """
    X,Y = data_set
    data = np.stack((Y, X), axis = 1)
    file_name = folder_name + name_data 
    np.savetxt(fname=file_name, data, header=info)


def write_and_save_dataset(name_data_set, nb_train, nb_test, nb_m, e_min = 0, 
                           e_max = np.log(2), systems = ['A', 'B'], check_e = False, 
                           extra = True, folder_name = None):
    """ Generate and save a data_set"""
    train, test, info = gen_dataset(nb_train, nb_test, nb_m, e_min , e_max,
                systems , check_e, extra, info_out = True)
    name_train = name + '_train.txt'
    name_test = name + '_test.txt'
    write_data_set(train, name_train, info = info, folder_name)
    write_data_set(test, name_test, info = info, folder_name)
    

def load_data_set(name_data_set, folder_name=None, print_infos = True, 
                  split_test = None):
    """ Write a data_set as a csv file
    Input
    -----
        
        split_test
    Output
    ------
        data_set: (data_train, data_test)
            With the same characteristics as the output of 
        
    """
    data_train = np.loadtxt(file_name + '_train.txt')
    data_test = np.loadtxt(file_name + '_train.txt')
    

def extend_dataset_symmetries(dataset):
    """ Create an extended dataset using the symmetries of the system:
    
    """
    pass

if __name__ == '__main__':
    test_creating = False
    test_loading = False

    if(test_creating):
        # Testing the creation and saving of data_set
        list_nb_measures = [1, 10, np.inf] #Nb of measurements 
        nb_examples = 1000000  # stepping in entropy values for which we generate states deterministically
        data_set = gen_bipartite_dataset(nb_train, nb_test, nb_m, e_min = 0, e_max = np.log(2),
                          systems = ['A', 'B'], check_entanglement = False)
        
        #write_data_set(data_set, file_name, folder_name=None)
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