import sys
sys.path.append('../../entangl')
from entangl.data import data_bipartite as data
from entangl.utility import bipartite 
import numpy as np
import matplotlib.pylab as plt

test_creating = True
test_save = True
test_loading = True

if(test_creating):
    train, test, info = data.gen_dataset(1000, 20, np.inf, check_e = True)

if(test_creating):
    data.write_and_save_dataset('_test', 1000, 20, np.inf, states = True)

if(test_loading):
    train, test = data.load_data_set('_test', states = True)
    X_train, Y_train, states_train = train
    ent =bipartite.entangl_of_states(states_train)
    assert np.allclose(ent, Y_train), "Entropies (computed and retrieved doesn't match)"
    xA = bipartite.meas_one_sub(states_train, np.inf, [1, 0, 0], 'A')
    yA = bipartite.meas_one_sub(states_train, np.inf, [0, 1, 0], 'A')    
    zA = bipartite.meas_one_sub(states_train, np.inf, [0, 0, 1], 'A')    
    xB = bipartite.meas_one_sub(states_train, np.inf, [1, 0, 0], 'B')
    yB = bipartite.meas_one_sub(states_train, np.inf, [0, 1, 0], 'B')    
    zB = bipartite.meas_one_sub(states_train, np.inf, [0, 0, 1], 'B')    
    assert np.allclose(np.c_[xA, yA, zA, xB, yB, zB], X_train), "Measurements (computed and retrieved doesn't match)"    
    plt.hist(Y_train)
    
#data.write_and_save_dataset('perfect_10k2k', 10000, 2000, np.inf, states = True)

    
#        # Testing the creation and saving of data_set
#        list_nb_measures = [1, 10, np.inf] #Nb of measurements 
#        nb_examples = 1000000  # stepping in entropy values for which we generate states deterministically
#        data_set = gen_bipartite_dataset(nb_train, nb_test, nb_m, e_min = 0, e_max = np.log(2),
#                          systems = ['A', 'B'], check_entanglement = False)
#        
#        #write_data_set(data_set, file_name, folder_name=None)
#        vecs = [rdm_states_from_lambda(1, lamb=l)[0] for l in lambdas]
#		np.random.shuffle(state_vectors)  # shuffle to discard the order of creation of these states
#
#		for nb_m in list_nb_measures:
#		    data_set = gen_dataset(state_vectors, nb_m) 
#		  
#		    with file(('dataset5_%smeas.txt' % nb_m), 'w+') as outfile:
#		        # Writing a header here for the sake of readability
#		        # Any line starting with "#" will be ignored by numpy.loadtxt
#		        outfile.write('# Array shape: {0}\n'.format(data_set.shape))
#		        outfile.write('# Number of measuremets: {0} \n'.format(nb_m) )
#		        outfile.write('# Number of state vectors with particular entropy value: {0} \n'.format(resolution*resolution))
#		        outfile.write('# VN Entropy spacing from 0 to 0.5: {0} \n'.format(entropy_spacing))
#		        outfile.write('# Third column calculates expectation value of sigma using bin_dist. Extra zeros for vne for numpy array compatibility \n')
#		    
#		        for data_slice in data_set:
#		            np.savetxt(outfile, data_slice, fmt='%5.12g') #'%-7.2f' '%.4e'
#		            outfile.write('# New vector with column format (xa, ya, za, xb, yb, zb, vne) \n')
#
#		with file('states5.txt', 'w+') as outfile:
#		    outfile.write('# State vectors for dataset3 shaped: {0}\n'.format(state_vectors.shape))	    
#		    for state in state_vectors:
#		        np.savetxt(outfile, state, fmt = '%5.12g')
#		        outfile.write('# State vector \n')
	


