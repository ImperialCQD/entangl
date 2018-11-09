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
import numpy as np

def gen_dataset(nb_train, nb_test, nb_m, e_min = 0, e_max = np.log(2),
                subsyst = ['A', 'B'], check_e = False, states = False):
    """ Generate data_set of measures based on random states of uniformly 
    distributed entropies

	Arguments
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
         states: bool
            should we return the underlying states used
         info
         
	Output
	------
        res: (train, test, info)
            train/test: (X, Y, states(optional))
            info: str 
                Provides information about the columns and parameters used to 
                generate the data
    """
    info = []
    info.append("Number of measuremets: {0} ".format(nb_m))
    info.append("Ent min/max required {0}/{1}".format(e_min, e_max))
    nb_total = nb_train + nb_test
    ent = np.random.uniform(e_min, e_max, nb_total)
    lambd = bipartite.ent_to_lambda(ent)
    st = bipartite.rdm_states_from_lambda(lambd)
    if check_e:
        ent_final = bipartite.entangl_of_states(st)
        assert np.allclose(ent_final, ent), "Entanglement produced don't match"
    
    X = np.zeros((nb_total, 3 * len(subsyst)))    
    for i, ss in enumerate(subsyst):
        info.append("X[{0}:{1}] X, Y, Z measurements on subsyst {2}".format(3*i, 3*i+2, ss))
        X[:, (3 * i)] = bipartite.meas_one_sub(st, nb_m, [1,0,0], ss)
        X[:, (3 * i + 1)] = bipartite.meas_one_sub(st, nb_m, [0,1,0], ss)
        X[:, (3 * i + 2)] = bipartite.meas_one_sub(st, nb_m, [0,0,1], ss)
        
    index = np.arange(nb_total)
    np.random.shuffle(index)
    index_train = index[:nb_train]
    index_test  = index[(nb_train+1):nb_total]   
        
    if states:
        train = (X[index_train, :], ent[index_train], st[index_train, :]) 
        test = (X[index_test, :], ent[index_test], st[index_test, :]) 
    else:
        train = (X[index_train, :], ent[index_train]) 
        test = (X[index_test, :], ent[index_test]) 
         
    return train, test, "\n".join(info)

def write_data_set(data_set, name_data, info = '', name_folder=''):
    """ Write a data_set as a txt file. if data_set contains the underlying 
    states they will be written in a separate file
    
    Input
    -----
        data_set: tuple (X, Y, states(optional))
            X: 2d-array
            Y: 1d-array
            states:
        name_data:
            name of the file to write - without any extension
        info: str
            meta-info about teh data
        
        folder_name: str
            name of the folder where the files are going to be written
    Output
    ------
        Write one (two) file(s), with some comments. 
            main: Y is the first column, X the others
    """
    X = data_set[0]
    Y = data_set[1]
    states = data_set[2] if len(data_set) == 3 else None
    np.savetxt(fname=name_folder + name_data + '.txt', X=np.c_[Y, X], 
               header=info)
    
    if states is not None:
        np.savetxt(fname=name_folder + name_data + '_states.txt', 
                   X=states, header=info)

def write_and_save_dataset(name_data, nb_train, nb_test, nb_m, e_min=0, 
                           e_max=np.log(2), subsyst=['A', 'B'], check_e=False, 
                           states=True, name_folder=''):
    """ Generate AND save a data_set
    Input
    -----
        name_data: str
        nb_train : int
        nb_test : int
        nb_m: int 
        e_min: float
        e_max: float
        subsyst=['A', 'B'], 
        check_e=False, 
        extra=True
        name_folder=None
        
    Output
    ------
        csv file, with some comments
        
    """
    train, test, info = gen_dataset(nb_train, nb_test, nb_m, e_min, e_max,
                                    subsyst, check_e, states)
    name_train = name_data + '_train'
    name_test = name_data + '_test'
    write_data_set(train, name_train, info , name_folder)
    write_data_set(test, name_test, info, name_folder)
    

def load_data_set(name_data, name_folder='', print_info=True, states=False):
    """ load a data_set with a given name and a given folder. Retrieve two files
        train and test. Split it in X,Y and extra (optional)
    Input
    -----
        name_data: str
            Name of the data e.g. '10meas_perfect'
        folder_name: str
            location of the folder in which to retrieve name_data_train.txt and
            name_data_test.txt
        print_info: bool
            print comments at the beginning of the txt file 
        states: bool
            
    Output
    ------
        data_set: (X_train, Y_train), (X_test, Y_test)
            X_train: (nb_train, nb_features) 
            Y_train: 1D numpy-array (nb_train)
            X_train: (nb_test, nb_features) 
            Y_train: 1D numpy-array (nb_test)
        
    TODO: make it more flexible i.e meta-info in comments - parse it and use it
    """
    X_train, Y_train= load_one_data_set(name_folder + name_data + '_train.txt')
    X_test, Y_test = load_one_data_set(name_folder + name_data + '_test.txt')
    if(states):
        #deal witth complex values
        states_train = np.loadtxt(
                name_folder + name_data + '_train_states.txt', dtype=complex, 
                converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
        states_test = np.loadtxt(name_folder + name_data + '_test_states.txt', 
                dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})        
        return (X_train, Y_train, states_train), (X_test, Y_test, states_test)
    else:
        return (X_train, Y_train), (X_test, Y_test)

def load_one_data_set(name_file, print_info=True):
    """ load one data set
    #TODO: implement print_info
    """
    data = np.loadtxt(name_file)
    if(print_info):
        pass
    return (data[:, 1:], data[:, 0])
    


def extend_dataset_symmetries(dataset):
    """ Create an extended dataset using the symmetries of the system:
    TODO Implement
    """
    pass

