import sys
sys.path.append('../../entangl')
from entangl.utility.bipartite import * 

test_entanglement = True
test_measurements = True
test_gen_rdm_states = True
test_ent_to_lambdas = True
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





    

