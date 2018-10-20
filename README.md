# entangl

The original contribution was done by Irtaza Khalid (in the context of an Urop
project), and now followed up by Kin Lo and Timothy Liu for their Bsc project.



## Installation

`entangl` is not available through `pip` or `conda`, and must be installed
manually, by cloning the repository and adding the resulting directory to the
`PYTHONPATH` environment variable (or otherwise making the inner `floq` folder
visible to the Python search path).

The requirements are listed in the file `requirements.txt` and intended to be used 
with python 3


## Overview
entangl provides to generate and 


`utility/` folder contains the main functions: 
qm.py should contain the relevant functions to simulate quantum mechanics 
(i.e. generation of quantum states, projective masurements, entanglement 
computation)

datagen.py encompasses tools to generate training/testing data

ann.py helper for the construction of artificial neural networks

`examples/`

There are a couple of examples in the `examples/` folder, and there is more help
available in the docstrings of the code.  Try calling `help()` on classes and
functions to find out more.


`dataset/` is intended to be the storage place for small dataset used in the examples