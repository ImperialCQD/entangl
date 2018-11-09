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
`entangl/` folder contains the core functions. it is split in `data/`
(how to generatte training / testing data) and `utility/` which contains 
functions to deal with quantum states (i.e. generation of quantum states, 
projective masurements, entanglement computation, etc..)

`dataset/` contains dataset used for training of the ANNs

`test/` contains some testing scripts

`training/` contains some training scripts
