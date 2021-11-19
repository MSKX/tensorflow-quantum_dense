# tensorflow-quantum_dense
The QuantumDense Tensorflow layer 

This repository, is an early attempt to integrate Tensorflow with IBM Qiskit in an effort to implement a parameterised Quantum Neural layer within a classical Neural Network  structure.

From a bird’s eye view, the behaviour of a quantum layer is not very different from that of a classical one. In a Tensorflow Sequential neural network structure, during the forward phase, the previous layer drives the quantum layer, and the weighted outputs are responsible to set the initial states of the quantum register. The mathematical transformations that perform the integration and similarly drive the output of the layer, reflect the probability of the register to collapse to one of the possible states. The QuantumDense layer can be used to drive forward the next layer in the neural network. 

The QuantumDense layer module inherits from the tensorflow.keras.layers.Layer overriding all necessary functionality to implement a quantum qiskit circuit that is able to calibrate its qubits depending on the input it receives. Because it inherits from a Tensorflow layer structure it can be used as any of the available layers in the framework. The output of the layer is a tensor with as many constituents as the qubits used. 

The layer can be executed either in simulated mode or can be sent to IBM Quantum for real-hardware execution, in the later case the layer will automatically choose the best Quantum node to execute in. Both mini constant gradient or parameter shift gradient updates can be used during the optimisation phase.

Please refer to the relevant developer community article for examples of usage at Refinitiv - an LSEG business. https://developers.refinitiv.com/en