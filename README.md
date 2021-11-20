# The QuantumDense Tensorflow layer 

This repository, is an early attempt to integrate Tensorflow with IBM Qiskit in an effort to implement a parameterised Quantum Neural layer within a classical Neural Network  structure.

From a bird’s eye view, the behaviour of a quantum layer is not very different from that of a classical one. In a Tensorflow Sequential neural network structure, during the forward phase, the previous layer drives the quantum layer, and the weighted outputs are responsible to set the initial states of the quantum register. The mathematical transformations that perform the integration and similarly drive the output of the layer, reflect the probability of the register to collapse to one of the possible states. The QuantumDense layer can be used to drive forward the next layer in the neural network. 

The QuantumDense layer module inherits from the tensorflow.keras.layers.Layer overriding all necessary functionality to implement a quantum qiskit circuit that is able to calibrate its qubits depending on the input it receives. Because it inherits from a Tensorflow layer structure it can be used as any of the available layers in the framework. The output of the layer is a tensor with as many constituents as the qubits used. 

The layer can be executed either in simulated mode or can be sent to IBM Quantum for real-hardware execution, in the later case the layer will automatically choose the best Quantum node to execute in. Both mini constant gradient or parameter shift gradient updates can be used during the optimisation phase.

Please refer to the relevant developer community article for examples of usage at Refinitiv - an LSEG business. https://developers.refinitiv.com/en

QuantumLayer class
--------------------------

Constructor accepts the following parameters:

*qubits*: Number of qubits in the register. [default=3]

*instructions*: Quantum circuit following the register. [default=None]

*execute_on_IBMQ*: If set to True, the QuantumDense layer will look for the optimal quantum device to execute the circuit. If set to False the layer will be simulated using Aer. [default=False]

*shots*: Number of times the circuit will be executed. [defaul=10]

*use_parameter_shift_gradient_flow*: If set to False the optimiser will apply small constant updates to the parameters. If set to True the full parameter shift rule will be applied resulting in a threefold increase in execution times as the circuit will need to be re-executed twice for every record in each learning epoch. [default = False]

The QuantumDense.py file contains an example of usage creating a three layer VQNN model:

```Python
class VQNNModel(tf.keras.Model):

    def __init__(self):
    
        super(VQNNModel, self).__init__(name='VQNN')

        self.driver_layer = tf.keras.layers.Dense(3, activation='relu')
        self.quantum_layer = QuantumLayer(2, execute_on_IBMQ=False, use_parameter_shift_gradient_flow=False)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor):
        try:
            x = self.driver_layer(input_tensor, training=True)
            x = self.quantum_layer(x, training=False)
            x = tf.nn.relu(x)
            x = self.output_layer(x, training=True)
        except QiskitCircuitModuleException as qex:
            print(qex)
        return x 
```
