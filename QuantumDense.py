import datetime
import numpy as np
import qiskit
import tensorflow as tf

from qiskit import transpile, assemble, QuantumRegister, QuantumCircuit
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.job import job_monitor
from qiskit.tools import backend_monitor
from tensorflow.keras.layers import Layer
from dataclasses import dataclass


@dataclass(frozen=True)
class QiskitCircuitModuleExceptionData:
    data: str


class QiskitCircuitModuleException(Exception):
    def __init__(self, exception_details):
        self.details = exception_details

    def to_string(self):
        return self.details.data


class QiskitCircuitModule:
    def __init__(self, qubits=3, instructions=None, shots=10):
        self.qubit_num = qubits
        self.instructions = instructions
        if not self.instructions:
            self.instructions = self.null_circuit(self.qubit_num)

        self.probabilities = tf.constant([[0.5] * self.qubit_num])
        self.phase_probabilities = tf.constant([1] * self.qubit_num)

        self.layer = self.superposition_qubits(self.probabilities, self.phase_probabilities)
        self.layer.append(self.instructions, range(self.qubit_num))
        self.layer.measure_all()

        self.backend = qiskit.Aer.get_backend('aer_simulator')

        self.shots = shots

    def p_to_angle(self, p):
        try:
            angle = 2 * np.arccos(np.sqrt(p))
        except Exception as e:
            raise QiskitCircuitModuleException(
                QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'p_to_angle',
                                         'message': '{e.message}'"""})))
        return angle

    def superposition_qubits(self, probabilities: tf.Tensor, phases: tf.Tensor):
        try:
            layer = qiskit.QuantumCircuit(self.qubit_num)
            reshaped_probabilities = tf.reshape(probabilities, [self.qubit_num])
            reshaped_phases = tf.reshape(phases, [self.qubit_num])
            static_probabilities = tf.get_static_value(reshaped_probabilities[:])
            static_phases = tf.get_static_value(reshaped_phases[:])

            for ix, p in enumerate(static_probabilities):
                p = np.abs(p)
                theta = self.p_to_angle(p)
                phi = self.p_to_angle(static_phases[ix])
                layer.u(theta, phi, 0, ix)
        except Exception as e:
            raise QiskitCircuitModuleException(
                QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'superposition_qubits',
                                         'message': '{e.message}'"""})))
        return layer

    def quantum_execute(self, probabilities, phases):
        try:
            self.layer = self.superposition_qubits(probabilities, phases)
            self.layer.append(self.instructions, range(self.qubit_num))
            self.layer.measure_all()

            transpiled_circuit = transpile(self.layer, self.backend)
            quantum_job_object = assemble(transpiled_circuit, shots=self.shots)
            quantum_job = self.backend.run(quantum_job_object)
            job_monitor(quantum_job)
            result = quantum_job.result().get_counts()

            qubit_set_probabilities = self.calculate_qubit_set_probabilities(result)
        except Exception as e:
            raise QiskitCircuitModuleException(
                QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'quantum_execute',
                                         'message': '{e.message}'"""})))
        return qubit_set_probabilities

    def calculate_qubit_set_probabilities(self, quantum_job_result):
        try:
            qubit_set_probabilities = [0] * self.qubit_num
            for state_result, count in quantum_job_result.items():
                for ix, q in enumerate(state_result):
                    if q == '1':
                        qubit_set_probabilities[ix] += count
            sum_counts = sum(qubit_set_probabilities)
            if not sum_counts == 0:
                qubit_set_probabilities = [i/sum_counts for i in qubit_set_probabilities]
        except Exception as e:
            raise QiskitCircuitModuleException(
                QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'calculate_qubit_set_probabilities',
                                         'message': '{e.message}'"""})))
        return qubit_set_probabilities

    def null_circuit(self, qubits):
        try:
            gate_register = QuantumRegister(qubits, 'q')
            gate_circuit = QuantumCircuit(gate_register, name='sub_circuit')
            gate_instructions = gate_circuit.to_instruction()
        except Exception as e:
            raise QiskitCircuitModuleException(
                QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'null_circuit',
                                         'message': '{e.message}'"""})))
        return gate_instructions


class QuantumLayer(Layer):
    def __init__(self, qubits=6, instructions=None, shots=10):
        super(QuantumLayer, self).__init__()
        self.qubits = qubits
        self.instructions = instructions
        self.tensor_history = []

        self.shots = shots
        self.circuit = QiskitCircuitModule(self.qubits,
                                           instructions=self.instructions,
                                           shots=self.shots)

    def build(self, input_shape):
        kernel_p_initialisation = tf.random_normal_initializer()
        self.kernel_p = tf.Variable(name="kernel_p",
                                    initial_value=kernel_p_initialisation(shape=(input_shape[-1],
                                                                          self.qubits),
                                                                          dtype='float32'),
                                    trainable=True)

        kernel_phi_initialisation = tf.zeros_initializer()
        self.kernel_phi = tf.Variable(name="kernel_phi",
                                      initial_value=kernel_phi_initialisation(shape=(self.qubits,),
                                                                              dtype='float32'),
                                      trainable=False)

    def call(self, inputs):
        try:
            output = tf.matmul(inputs, self.kernel_p)
            qubit_output = self.circuit.quantum_execute(tf.reshape(output, [1, self.qubits]),
                                                                       self.kernel_phi)
            qubit_output = tf.reshape(tf.convert_to_tensor(qubit_output), (1, 1, self.qubits))
            output += (qubit_output - output)
        except QiskitCircuitModuleException as qex:
            raise qex
        return output


class VQNNModel(tf.keras.Model):
    def __init__(self):
        super(VQNNModel, self).__init__(name='VQNN')

        self.driver_layer = tf.keras.layers.Dense(10, activation='relu')
        self.quantum_layer = QuantumLayer(5)
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
