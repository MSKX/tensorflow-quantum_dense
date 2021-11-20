import datetime
import numpy as np
import pandas as pd
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
    def __init__(self, qubits=3, instructions=None, execute_on_IBMQ=False, shots=10):
        self.qubit_num = qubits
        self.instructions = instructions
        if not self.instructions:
            self.instructions = self.null_circuit(self.qubit_num)

        self.probabilities = tf.constant([[0.5] * self.qubit_num])
        self.phase_probabilities = tf.constant([1] * self.qubit_num)

        self.layer = self.superposition_qubits(self.probabilities, self.phase_probabilities)
        self.layer.append(self.instructions, range(self.qubit_num))
        self.layer.measure_all()

        if not execute_on_IBMQ:
            self.backend = qiskit.Aer.get_backend('aer_simulator')
        else:
            self.backend = self.detect_optimal_quantum_device()
        self.shots = shots

    def detect_optimal_quantum_device(self, verbose=False):
        try:
            if not qiskit.IBMQ.active_account():
                qiskit.IBMQ.load_account()

            provider = qiskit.IBMQ.get_provider()
            large_enough_devices = provider.backends(
                filters=lambda x: x.configuration().n_qubits >= self.qubit_num and not x.configuration().simulator)
            backend = least_busy(large_enough_devices)
            print("The best available quantum device to execute the layer is " + backend.name())

            if verbose:
                print(backend_monitor(backend))
        except Exception as e:
            raise QiskitCircuitModuleException(
                        QiskitCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                                  strftime("%m/%d/%Y, %H:%M:%S")}',
                                                  'function': 'detect_optimal_quantum_device',
                                                  'message': '{e.message}'"""})))
        return backend

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
    def __init__(self, qubits=6, instructions=None, execute_on_IBMQ=False, shots=10,
                 use_parameter_shift_gradient_flow=False):
        super(QuantumLayer, self).__init__()
        self.use_parameter_shift_gradient_flow = use_parameter_shift_gradient_flow

        self.qubits = qubits
        self.instructions = instructions
        self.tensor_history = []

        self.execute_on_IBMQ = execute_on_IBMQ
        self.shots = shots
        self.circuit = QiskitCircuitModule(self.qubits,
                                           instructions=self.instructions,
                                           execute_on_IBMQ=self.execute_on_IBMQ,
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
            if not self.use_parameter_shift_gradient_flow:
                output = tf.matmul(inputs, self.kernel_p)
                qubit_output = self.circuit.quantum_execute(tf.reshape(output, [1, self.qubits]), self.kernel_phi)
                qubit_output = tf.reshape(tf.convert_to_tensor(qubit_output), (1, 1, self.qubits))
                output += (qubit_output - output)
            else:
                output = self.quantum_flow(inputs)

        except QiskitCircuitModuleException as qex:
            raise qex
        return output

    @tf.custom_gradient
    def quantum_flow(self, x):
        output = tf.matmul(x, self.kernel_p)
        qubit_output = tf.reshape(tf.convert_to_tensor(self.circuit.quantum_execute(tf.reshape(output,
                                                                                               [1, self.qubits]),
                                                                                    self.kernel_phi)),
                                  (1, 1, self.qubits))

        output = qubit_output

        def grad(dy, variables=None):
            shift = np.pi / 2
            shift_right = x + np.ones(x.shape) * shift
            shift_left = x - np.ones(x.shape) * shift

            input_left = tf.matmul(shift_left, self.kernel_p)
            input_right = tf.matmul(shift_right, self.kernel_p)

            output_right = self.circuit.quantum_execute(tf.reshape(input_right, [1, self.qubits]), self.kernel_phi)
            output_left = self.circuit.quantum_execute(tf.reshape(input_left, [1, self.qubits]), self.kernel_phi)

            quantum_gradient = [output_right[i] - output_left[i] for i in range(len(output_right))]
            input_gradient = dy * quantum_gradient
            dy_input_gradient = tf.reshape(tf.matmul(input_gradient, tf.transpose(self.kernel_p)),
                                           shape=[1, 1, x.get_shape().as_list()[-1]])

            grd_w = []
            for i in range(self.qubits):
                w = self.kernel_p[:, i]
                w += dy_input_gradient
                grd_w.append(w)

            tf_grd_w = tf.convert_to_tensor(grd_w)
            tf_grd_w = tf.reshape(tf_grd_w, shape=(x.get_shape().as_list()[-1], self.qubits))

            return dy_input_gradient, [tf_grd_w]

        return output, grad


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
