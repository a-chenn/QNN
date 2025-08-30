# author: Rohan Ramkumar
# date: 7.23.24
# Creates a quantum feature map given the number of qubits and an array to encode.
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math

def createFeatureMap(length, arr):
    reg = QuantumRegister(length)
    qc = QuantumCircuit(reg)
    for i in range(length):
        qc.h(i)
        qc.rz(arr[i], i)
    for i in range(length-1):
        for j in range(i+1, length):
            qc.cx(i, j)
            qc.rz((math.pi-arr[i])*(math.pi-arr[j]),j)
            qc.cx(i, j)
            qc.barrier()
    return qc

#run a random test
#length = 10
#arr = np.random.uniform(low = -1.0, high = 1.0, size = (length,))
#print(createFeatureMap(length, arr))
