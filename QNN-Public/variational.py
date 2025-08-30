# author: Rohan Ramkumar
# date: 7.24.24
# variational part of the QNN
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector

# creates the variational part of the quantum neural network, given a depth and the number of qubits
# returns the circuit along with all the parameters
def createVariationalCircuit(numQubits, depth) -> tuple[QuantumCircuit, list[ParameterVector]]:
    reg = QuantumRegister(numQubits)
    qc = QuantumCircuit(reg)
    params = [None] * (depth + 1) # initialize empty parameter array
    for d in range(0, depth+1):
        tempCirc, params[d] = createParameterizedTransform(numQubits, d)
        qc.compose(tempCirc, inplace = True)
        if d < depth: # add entanglement layer between every two variational layers
            for i in range(numQubits-1):
                for j in range(i+1, numQubits):
                    qc.cx(i,j)
        #qc.barrier()
    return [qc, params]

# creates a set of parameterized transforms for each depth, and returns the parameters along with the circuit
def createParameterizedTransform(numQubits, currentDepth) -> tuple[QuantumCircuit, ParameterVector]:
    qc = QuantumCircuit(numQubits)
    params = ParameterVector("θ"+str(currentDepth), numQubits)  
    for i in range(numQubits):
        qc.ry(params[i], i)
    return [qc, params]

# print(createVariationalCircuit(10, 5)[0])

