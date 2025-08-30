# author: Rohan Ramkumar
# date: 7.25.24
# Runs gradient descent algorithm on the Quantum Neural Network, using the parameter shift rule to calculate the gradient


import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
from qiskit_algorithms.gradients import *
from qiskit.primitives import * 
import variational
import featureMap
from qiskit_aer import AerSimulator

# combines feature map and variational part to create full circuit
def makeCircuit(x, var):
    n = len(x)
    reg = QuantumRegister(n)
    qc = QuantumCircuit(reg).compose(featureMap.createFeatureMap(n, x)).compose(var)
    return qc 

# gets expected value of an operator with respect to a vector
def expectedValue(op, psi):
    return np.real(psi.inner(psi.evolve(op))) # <ψ|O|ψ>, taking real part to remove floating point errors that yield a small imaginary part
def globalObservableFromState(v):
    s = ""
    for k in v:
        s += str(k)
    n = 2**len(s)
    arr = [-1]*n
    arr[int(s, 2)] = 1
    return SparsePauliOp.from_operator(np.identity(n) - np.diag(arr))

def localObservableFromState(v):
    kron = lambda li: np.kron(kron(li[:-1]), li[-1]) if len(li) > 1 else li[0]
    projZero = np.array([[1, 0], [0, 0]])
    projOne = np.array([[0, 0], [0, 1]])
    n = len(v)
    mat = np.zeros((n, n, 2, 2))
    kr = np.zeros((n, 2**n, 2**n))
    for i in range(n):
        for j in range(n):
            mat[i][j] = np.identity(2)
        mat[i][i] = projZero if v[i]==0 else projOne
        kr[i] = np.array(kron(mat[i]))
    return SparsePauliOp.from_operator(np.identity(2**n) - np.average(kr, axis = 0))

def randomizeCirc(d, n):
    RAND_RANGE = [-1., 1.]
    arr = np.zeros((d+1, n*2))
    for i in range(d+1):
        for j in range(len(params[i])):
            if j < n:
                arr[i][j] = random.uniform(RAND_RANGE[0], RAND_RANGE[1]) # instantiate the parameters to random values
            else:
                arr[i][j] = -arr[i][j-n]
    return arr 

def train(tr, obs, circ, initParams):
    gradCalcer = ReverseEstimatorGradient()
    errorCalcer = Estimator()
    THRESHOLD = -1
    MAX_ITER = 1E6
    ALPHA = 0.001
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1E-8
    params = np.repeat([np.ndarray.flatten(np.copy(initParams))], len(tr), axis = 0)
    err = 2
    errArr = []
    it = 0
    m = np.zeros((len(tr), len(params[0])))
    v = np.zeros((len(tr), len(params[0])))
    try:
        while err > THRESHOLD and it < MAX_ITER:
            it += 1
            g = np.array(gradCalcer.run(qc, obs, params).result().gradients)
            m = BETA1 * m + (1-BETA1) * g
            v = BETA2 * v + (1-BETA2) * np.square(g)
            delta = ALPHA * (m/(1-BETA1**it))/(np.sqrt(v/(1-BETA2**it)) + EPS)
            params -= delta.sum(axis = 0)
            err = np.average(errorCalcer.run(circ, obs, params).result().values)
            errArr.append(err)
            print("Error: "+ str(err))
    except KeyboardInterrupt:
        pass
    return params, errArr

n = 4
d = 5
usingImages = False
if usingImages:
    print("Loading Images")
    arr = np.loadtxt("images.txt")[:1000]
    tr = arr[:,1:]
    out = [[int(dig) for dig in format(int(num), "0"+str(n)+"b")] for num in arr[:, 0]]
else:
    tr = [[int(dig) for dig in format(i, "0"+str(n)+"b")] for i in range(2**n)] # python moment
    out = tr
training =  True
if training:
    print("Generating Observables")
    obs = [localObservableFromState(y) for y in out] # observables
    variationalCirc, params = variational.createVariationalCircuit(n, d)
    print("Generating Circuits")
    qc = [makeCircuit(x, variationalCirc) for x in tr]
    initParams = randomizeCirc(d, n) 
    print("Training")
    params, errArr = train(tr, obs, qc, initParams)
    plt.plot(errArr)
    plt.show()

    # save weights

    np.savetxt("weights.txt", params[0])
else:
    print("Running")
    weights = np.loadtxt("weights.txt")
    variationalCirc, params = variational.createVariationalCircuit(n, d)
    qc = [makeCircuit(x, variationalCirc) for x in tr]
    count = 0
    for i in range(len(qc)):
        circ = qc[i].assign_parameters(weights).measure_all(inplace = False)
        result = AerSimulator().run(circ, shots=10000, memory=True).result()
        measurements = result.get_memory()[0]
        if int(measurements, 2)==int("".join(map(str,out[i])),2):
            count+=1
    print("Success rate: "+str(count/len(out)*100) + "%")
