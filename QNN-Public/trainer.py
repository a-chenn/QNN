# author: Rohan Ramkumar
# date: 7.25.24
# Runs gradient descent algorithm on the Quantum Neural Network, using the parameter shift rule to calculate the gradient


import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator

import variational
import featureMap

# combines feature map and variational part to create full circuit
def makeCircuit(x, var):
    n = len(x)
    reg = QuantumRegister(n)
    qc = QuantumCircuit(reg).compose(featureMap.createFeatureMap(n, x)).compose(var)
    return qc 

# gets expected value of an operator with respect to a vector
def expectedValue(op, psi):
    return np.real(psi.inner(psi.evolve(op))) # <ψ|O|ψ>, taking real part to remove floating point errors that yield a small imaginary part

# loss function, quadratic to improve training
def loss(op, psi):
    return 0.5 * (1 - expectedValue(op, psi))**2

# gradient of loss function using chain rule
def lossGrad(op, psi, gradExp):
    return -gradExp * (1 - expectedValue(op, psi))

# gets the operator corresponding to an observable with positive eigenvalue of 1 for v and negative eigenvalue of -1 for any other pure state
def observableFromState(v):
    s = ""
    for k in v:
        s += str(k)
    n = 2**len(s)
    arr = [-1]*n
    arr[int(s, 2)] = 1

    return Operator(np.diag(arr))

# applys the gradient shift rule on one parameters to find the derivative of the expected value with respect to the specific parameter p
def shiftOneParameter(qc, dic, p, obs):
    dic[p] += math.pi/2
    forward = Statevector(qc.assign_parameters(dic)) # this step is really slow, need to optimize
    dic[p] -= math.pi
    backward = Statevector(qc.assign_parameters(dic)) # this step is really slow, need to optimize
    dic[p] += math.pi/2
    
    return (expectedValue(obs, forward) - expectedValue(obs, backward))/2

# calculate all the gradients with respect to all the parameters, averages it for all the training cases, and returns an array of the partial derivatives
def calcGradients(var, dic, params, tr, obs):
    grads = np.zeros((len(params), len(params[0])))
    for i in range(len(tr)):
        qc = makeCircuit(tr[i], var)
        psi = Statevector(qc.assign_parameters(dic))
        for d in range(len(params)):
            for j in range(len(params[d])):
                grads[d][j] += lossGrad(obs[i], psi, shiftOneParameter(qc, dic, params[d][j], obs[i]))
    return grads/len(tr)

# averaged loss function
def error(var, tr, dic, obs):
    err = 0
    for i in range(len(tr)):
        qc = makeCircuit(tr[i], var).assign_parameters(dic)
        err += loss(obs[i], Statevector(qc))
    return err/len(tr)

# repeats the gradient descent method using ADAM optimizer until the program terminates
def train(var, tr, dic, params, obs):
    THRESHOLD = 0.5
    MAX_ITER = 1E6
    ALPHA = 0.001
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1E-8
    err = 2
    errArr = []
    it = 0
    m = np.zeros((len(params), len(params[0])))
    v = np.zeros((len(params), len(params[0])))
    try:
        while err > THRESHOLD and it < MAX_ITER:
            it += 1
            err = 0
            g = calcGradients(var, dic, params, tr, obs)
            m = BETA1 * m + (1-BETA1) * g
            v = BETA2 * v + (1-BETA2) * np.square(g)
            delta = ALPHA * (m/(1-BETA1**it))/(np.sqrt(v/(1-BETA2**it)) + EPS)
            for d in range(len(params)):
                for j in range(len(params[d])):
                    dic[params[d][j]] -= delta[d][j]
            err = error(var, tr, dic, obs)
            errArr.append(err)
            print("Error: "+ str(err))
    except KeyboardInterrupt:
        pass
    return errArr

RAND_RANGE = [-1., 1.]
n = 5 # number of qubits
d = 5 # depth of variational circuit

# initialize input array
tr = [[int(dig) for dig in format(i, "0"+str(n)+"b")] for i in range(2**n)][:10] # python moment

variationalCirc, params = variational.createVariationalCircuit(n, d)

dic={}
for i in range(d+1):
    for p in params[i]:
        dic[p] = random.uniform(RAND_RANGE[0], RAND_RANGE[1]) # instantiate the parameters to random values

out = tr

obs = [observableFromState(y) for y in out]
errArr = train(variationalCirc, tr, dic, params, obs)
# plot error
plt.plot(errArr)
plt.show()

# save weights
weights = []
for i in range(d+1):
    for p in params[i]:
        weights.append(dic[p])
with open('weights.txt', 'w') as filehandle:
    json.dump(weights, filehandle)
