import numpy as np
import math
import random
import pandas as pd

# random between a, b
def rand(a, b):
    return (b-a)*random.random() + a
# optional activation 
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
def tanh(x):
    return math.tanh(x)
def relu(x):
    return max(x, 0)
activateFunc = {
    'sigmoid':sigmoid,
    'tanh':tanh,
    'relu':relu
}
def derSigmoid(out):
    return out * (1 - out)
def derTanh(out):
    return 1 - out**2
def derRelu(out):
    return 1 if (out > 0) else 0
derActivateFunc = {
    'sigmoid':derSigmoid,
    'tanh':derTanh,
    'relu':derRelu
}
# activation function
def activate(x, type):
    if (type not in activateFunc.keys()):
        raise RuntimeError('activate type not found')
    return activateFunc[type](x)

# partial derivative of the activation function
# note, the argu out is activate(x)
def derActivate(out, type):
    if (type not in derActivateFunc.keys()):
        raise RuntimeError('activate type not found')
    return derActivateFunc[type](out)

np.random.seed(1)

class Node:
    def __init__(self, previousSize, activate, bias):
        # each node should record the layersize of previous layer
        # and maintain an array of weight flow into the node
        # which is initialed with random value
        #self.weight = [random.random() for i in range(previousSize)]
        self.weight = [rand(-0.5, 0.5) for i in range(previousSize)]
        #self.weight = [0.15]*8
        self.change = [0.0] * previousSize
        self.activate = activate
        self.bias = bias

    def receive(self, previousData):
        # called when previous data flow into
        # the number of input data should be previousSize
        # sum weight * input, and add bias                                  # what is this bias used for, why it won't update
        # activate sum as output
        # output should be saved for back-propagation use
        inputSum = sum([i * j for i, j in zip(self.weight, previousData)], self.bias)
        self.output = activate(inputSum, self.activate)        
        return self.output
    
    def updateWeight(self, penalty, learningRate=0.5):
        #self.weight = self.weight - learningRate * penalty         # the same
        self.weight = [w - learningRate * penalty for w in self.weight]

    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight

class Layer:
    def __init__(self, previousSize, ownSize, activate, bias):
        # each layer should record his previous layer size and own size
        # should have a bias
        # should maintain a set of node within this layer
        self.previousSize = previousSize
        self.ownSize = ownSize
        self.activate = activate
        self.bias = bias
        self.nodeSet = [Node(previousSize, self.activate, self.bias) for i in range(ownSize)]
        self.output = [0.0] * ownSize

    def receive(self, previousData):
        # reveive previous layer data,
        # pass to each node and get output which will be flow into next or output
        self.output = [node.receive(previousData) for node in self.nodeSet]
        return self.output

    def getError(self, currentDelta):
        # error is uesd for computing delta of previous layer when back-propagation
        # delta = error * partial derivative
        # each error is computed by sum (each currentDelta * each weight who flow into this Node that own the currentDelta) 
        # and each error used for privous layer, that node flow into each Node in this layer
        def oneError(currentDelta, i):
            return sum([delta * node.weight[i] for delta, node in zip(currentDelta, self.nodeSet)])
        return [oneError(currentDelta, previousNode) for previousNode in range(self.previousSize)]

    def getDeltaInHidden(self, lastError):
        # using next layer error to compute this layer delta
        self.delta = [error * derActivate(hout, self.activate) 
                        for hout, error in zip(self.output, lastError)]
        return self.delta

    def updateWeight(self, penalty, learningRate=0.5):
        # update weight of each node 
        # after computing every delta in every layer except the input(though no this layer)
        for p, node in zip(penalty, self.nodeSet):
            node.updateWeight(p, learningRate)
    
    def getAllWeight(self):
        return [node.getWeight() for node in self.nodeSet]

class BPNN:
    def __init__(self, inputSize, outputSize, learingRate=0.05):
        # BPNN need inputSize for create next layer
        # outputSize can be set in addOutputLayer also
        self.hiddenList = []
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.learingRate = learingRate

        self.lastLayerSize = inputSize
        self.finishAdd = False

    def addLayer(self, thisSize, activate, bias=random.random()):
        # add layer containing thisSize node with activate
        # random bias
        if (self.finishAdd == True):
            raise RuntimeError("add layer failed, already add output layer")
        self.hiddenList.append(Layer(self.lastLayerSize, thisSize, activate, bias))
        self.lastLayerSize = thisSize

    def addOutPutLayer(self, outActivate, outBias):
        # add output layer 
        # finsh add
        self.outputLayer = Layer(self.lastLayerSize, self.outputSize, outActivate, outBias)
        self.outActivate = outActivate
        self.finishAdd = True

    def forward(self, inputData):
        # forwarding,
        # and each layer output will be record inner layer
        if (len(inputData) != self.inputSize):
            raise RuntimeError('input size matter failed')
        # hidden layer forward
        lastOutput = inputData
        for layer in self.hiddenList:
            lastOutput = layer.receive(lastOutput)
            #print(layer.getAllWeight())
        # forward output layer, record output in BPNN
        # for backPropagation use
        self.output = self.outputLayer.receive(lastOutput)
        return self.output

    def backPropagation(self, target):
        # back propagation,
        # delta is used to update weight in each layer
            # new_weight = old_weight - delta * output, for each node
        # error is used to compute previous layer delta
            # previous_layer_delta = error * partial derivative
        if (len(target) != self.outputSize):
            raise RuntimeError('input size matter failed')

        #output_delta = self.outputLayer.getDeltaInOutput(target)
        output_delta = [-(t - o) * derActivate(o, self.outActivate) for t, o in zip(target, self.output)]
        last_error = self.outputLayer.getError(output_delta)
        self.outputLayer.updateWeight(output_delta, self.learingRate)
        for layer in self.hiddenList[::-1]:
            last_detla = layer.getDeltaInHidden(last_error)
            last_error = layer.getError(last_detla)
            layer.updateWeight(last_detla, self.learingRate)

    def train(self, inputData, targetData, times, learningRate, display=False, display_time=10):
        # inputdata is a two-dimensional array
        # one line of which is one sample
        # each col is an attribute.
        # the same as targetData
        self.learingRate = learningRate
        if (len(inputData[0]) != self.inputSize):
            raise RuntimeError("input data not match")
        if (len(targetData[0]) != self.outputSize):
            raise RuntimeError("target data not match")
        if (len(inputData) != len(targetData)):
            raise RuntimeError("input target not match")

        for time in range(times):
            for inputd, target in zip(inputData, targetData):
                o = self.forward(inputd)
                self.backPropagation(target)
            if (time % (times / display_time) == 0 and display):
                print(self.test(inputData, targetData))
    
    def loss(self, out, target):
        return sum([0.5* (o-t)**2 for o, t in zip(out, target)])
        
    def test(self, inputData, targetData):
        total_loss = 0
        for inputd, target in zip(inputData, targetData):
            out = self.forward(inputd)
            total_loss += self.loss(out, target)
        return total_loss / len(inputData)
    
    def predict(self, inputData, targetData):
        for inputd, target in zip(inputData,targetData):
            print(inputd, ' --> ', self.forward(inputd), ' expect: ', target)

if __name__ == '__main__':

'''
    inputdata = [[0.05,0.1]]
    targetdata = [[0.01, 0.99]]

    inputSize = len(inputdata[0])
    outputSize = len(targetdata[0])
    bias = [0.35, 0.60]

    print("begin BPNN")

    bpnn = BPNN(inputSize, outputSize, learingRate=0.05)
    bpnn.addLayer(thisSize=2, activate='sigmoid', bias=bias[0])
    bpnn.addOutPutLayer(outActivate='sigmoid', outBias=bias[1])
    bpnn.train(inputdata, targetdata, times=100000, learningRate=0.05, display=True, display_time=10)

    print(bpnn.test(inputdata, targetdata))
    print(bpnn.predict(inputdata, targetdata))
'''
'''
    inputdata = [[0.7, 0.2, 0.18],
                 [0.2, 0.82, 0.46],
                 [0.02, 0.3, 0.73],
                 [0.56, 0.1, 0.49],
                ]
    targetdata = [[0.5, 0.12],
                  [0.8, 0.88],
                  [0.7, 0.19],
                  [0.4, 0.52]
                ]

    inputSize = len(inputdata[0])
    outputSize = len(targetdata[0])
    bias = [0.35, 0.8, 0.60]

    print("begin BPNN")

    bpnn = BPNN(inputSize, outputSize, learingRate=0.05)
    bpnn.addLayer(thisSize=12, activate='tanh', bias=bias[0])
    bpnn.addOutPutLayer(outActivate='tanh', outBias=bias[2])
    bpnn.train(inputdata, targetdata, times=100000, learningRate=0.05, display=True, display_time=10)

    print(bpnn.test(inputdata, targetdata))
    print(bpnn.predict(inputdata, targetdata))
    
'''
'''   
    # read dataset
    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal width', 'class']
    df = pd.read_csv('iris.csv', names=headers)
    # binarization classification
    dummy = pd.get_dummies(df['class'])
    dummy.rename(columns={'Iris-setosa':'Iris-type-setosa', 
                          'Iris-versicolor':'Iris-type-versicolor',
                          'Iris-virginica':'Iris-type-virginica'}
                          , inplace=True)
    df = pd.concat([df, dummy], axis=1)
    df.drop("class", axis = 1, inplace=True)
    # get input and target data
    # convert dataframe to numpy array
    inputdata = df[['sepal_length', 'sepal_width', 'petal_length', 'petal width']].values
    targetdata = df[['Iris-type-setosa', 'Iris-type-versicolor', 'Iris-type-virginica']].values
    inputSize = len(inputdata[0])
    outputSize = len(targetdata[0])
    bias = [0.35, 0.60]
    
    bpnn = BPNN(inputSize, outputSize, learingRate=0.05)
    bpnn.addLayer(16, 'tanh', bias[0])
    bpnn.addLayer(5, 'sigmoid', 0.8)
    bpnn.addOutPutLayer('tanh', 0.7)
    bpnn.train(inputdata, targetdata, times=10000, learningRate=0.05, display=True, display_time=10)

    print(bpnn.test(inputdata, targetdata))
    print(bpnn.predict(inputdata[:5], targetdata[:5]))
'''


