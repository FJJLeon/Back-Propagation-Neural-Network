# Back Propagation Neural Network 实验报告
* 方俊杰
* 516030910006
## 实验内容
* 实现基本后传递神经元网络 Back Propagation Neural Networks (**BPNN**) 算法
## 实验环境
* WIN10
* Python 3.6.3
* Numpy 1.15.3
* Pandas 0.23.4
## 实现要求
1. 网络结构：1 输入层、N 层隐藏层、1 隐藏层 （N可变）
2. 隐藏层节点数：N（可变）
3. 激活函数：Logistic、tanh、ReLU
4. Learning Rate 和 Bias：可变 
5. 数据输入：矩阵
6. 输入层维度：N（可变）
7. 输出层维度：N（可变）
8. 实验数据：课件数据、自建数据、UCI ML Repository IRIS数据
## 实验原理
* Back Propagation Neural Networks (BPNN)。BP神经网络是最基础的神经网络，其输出结果采用前向传播，误差采用反向（Back Propagation）传播方式进行。前向传播中输入数据加权求和经过激活函数输出到下一层级，层层传播到输出。与实际值的误差通过反向传播回来用链式法则和梯度下降的方式更新权值。多次迭代希望最优化参数。

## 对象类定义
1. 定义了 3 个类，分别是表示单个神经元的Node、表示单层的Layer、表示整个神经网络的BPNN
* Node类维护一个权值向量
```
class Node:
    def __init__(self, previousSize, activate, bias):
        # previousSize 为前一个层级的神经元数量
        # activate 设定该层级神经元激活函数类型
        # bias 设定该层偏置
        self.weight = [rand(-0.5, 0.5) for i in range(previousSize)]
        # 权值向量，用随机值初始化

    def receive(self, previousData):
        # 用于接收前一层layer的输出数据计算出该Node的输出数据并保存

    def updateWeight(self, penalty, learningRate=0.5):
        # 使用反向传播的误差来更新该Node的权重
```
* Layer类维护一个Node数组
```
class Layer:
    def __init__(self, previousSize, ownSize, activate, bias):
        # each layer should record his previous layer size and own size
        # should have a bias
        # should maintain a set of node within this layer
        self.nodeSet = [Node(previousSize, self.activate, self.bias) for i in range(ownSize)]
        ···
    def receive(self, previousData):
        # reveive previous layer data,
        # pass to each node and get output which will be flow into next or output

    def getError(self, currentDelta):
        # compute error of this layer for computing delta

    def getDeltaInHidden(self, lastError):
        # using next layer error to compute this layer delta

    def updateWeight(self, penalty, learningRate=0.5):
        # update weight of each node 
        # after computing every delta in every layer
```
* BPNN类维护一个Layer数组
* 使用整个网络的输入维度、输出维度、学习率初始化
* 提供接口
    addLayer：添加特定层级
    addOutPutLayer：确定输出层激活函数和偏置
    train：接收二维矩阵作为输入、实际值、迭代次数、学习率、是否显示过程来训练模型
    test：接收输入来计算损失函数值
    predict：接收输入和实际值打印预测结果
```
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

    def addOutPutLayer(self, outActivate, outBias):
        # add output layer 
        # finsh add

    def forward(self, inputData):
        # forwarding,
        # and each layer output will be record inner layer

    def backPropagation(self, target):
        # back propagation,
        # delta is used to update weight in each layer
            # new_weight = old_weight - delta * output, for each node
        # error is used to compute previous layer delta
            # previous_layer_delta = error * partial derivative

    def train(self, inputData, targetData, times, learningRate, display=False, display_time=10):
        # inputdata is a two-dimensional array
        # one line of which is one sample
        # each col is an attribute.
        # iteration for times
        
    def test(self, inputData, targetData):
        # test model and return loss
    
    def predict(self, inputData, targetData):
        # give prediction of input compared with target
        for inputd, target in zip(inputData,targetData):
            print(inputd, ' --> ', self.forward(inputd), ' expect: ', target)
```

## 激活函数选择
* 可以选择三种激活函数 sigmoid、 tanh、 ReLU
```
    activateFunc = {
        'sigmoid':sigmoid,
        'tanh':tanh,
        'relu':relu
        }
    # activation function
    def activate(x, type):
        if (type not in activateFunc.keys()):
            raise RuntimeError('activate type not found')
        return activateFunc[type](x)

    derActivateFunc = {
        'sigmoid':derSigmoid,
        'tanh':derTanh,
        'relu':derRelu
    }
    # partial derivative of the activation function
    # note, the argu out is activate(x)
    def derActivate(out, type):
        if (type not in derActivateFunc.keys()):
            raise RuntimeError('activate type not found')
        return derActivateFunc[type](out)
```
* 可以直接用形如 activate(x, 'sigmoid') 的方式在类中调用

## 测试
* 课件数据测试
1. 测试代码, 一层隐藏层，激活函数sigmoid，迭代100000次，学习率0.05
```
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
```
2. 测试结果, 损失函数迅速收敛，效果较好，不同的激励函数收敛迭代所需次数有一定差别
![课件数据测试](slide.png)

* 自定义矩阵数据测试
1. 测试代码, 两层隐藏层，一层12节点用tanh激活，一层8节点用sigmoid激活，迭代100000次，学习率0.05
```
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
    bpnn.addLayer(8, 'sigmoid', bias[1])
    bpnn.addOutPutLayer(outActivate='tanh', outBias=bias[2])
    bpnn.train(inputdata, targetdata, times=100000, learningRate=0.05, display=True, display_time=10)

    print(bpnn.test(inputdata, targetdata))
    print(bpnn.predict(inputdata, targetdata))
```
2. 测试结果，不总能收敛，效果不好，有时出现梯度弥散或梯度消失的情况
* 成功情况
![succ](succ.png)
* 失败情况
![fail](failed.png)
* IRIS数据集测试
1. 测试代码，数据集先用pandas对分类进行二值离散化作为输入
```
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
```
2. 测试结果，几乎不成功，可能由于我的实现有问题，或者我的参数设定，网络结构问题，或者BPNN网络本身原因，未见收敛情况
![failed](irisfailed.png)

## 实验感想
* 人工智能、机器学习等领域如今热火朝天，作为一名软件学院的大学生，学习了解使用相关知识对自己的成长大有裨益。在本次实验对于BPNN网络的实现中，我了解了其背景知识，实现方法等相关知识，查阅了许多相关资料，了解到对于该神经网络的实现细节，加深了我对BPNN的理解。同时，在对不同数据集的测试中发现该网络在使用中并不能很好的投入实践，可能需要调整参数和结构，这也让我意识到要学习更多优化知识，学习更多的模型方法。
