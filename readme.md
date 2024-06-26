## README

### 项目简介
本项目实现了一个简单的神经网络，用于分类任务。包含数据预处理、神经网络模型的定义、前向传播、反向传播、训练过程以及损失曲线的绘制和结果展示。

### 文件列表
1. `n3.py`：主程序文件，包含神经网络的定义、训练和测试。
2. `README.md`：项目说明文档。

### 环境依赖
- Python 3.x
- numpy
- matplotlib

可以使用如下命令安装依赖：
```bash
pip install numpy matplotlib
pip install numpy numpy
```

### 使用说明
1. 数据归一化：使用 `minmax_` 或 `z_score_` 函数对数据进行归一化处理。
2. 神经网络初始化：创建 `Net` 类的实例，设置学习率、激活函数、激活函数的导数以及各层的节点数。
3. 训练网络：调用 `train` 方法进行训练，传入训练数据和目标数据，以及迭代次数。
4. 可视化结果：绘制损失曲线和预测结果与实际结果的对比图。

### 主要功能
- 数据归一化
- 神经网络前向传播
- 神经网络反向传播
- 训练过程
- 损失曲线和预测结果可视化

### 运行示例
```python
# 数据集
data = np.array([[1.76, -0.01, 1],
                 [0.59, 0.60, 0],
                 [-0.11, 0.32, 1],
                 [0.48, 0.19, 0],
                 [1.16, -0.80, 1],
                 [-0.09, 0.85, 0],
                 [1.86, 0.07, 1],
                 [-1.13, 0.12, 0],
                 [-0.20, 0.79, 0],
                 [0.32, -0.31, 1]]).T

data1 = data.copy()
data1[0] = minmax_(data[0])
data1[1] = minmax_(data[1])

net = Net(0.5, sigmoid, sigmoid_diff, [2, 2, 1])
net.w[0] = np.array([[0., 2, 0], [2, 0, 0]])
net.w[1] = np.array([[1., 1, 0]])

loss = net.train(data1[0:2], data1[2].reshape(1, -1), 50000)

plt.plot(loss)
plt.title("损失曲线")
plt.grid()
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.show()

plt.plot(net.y[-1][0], marker='o')
plt.plot(data1[2], marker='o')
plt.legend(["估计值", "真实值"])
plt.title("预测值与实际值对比")
plt.grid()
plt.show()
```

### 结果展示
通过以上代码，可以训练一个简单的神经网络并查看其损失曲线和预测值与实际值的对比图。代码展示了使用 `sigmoid` 激活函数的训练效果。

### 结论
本项目实现了一个简单的多层神经网络，通过归一化数据和训练过程展示了网络的训练效果。使用不同的激活函数可以看到网络在训练过程中的不同表现。希望通过这个项目，您能够理解神经网络的基本原理和实现方法。
