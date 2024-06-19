#21310032-郝斌-期末大作业3
import numpy as np
import matplotlib.pyplot as plt
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据归一化函数
def minmax_(data):
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def z_score_(data):
    return (data - np.mean(data)) / np.std(data)

# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_diff(t):
    return t * (1 - t)

def tanh2(x):
    return 2 / (1 + np.exp(-x))-1

def tanh2_diff(t):
    return (1 - t**2)/2

# 神经网络类
class Net:
    def __init__(self, alpha, f, f_diff, num_node):
        self.alpha_m = 0.6
        self.alpha = alpha
        self.f1 = np.vectorize(f_diff)
        self.f = np.vectorize(f)
        self.w = [np.random.randn(num_node[i + 1], num_node[i] + 1) for i in range(len(num_node) - 1)]
        self.layer = len(num_node)
        self.layers = num_node
        self.y=[]
    # 前向传播
    def forward(self, x):
        self.y = [x] 
        for i in range(self.layer - 1):
            self.y.append(self.f(self.w[i] @ np.vstack([self.y[-1], np.ones(self.y[-1].shape[1])])))
        return self.y[-1]

    # 反向传播
    def backward(self, target):
        grads = [np.zeros_like(i) for i in self.w]
        d = (self.y[-1] - target) * self.f1(self.y[-1])
        y_ones = [np.vstack([i, np.ones([1, i.shape[1]])]) for i in self.y[:-1]]
        for i in range(self.layer - 1, 0, -1):
            grad = -self.alpha * d @ y_ones[i - 1].T
            grads[i-1] = grad + self.alpha_m * grads[i-1]
            if i == 1:
                break
            d = (self.w[i - 1].T @ d) * self.f1(y_ones[i - 1])
            d = d[:-1]
        for i in range(len(grads)):
            self.w[i] += grads[i]

    # 顾名思义
    def loss(self, target):
        return np.mean((self.y[-1] - target) ** 2)

    # 顾名思义
    def train(self, x, target, n):
        loss = []
        for _ in range(n):
            self.forward(x)
            self.backward(target)
            current_loss = self.loss(target)
            loss.append(current_loss)
            
        return loss

# 数据集
data=[[float(j) for j in i.split()]for i in """1.76	-0.01	1
0.59	0.60	0
-0.11	0.32	1
0.48	0.19	0
1.16	-0.80	1
-0.09	0.85	0
1.86	0.07	1
-1.13	0.12	0
-0.20	0.79	0
0.32	-0.31	1""".split('\n')]

data=np.array(data).T
# 训练与结果展示
for i in range(2, 3):
    data1 = data.copy()
    if i == 1:
        a = (0.5, sigmoid, sigmoid_diff, [2, 2, 1])
        data1[0] = minmax_(data[0])
        data1[1] = minmax_(data[1])
        # data1[2] = minmax_(data[2])
        title = "第一问sigmoid"
    else:
        a = (0.5, tanh2, tanh2_diff, [2, 2, 1])
        data1[0] = z_score_(data[0])
        data1[1] = z_score_(data[1])
        # data1[2] = z_score_(data[2])
        title = "第二问tanh"

    net = Net(*a)
    # 初始化权重
    net.w[0] = np.array([[0., 2, 0], [2, 0, 0]])
    net.w[1] = np.array([[1., 1, 0]])

    # 训练网络
    loss = net.train(data1[0:2], data1[2].reshape(1, -1), 50000)  # 增加训练次数
    plt.show()
    print(title)
    for j in range(1, 3):
        print(f"W{j}")
        print(net.w[j - 1][:, :2])
        print(f"W{j}偏置")
        print(net.w[j - 1][:, 2])
    
    # 绘制损失曲线
    plt.plot(loss)
    plt.title(f"{title}损失曲线")
    plt.grid()
    plt.xlabel("迭代次数")
    plt.ylabel("损失")
    plt.show()

    # 绘制预测值与真实值对比图
    plt.plot(net.y[-1][0], marker='o')
    plt.plot(data1[2], marker='o')
    plt.legend(["估计值", "真实值"])
    plt.title(f"{title}预测值与实际值对比")
    plt.grid()
    plt.show()
