
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(
    nn.Linear(20, 256), 
    nn.ReLU(), 
    nn.Linear(256, 10)
)

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
  
        self.out = nn.Linear(256, 10)  # 输出层
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        print(self._modules)

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


X = torch.rand(2, 20)
res = net(X)

print(res, res.shape)

net2 = MLP()
res2 = net2(X)
# print(res2, res.shape)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, block in enumerate(args):
            self._modules[str(idx)] = block
        
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X



net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net._modules)
print(net)


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)

        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net3 = FixedHiddenMLP()
print(net3(X))

if __name__ == '__main__':
    print('main')

    mat1 = torch.tensor([[1, 2], [2, 3]])
    mat2 = torch.tensor([[1, 2], [1, 2]])

    # res = torch.mm(mat1, mat2)
    # print(res)


