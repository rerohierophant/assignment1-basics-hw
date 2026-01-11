import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter( #作用在于告诉模型这个参数需要学习，让优化器更新
            torch.Tensor(out_features, in_features)
            #为什么先out再in：为了提升内存访问效率。belike [[neuron0权重], [neuron1权重]], 在内存里是连续的
            #以及转置操作很轻量，只改变步长
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        # 做Kaiming权重初始化, 防止Relu的梯度爆炸/梯度消失
        bound = 1.0 / self.in_features ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (batch, in_features)
        # weight: (out, in)
        y = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.scale * self.gamma * x / (x.norm(2, dim=-1, keepdim=True) + self.eps)


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):
        x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values) #在实现中需要减去最大值，防止exp溢出
        return x / x.sum(dim=dim, keepdim=True)


class SwiGLU(nn.Module):
    def __init__(self, ):
        super().__init__()
