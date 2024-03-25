import torch
from pmath import poincare_mean, dist_matrix 
from nn import ToPoincare, HyperbolicDistanceLayer

dist = HyperbolicDistanceLayer(c=1)

def hyperbolic_distance(x1, x2):
    # 这里替换为你的 hyperbolic space 距离计算公式
    # 示例：使用 Poincaré 距离
    # Poincaré 距离公式：d(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))
    d = dist(x1, x2)
    norm_x1 = torch.norm(x1, dim=1)
    norm_x2 = torch.norm(x2, dim=1)

    numerator = 2 * torch.norm(x1.unsqueeze(1) - x2, dim=-1)**2
    a = (1 - norm_x1.unsqueeze(dim=1)**2)
    b = (1 - norm_x2.unsqueeze(dim=1)**2)
    denominator = a * b.t()
    
    poincare_distance = torch.acosh(1 + numerator/denominator)

    return poincare_distance

def distance_matrix(train_tensor, test_tensor):
    num_train = train_tensor.size(0)
    num_test = test_tensor.size(0)

    distances = torch.zeros(num_test, num_train)
    d = hyperbolic_distance(test_tensor, train_tensor)
    for i in range(num_test):
        distances[i, :] = hyperbolic_distance(test_tensor[i, :], train_tensor)

    return distances

# 示例用法
train_tensor = torch.randn(100, 3)
test_tensor = torch.randn(10, 3)

distances_matrix = distance_matrix(train_tensor, test_tensor)

print(distances_matrix)