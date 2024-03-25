from nn import ToPoincare
from pmath import poincare_mean, dist_matrix
import torch
import torch.nn as nn

class Hyperbolic(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.e2p = ToPoincare(c=1, train_c=False, train_x=False)

    def forward(self, x):
        x_h = self.e2p(x)
        x_h = x_h.unsqueeze(1)
        center = poincare_mean(x_h, dim=0, c=self.e2p.c)
        return center

if __name__ == "__main__":
    file_path = "/home/gmr/ood/OpenOOD/features/ImageNet200/feature.pth"
    label_path = "/home/gmr/ood/OpenOOD/features/ImageNet200/labels.pth"

    feature = torch.load(file_path)
    feature = torch.cat(feature)

    label = torch.load(label_path)
    label = torch.cat(label)

    hb = Hyperbolic()

    a = 1

