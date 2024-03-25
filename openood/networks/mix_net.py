"""
ContextCluster implementation
# --------------------------------------------------------
# Context Cluster -- Image as Set of Points, ICLR'23 Oral
# Licensed under The MIT License [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)
# --------------------------------------------------------
"""
import os
import copy
import torch
import torch.nn as nn
from resnet50 import ResNet
from context_cluster_net import coc_medium


class MixNet(nn.Module):
    def __init__(self, num_classes, pretrined=False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrined = pretrined
        self.net1 = self.init_net(coc_medium, self.num_classes, "/home/gmr/ood/OpenOOD/results/pretrained_weights/coc_medium.pth")
        self.net2 = self.init_net(ResNet, self.num_classes, "/home/gmr/ood/OpenOOD/results/pretrained_weights/resnet50_imagenet1k_v1.pth")
        # self.fc

    def init_net(self, net, num_classes, pretrained=False):
        net = net(num_classes=num_classes)
        if pretrained:
            net.load_state_dict(torch.load(pretrained), strict=False)
        return net

    def forward(self, x):
        featrue1 = self.net1(x)
        featrue2 = self.net1(x)
        featrue = torch.min(featrue1, featrue2)
        # logits_cls = self.fc(featrue)

        return 1

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = MixNet(num_classes=1000)
    out = model(input)
    print(model)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters/1024**2))