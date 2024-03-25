from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from .base_postprocessor import BasePostprocessor
from ..hyptorch.nn import ToPoincare, HyperbolicDistanceLayer
from ..hyptorch.pmath import poincare_mean, dist_matrix 

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)

def _mobius_addition_batch(x, y, c):
    xy = torch.einsum("ij,kj->ik", (x, y))  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res





class HbKNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(HbKNNPostprocessor, self).__init__(config)
        # init hyperbolic 
        self.e2p = ToPoincare(c=1, train_c=False, train_x=False)

        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    feature = torch.cat(torch.load("/home/gmr/ood/OpenOOD/f/feat.pth"))

                    # feature into the hyperbolic space
                    feature = self.e2p(feature)
                    activation_log.append(feature.data.cpu().numpy())
                    break
            self.activation_log = np.concatenate(activation_log, axis=0)
            self.activation_log = feature.data.cpu().numpy()
            
            # Hb distance
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)

        feature = self.e2p(feature)
        # feature_normed = normalizer(feature.data.cpu().numpy())

        D, _ = self.predict(self.activation_log, feature, K=10)

        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K

    def dist_matrix(self, x, y, c=1.0):
        c = torch.as_tensor(c).type_as(x)
        sqrt_c = c ** 0.5
        return (
            2
            / sqrt_c
            * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
        )

    def predict(self, x, y, K, c=1.0):
        x = torch.tensor(x, device='cpu')
        y = torch.tensor(y, device='cpu')
        distance_matrix = self.dist_matrix(x, y)
        distances_sort = np.sort(distance_matrix)[:, :K]
        indices = np.argsort(distance_matrix)[:, :K]
        return distances_sort, indices