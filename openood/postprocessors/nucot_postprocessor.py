from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NucotPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            self.w, self.b = net.get_fc()

            with torch.no_grad():
                feature_id_train = []
                # for batch in tqdm(id_loader_dict['train'],
                #                   desc='Setup: ',
                #                   position=0,
                #                   leave=True):
                #     data = batch['data'].cuda()
                #     data = data.float()
                #     _, feature = net(data, return_feature=True)
                #     feature_id_train.append(feature.cpu().numpy())

                # 可以在这里保存一下feature以后就可以吧28-35行注释掉
                # torch.save(feature_id_train, "home/wy/ood/OpenOOD/f/feat.pth")

                # 保存后就可以直接用41行的代码加载了
                feature_id_train = torch.load("/home/gmr/ood/OpenOOD/f/feat.pth")
                feature_id_train = np.concatenate(feature_id_train, axis=0)

            # PCA to get the P matrix 
            self.u = -np.matmul(pinv(self.w), self.b)

            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        
        # Nuco
        numerator = np.linalg.norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1)
        denominator = np.linalg.norm(feature_ood, axis=-1)
        frac = numerator / denominator
        _, pred = torch.max(output, dim=1)
        logit_ood = feature_ood @ self.w.T + self.b

        energy_ood = logsumexp(logit_ood.numpy(), axis=-1) 
        score_ood = energy_ood + 100 * (-frac)
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
