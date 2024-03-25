from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor
from copy import deepcopy
from typing import Any
from ..hyptorch.nn import ToPoincare
from ..hyptorch.pmath import poincare_mean, dist_matrix 
import sklearn.covariance

from .info import num_classes_dict
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class HbPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.e2p = ToPoincare(c=1, train_c=False, train_x=False)
        self.setup_flag = False
        self.num_classes = num_classes_dict[self.config.dataset.name]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            # with torch.no_grad():
            #     for batch in tqdm(id_loader_dict['train'],
            #                       desc='Setup: ',
            #                       position=0,
            #                       leave=True):
            #         data, labels = batch['data'].cuda(), batch['label']
            #         logits, features = net(data, return_feature=True)
            #         all_feats.append(features.cpu())
            #         all_labels.append(deepcopy(labels))
            #         all_preds.append(logits.argmax(1).cpu())

            # test hb
            all_feats = torch.load("/home/gmr/ood/OpenOOD/f/feat.pth")
            all_labels = torch.load("/home/gmr/ood/OpenOOD/f/label.pth")
            all_preds = torch.load("/home/gmr/ood/OpenOOD/f/pred.pth")

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # map features from euclidean space to hyperbolic space
            all_feats_h = self.e2p(all_feats)
            all_feats_h = all_feats_h.unsqueeze(1)

            self.centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats_h[all_labels.eq(c)].data
                center = poincare_mean(class_samples, dim=0, c=self.e2p.c)
                self.centered_data.append(center)

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)

        # compute the hyperbolic distance between features and centers
        features_h = self.e2p(features)
        
        center_data = torch.cat(self.centered_data).to('cuda')   
        dist = -dist_matrix(features_h, center_data, c=self.e2p.c)


        conf = torch.max(dist, dim=1)[0]
        return pred, conf


