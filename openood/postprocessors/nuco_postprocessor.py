from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .k_means import KMeans


import numpy as np

import faiss
from copy import deepcopy
from sklearn import datasets, manifold


from torch.nn.functional import linear, normalize

others_num = 0
label_num = 200

others_num = 0
label_num = 10

# others_num = 20
# label_num = 100


print(others_num)

class NucoPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    #break
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature,_ = net(data, return_feature=True)

                    feature_id_train.append(feature.cpu().numpy())

                feature_id_train = np.concatenate(feature_id_train, axis=0)

                #feature_id_train = np.load("/home/wy/new/OpenOOD/123.npy")
                
                self.k_means = KMeans(k=label_num+others_num, pca_dim=10)
                train_cluster_id = self.k_means.cluster(feature_id_train)
                unique, counts = np.unique(train_cluster_id, return_counts=True)
                d = dict(zip(unique, counts))
                d = sorted(d.items(), key=lambda x:x[1])
                d_others = d[:others_num]
                ood_train_feature = []
                pure_feature_id_train = []
                indices = []
                for i in range(others_num):
                    indices.append(d_others[i][0])
                    
            
                for i in range(len(train_cluster_id)):
                    if train_cluster_id[i] in indices:
                        ood_train_feature.append(torch.tensor(feature_id_train[i]).unsqueeze(0))
                    else:
                        pure_feature_id_train.append(torch.tensor(feature_id_train[i]).unsqueeze(0))
                #self.ood_train_feature = torch.cat(ood_train_feature,dim=0).cuda()
                self.pure_feature_id_train = torch.cat(pure_feature_id_train,dim=0).cuda()
                


            # PCA to get the P matrix 
            ec1 = EmpiricalCovariance(assume_centered=True)
            ec1.fit(feature_id_train)
            eig_vals, eigen_vectors = np.linalg.eig(ec1.covariance_)
            
            ec2 = EmpiricalCovariance(assume_centered=True)
            ec2.fit(self.pure_feature_id_train.cpu().numpy())
            eig_vals2, eigen_vectors2 = np.linalg.eig(ec2.covariance_)
            
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)
            
            
            P_dim = 40
            cifar10_dim = 8
            
            cifar10_dim = 500
            
            self.P = np.ascontiguousarray(
                (eigen_vectors2.T[np.argsort(eig_vals2 * -1)[ : P_dim]]).T)
            
            self.P = np.ascontiguousarray(
                (eigen_vectors2.T[np.argsort(eig_vals * -1)[ : cifar10_dim]]).T)
            

            self.setup_flag = True
            
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        
        _, feature_ood,_ = net(data, return_feature=True)
        
        
        calibration = feature_ood.max()
        feature_ood = feature_ood.cpu()
        
        output = net(data)
        
        
        numerator = np.linalg.norm(np.matmul(feature_ood.numpy(), self.NS),axis=-1)
        neco_numerator = np.linalg.norm(np.matmul(feature_ood.numpy(), self.P),axis=-1)

        denominator = np.linalg.norm(feature_ood, axis=-1)
        # frac = numerator / (denominator+1e-6)
        # neco_frac = neco_numerator / (denominator+1e-6)
        
        
        frac = numerator / (denominator+1e-6)
        neco_frac = neco_numerator / (denominator+1e-6)

        _, pred = torch.max(output, dim=1)
        energy_ood = logsumexp(output.cpu().numpy(), axis=-1) 
   
        #score_ood =  energy_ood + neco_frac*calibration.cpu().numpy() -10*frac
        #score_ood =  energy_ood + frac*calibration.cpu().numpy()
        score_ood =  energy_ood + neco_frac*calibration.cpu().numpy()
      
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
