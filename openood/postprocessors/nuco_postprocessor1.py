from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from tqdm import tqdm
import math
from .base_postprocessor import BasePostprocessor

def robust_pca(M):
    """ 
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    print(M.shape)
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    while not converged(M,L,S):
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
    return L,S
    
def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.
    The parameter tau is used as the scaling parameter to the shrink function.
    Returns the matrix obtained by computing U * shrink(s) * V where 
        U are the left singular vectors of X
        V are the right singular vectors of X
        s are the singular values as a diagonal matrix
    """
    U,s,V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U, np.dot(np .diag(shrink(s, tau)), V))
    
def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    """
    V = np.copy(X).reshape(X.size)
    for i in range(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)
            
def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = np.reshape(X,X.size)
    for i in range(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return max(np.sum(X,axis=0))

def converged(M,L,S):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    # print("error =", error)
    return error <= 10e-6




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
            self.w, self.b = net.get_fc()
            self.cls = net.get_cls()

            with torch.no_grad():
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net(data, return_feature=True)
                    feature_id_train.append(feature.cpu().numpy())

                # 可以在这里保存一下feature以后就可以吧28-35行注释掉
                # torch.save(feature_id_train, "home/wy/ood/OpenOOD/f/feat.pth")

                # 保存后就可以直接用41行的代码加载了
                # feature_id_train = torch.load("/home/gmr/ood/OpenOOD/f/feat.pth")
                feature_id_train = np.concatenate(feature_id_train, axis=0)

            # PCA to get the P matrix 
            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[int(self.dim):]]).T)
            
            # self.NS = self.cls
            ### LLE to get the P matrix
            # 5 30, 10 35 100 39
            # lle = LocallyLinearEmbedding(n_neighbors=20, n_components=self.dim, random_state=42, n_jobs=-1)
            # self.NS = lle.fit_transform(feature_id_train - self.u)

            # TSNE
            # tsne = TSNE(n_components=self.dim, method='exact', random_state=42)
            # self.NS = tsne.fit_transform(self.pca)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()

        
        # Nuco
        numerator = np.linalg.norm(np.matmul(feature_ood.numpy(), self.NS),
                          axis=-1)
        denominator = np.linalg.norm(feature_ood, axis=-1)
        frac = numerator / denominator

        recon = np.matmul(feature_ood.numpy(), self.NS)

        conf, pred = torch.max(output, dim=1)
        # msp = conf.cpu().numpy()
        energy_ood = logsumexp(conf.cpu().numpy(), axis=-1)
        output = output.cpu().numpy()

        result = output - recon
        energy_ood = logsumexp(recon, axis=-1)

        
        score_ood = energy_ood
        # score_ood = self.mahalanobis_distance(output, recon)


        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim

    def mahalanobis_distance(self, X, Y):
        # 计算均值向量
        mean_X = np.mean(X, axis=0)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X, rowvar=True)
        
        # 计算协方差矩阵的逆矩阵
        cov_inv = np.linalg.inv(cov_matrix)
        
        # 计算马氏距离
        distance = np.sqrt(np.trace(np.dot(np.dot((X - Y).T, cov_inv), X - Y)))
        
        return distance