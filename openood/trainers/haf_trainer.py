import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing

import torch
from torch.autograd import Function



from torch.nn.functional import linear, normalize

class PLogDet(Function):
  @staticmethod
  def forward(ctx, x):
    l = torch.cholesky(x)    
    ctx.save_for_backward(l)
    return 2 * l.diagonal(dim1=-2, dim2=-1).log().sum(-1)

  @staticmethod
  def backward(ctx, g):
    l, = ctx.saved_tensors
    n = l.shape[-1]
    # use cholesky_inverse once pytorch/pytorch/issues/7500 is solved
    return g * torch.cholesky_solve(torch.eye(n, out=l.new(n, n)), l)


plogdet = PLogDet.apply

class HAFTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.num_classes = self.config.dataset.num_classes
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        # self.pi = self.get_pi()
        # self.loss_fn = LogitNormLoss(tau=config.trainer.trainer_args.tau)
        self.loss_hce = Hierarical_CELoss(self.net)
        self.loss_ce = nn.CrossEntropyLoss()


    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits = self.net(data)
            #logits_classifier, features = self.net(data, return_feature=True)
            
            # normalizing features ！！！！########### m2
            # embeddings = self.net.penultimate_feature(data)
            # norm_embeddings = normalize(embeddings)
            # norm_weight_activated = normalize(self.net.classifier_3.weight)
            # logits = linear(norm_embeddings, norm_weight_activated)
            ############################################################
            
            loss_hce = self.loss_hce(logits, target)
            #loss_ce = self.loss_ce(logits_classifier, target)
            #loss = loss_hce+loss_ce
            loss = loss_hce

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


class Hierarical_CELoss2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        # self.fix_layer = self.get_fixlayer(self.net)
        self.distance_matrix = net.distance_matrix
        self.mapping = net.mapping_function
        self.dmax = net.distance_matrix.max()
        self.gamma = 1
        self.sim  = self.mapping(hdistance=self.distance_matrix,
        gamma=3,
        min_similarity=0)

    def hie_cross_entropy_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        targets = targets.view(-1,1)
        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1,targets)
        logpt = logpt.view(-1)
        score = torch.softmax(inputs, dim=1)
        conf, pred = torch.max(score, dim=1)
        # make batch distance
        batch_size = inputs.shape[0]
        batch_dis = torch.zeros(batch_size)
        batch_sim = torch.zeros(batch_size)
        for i in range(batch_size):
            dis = self.distance_matrix[pred[i],targets[i]]
            batch_dis[i] = dis/self.dmax
            sim = self.mapping(hdistance=self.distance_matrix,gamma=3,min_similarity=0)
            batch_sim[i] = sim[pred[i],targets[i]]
        #loss_focal = -1 * batch_dis.cuda()**self.gamma * logpt
        loss_focal = -1 * (1-batch_sim.cuda())**self.gamma * logpt
        loss_focal = loss_focal.mean()
        lambda1 = 0.5
        # lambda2 = 0.5
        lambda2 = 0
        loss = lambda1 * loss_focal + lambda2 * ce_loss        
        return loss

    def forward(self, y_pred, y_true):
        return self.hie_cross_entropy_loss(y_pred, y_true)
    
    def get_fixlayer(self, net):
        fix_layer = net.classifier_3.weight.cpu().numpy()
        return fix_layer


# version 2024/1/2
class Hierarical_CELoss1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fix_layer = self.get_fixlayer(self.net)
        self.distance_matrix = self.net.distance_matrix

    def hie_cross_entropy_loss(self, inputs, target):
        # 计算负对数似然损失
        h_info = torch.tensor(self.fix_layer, device=inputs.device)
        log_softmax = F.log_softmax(inputs, dim=1)
        target_vec = F.one_hot(target, num_classes=inputs.shape[-1])
        _, pred = torch.max(inputs, dim=1)
        h_weights = [h_info[pred[i], target[i]].reshape(-1, 1) for i in range(inputs.size(0))] 
        h_weights = 1 - torch.cat(h_weights)
        # sim = 1 - h_info[target]
        # loss = -torch.sum(log_softmax * target_vec * sim) / target.size(0)
        # loss = -torch.sum(log_softmax * target_vec * h_weights) / target.size(0)

        h_matrix = [torch.tensor(self.distance_matrix[pred[i], target[i]], device=inputs.device).reshape(-1, 1) for i in range(inputs.size(0))]
        h_matrix = torch.cat(h_matrix) / self.distance_matrix.max()
        # dist matrix
        loss = -torch.sum(log_softmax * target_vec * (h_matrix)) / target.size(0)
        return loss

    def forward(self, y_pred, y_true):
        return self.hie_cross_entropy_loss(y_pred, y_true)
    
    def get_fixlayer(self, net):
        fix_layer = net.classifier_3.weight.cpu().numpy()
        return fix_layer


# version 2024/1/
class Hierarical_CELoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fix_layer = net.classifier_3.weight
        self.distance_matrix = net.distance_matrix
        self.mapping = net.mapping_function
        self.dmax = net.distance_matrix.max()
        self.gamma = 1
        self.sim  = self.mapping(hdistance=self.distance_matrix,
        gamma=3,
        min_similarity=0)

    def hie_cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]


        score = torch.softmax(logits, dim=1)
        conf, pred = torch.max(score, dim=1)
        
        batch_wi = torch.zeros(logits.size(0),logits.size(1)).cuda()
        batch_wj = torch.zeros(logits.size(0),logits.size(1)).cuda()
        


        batch_wi = self.fix_layer[:,pred] # 230,256
        batch_wj = self.fix_layer[:,labels] #230,256,
        wiwj = torch.sum(batch_wi*batch_wj,dim=0)
        margin = wiwj

        s = 1
        
        
        index = torch.where(labels != -1)[0]
        
        #m2
        final_target_logit = target_logit - margin
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * s
        
        
        #m3
        # with torch.no_grad():
        #     target_logit.arccos_()
        #     logits.arccos_()
        #     final_target_logit = target_logit + margin
        #     logits[index, labels[index].view(-1)] = final_target_logit
        #     logits.cos_()
        # logits = logits * s   

        
        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss
   
        return loss

    def forward(self, y_pred, y_true):
        return self.hie_cross_entropy_loss(y_pred, y_true)
 
# version 2024/1/4    
class Hierarical_CELoss4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fix_layer = net.classifier_3.weight


    def hie_cross_entropy_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        
       
        score = torch.softmax(logits, dim=1)
        conf, pred = torch.max(score, dim=1)
        
        batch_wi = torch.zeros(logits.size(0),logits.size(1)).cuda()
        batch_wj = torch.zeros(logits.size(0),logits.size(1)).cuda()
        


        batch_wi = self.fix_layer[:,pred] # 230,256
        batch_wj = self.fix_layer[:,labels] #230,256,
        wiwj = torch.sum(batch_wi*batch_wj,dim=0)
        margin = wiwj
        s = 0.64
        
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - margin
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * s
        
        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss
   
        return loss

    def forward(self, y_pred, y_true):
        return self.hie_cross_entropy_loss(y_pred, y_true)


# Squared cosine similarity
class GeneralSquareCosineSimilarityLoss(nn.Module):
    def __init__(self, num_classes, reduction='batchmean'):
        super(GeneralSquareCosineSimilarityLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, fixed_cls, x, y):
        """
        fixed_cls: tensor with shape (num_classes, num_features)
        x: tensor with shape (batch_size, num_features), raw features
        y: tensor with shape (batch_size, num_classes), simlabel
        """

        sim = []
        for c in range(self.num_classes):
            # similarity between features and classifier c
            sim.append(
                torch.unsqueeze(
                    nn.functional.cosine_similarity(fixed_cls[c], x, dim=1), dim=1
                )
            )

        # sim tensor: (batch_size, num_classes) -> mimic (example,logits) format
        sim = torch.cat(sim, dim=1)
        sq_diff = torch.square(sim - y)
        if self.reduction == 'batchmean':
            loss = torch.sum(sq_diff)
            loss = loss / y.size()[0]
        elif self.reduction == 'sum':
            loss = torch.sum(sq_diff)
        else:
            raise ValueError(f"unrecognized reduction '{self.reduction}'")
        return loss

class MixedLoss_CEandGeneralSCSL(nn.Module):
    def __init__(self, num_classes, reduction='batchmean'):
        super(MixedLoss_CEandGeneralSCSL, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        # kl-divergence loss
        if self.reduction == 'batchmean':
            self.CE = nn.CrossEntropyLoss(reduction='mean')
            # squared cosine similarity loss
            self.GSCSL = GeneralSquareCosineSimilarityLoss(self.num_classes, reduction=self.reduction)
        elif self.reduction == 'sum':
            self.CE = nn.CrossEntropyLoss(reduction='sum')
            self.GSCSL = GeneralSquareCosineSimilarityLoss(self.num_classes, reduction=self.reduction)

    def forward(self, beta, alpha, fixed_cls, x, y_scaler, y_simlabel):
        """
        mixed_ratio: ratio between KL and SCSL
        fixed_cls: fixed classifier vectors, (num_classes, num_features)
        x: raw penultimate features, (batch_size, num_features)
        y_scaler: single labels, (batch_size, )
        y_softlabel: hie-aware soft-labels (similarities), (batch_size, num_classes)
        """
        logits = torch.matmul(x, fixed_cls.T)

        loss = beta * self.CE(logits, y_scaler) + \
               alpha * self.GSCSL(fixed_cls, x, y_simlabel)
        return loss