from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

import os



# npy_path = "/home/g/ood/OpenOOD/taxonomy_level0.npy"



class soft_label_loss():
  
    def __init__(self, classes_num,group_slice):
        self.hardness = 4
        self.classes_num = classes_num
        self.group_slice = group_slice
        self.distance_matrix = self.get_distance_matrix()
        self.all_soft_labels = self.make_all_soft_labels()

    def get_distance_matrix(self):
        classes_num = self.classes_num
        group_slice = self.group_slice
        distance_matrix = torch.zeros([classes_num, classes_num])
        distance_matrix = distance_matrix.cuda()
        distance_matrix = torch.add(distance_matrix,2)
        
        group_num = 0
        star = group_slice[group_num][0]
        end = group_slice[group_num][1]

        for i in group_slice:
            star = i[0]
            end = i[1]
            distance_matrix[star:end][star:end] = 1
        
        distance_matrix.fill_diagonal_(0)

        return distance_matrix


    def make_all_soft_labels(self):
        hardness = self.hardness
        distance_matrix = self.distance_matrix
        hardness = self.hardness
        max_distance = torch.max(distance_matrix) 
        distance_matrix /= max_distance
        soft_labels = torch.exp(-hardness * distance_matrix) / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)
        return soft_labels

    def make_batch_soft_labels(self,all_soft_labels, target, num_classes, batch_size):
        
        target_len = len(target)
        if target_len < batch_size:
            batch_size = target_len
        
        
        soft_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda()
        for i in range(batch_size):
            this_label = all_soft_labels[:, target[i]] 
            soft_labels[i, :] = this_label 
        return soft_labels
    
    def get_the_loss(self,target,output,batch_size):
        output = torch.nn.functional.log_softmax(output, dim=1) # not sure
        target_distribution = self.make_batch_soft_labels(self.all_soft_labels,target,self.classes_num,batch_size)
        loss_function = nn.KLDivLoss().cuda()
        loss = loss_function(output, target_distribution)
        return loss


            

def get_mixup(dataset_size):
    return 0.0 if dataset_size < 20_000 else 0.1


def get_group_slices(classes_per_group):
    group_slices = []
    start = 0
    for num_cls in classes_per_group:
        end = start + num_cls + 1
        group_slices.append([start, end])
        start = end
    return torch.LongTensor(group_slices)


def get_schedule(dataset_size,batch_size,epoch_num):
    multiply_factor = 256/batch_size * (epoch_num/5)
    if dataset_size < 20_000:
        return [item*multiply_factor for item in [100, 200, 300, 400, 500]]
    elif dataset_size < 500_000:
        return [item*multiply_factor for item in [500, 3000, 6000, 9000, 10_000]]
    else:
        return [item*multiply_factor for item in [500, 6000, 12_000, 18_000, 20_000]]


def get_lr(step, dataset_size, base_lr=0.003,batch_size=8,epoch_num=16):
    """Returns learning-rate for `step` or None at the end."""
    #ww
    multiply_factor = 256/batch_size
    base_lr = base_lr/multiply_factor #batch size 256-> 8
    supports = get_schedule(dataset_size,batch_size,epoch_num) # [500, 6000, 12000, 18000, 20000]
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif step >= supports[-1]:
        return None
    # Staircase decays by factor of 10
    else:
        for s in supports[1:]:
            if s < step:
                base_lr /= 10
        return base_lr


def mixup_data(x, y, lam):
    """Returns mixed inputs, pairs of targets, and lambda."""
    indices = torch.randperm(x.shape[0]).to(x.device) # disturb the order

    mixed_x = lam * x + (1 - lam) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion_group(criterion, pred, y_a, y_b, lam, group_slices):
    return lam * calc_group_softmax_loss(criterion, pred, y_a, group_slices) \
           + (1 - lam) * calc_group_softmax_loss(criterion,
                                               pred, y_b, group_slices)


def calc_group_softmax_loss(criterion, logits, labels, group_slices): #logit.shape: [16,1008]  labels.shape:[16,8]
    num_groups = group_slices.shape[0]
    loss = 0
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]
        group_label = labels[:, i] # take the column then it change to the label of group, not label recoding every group state fro a sample , but just for each group by taking every samples in a batch    group_label shape [16]

        loss += criterion(group_logit, group_label) #group_logit.shape [16,399],,[16] [16.399]

    return loss


def calc_group_softmax_acc(logits, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    num_samples = logits.shape[0]

    all_group_max_score, all_group_max_class = [], []

    smax = torch.nn.Softmax(dim=-1).cuda()
    cri = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]:group_slices[i][1]]
        group_label = labels[:, i]
        loss += cri(group_logit, group_label)

        group_softmax = smax(group_logit)
        group_softmax = group_softmax[:, 1:]  # disregard others category
        group_max_score, group_max_class = torch.max(group_softmax, dim=1)
        group_max_class += 1  # shift the class index by 1

        all_group_max_score.append(group_max_score)
        all_group_max_class.append(group_max_class)

    all_group_max_score = torch.stack(all_group_max_score, dim=1)
    all_group_max_class = torch.stack(all_group_max_class, dim=1)

    final_max_score, max_group = torch.max(all_group_max_score, dim=1)

    pred_cls_within_group = all_group_max_class[torch.arange(num_samples),
                                                max_group]

    gt_class, gt_group = torch.max(labels, dim=1)

    selected_groups = (max_group == gt_group)

    pred_acc = torch.zeros(logits.shape[0]).bool().cuda()

    pred_acc[selected_groups] = (
        pred_cls_within_group[selected_groups] == gt_class[selected_groups])

    return loss, pred_acc


def topk(output, target, ks=(1, )):
    """Returns one boolean vector for each k, whether the target is within the
    output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def run_eval(model, data_loader, step, group_slices, num_group):
    # switch to evaluate mode
    model.eval()

    all_c, all_top1 = [], []

    train_dataiter = iter(data_loader)
    for train_step in tqdm(range(1,
                                 len(train_dataiter) + 1),
                           desc='Test : ',
                           position=0,
                           leave=True,
                           disable=not comm.is_main_process()):
        batch = next(train_dataiter)
        data = batch['data'].cuda()
        group_label = batch['group_label'].cuda()
        class_label = batch['class_label'].cuda()
        labels = []
        for i in range(len(group_label)):
            label = torch.zeros(num_group, dtype=torch.int64)
            label[group_label[i]] = class_label[i] + 1
            labels.append(label.unsqueeze(0))
        labels = torch.cat(labels, dim=0).cuda()

        with torch.no_grad():
            x = data
            y = labels

            # compute output, measure accuracy and record loss.
            logits = model(x)
            if group_slices is not None:
                c, top1 = calc_group_softmax_acc(logits, y, group_slices)
            else:
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1 = topk(logits, y, ks=(1, ))[0]

            all_c.extend(c.cpu())  # Also ensures a sync point.
            all_top1.extend(top1.cpu())

    model.train()
    # print(f'Validation@{step} loss {np.mean(all_c):.5f}, '
    #       f'top1 {np.mean(all_top1):.2%}')

    # writer.add_scalar('Val/loss', np.mean(all_c), step)
    # writer.add_scalar('Val/top1', np.mean(all_top1), step)
    return all_c, all_top1


class MOSTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        
        
        
        if config.model.checkpoint is not None:
            self.net = self.load_checkpoint(config,net)

        self.net = net.cuda()
        self.train_loader = train_loader
        self.config = config
        self.lr = config.optimizer.lr

        self.soft_label_loss = 0

        trainable_params = filter(lambda p: p.requires_grad, net.parameters())
        self.optim = torch.optim.SGD(trainable_params,
                                     lr=self.lr,
                                     momentum=0.9)
        self.optim.zero_grad()
        self.net.train()

        # train_set len
        self.train_set_len = config.dataset.train.batch_size * len(
            train_loader)
        self.mixup = get_mixup(self.train_set_len)
        self.cri = torch.nn.CrossEntropyLoss().cuda()

        self.accum_steps = 0
        self.mixup_l = np.random.beta(self.mixup,
                                      self.mixup) if self.mixup > 0 else 1

        # if specified group_config
        if (config.trainer.group_config.endswith('npy')):
            self.classes_per_group = np.load(config.trainer.group_config)
        elif (config.trainer.group_config.endswith('txt')):
            self.classes_per_group = np.loadtxt(config.trainer.group_config,
                                                dtype=int)
        else:
            # self.classes_per_group = np.load(npy_path)
            self.cal_group_slices(self.train_loader) # here!

        self.num_group = len(self.classes_per_group) # 8
        self.group_slices = get_group_slices(self.classes_per_group)# indices pairs array
        self.group_slices = self.group_slices.cuda()

        self.soft_label_loss = soft_label_loss(1008,self.group_slices)


        self.step = 0
        self.batch_split = 1
        
        
        self.start_epoch = 0

    
    
    
    


    def load_checkpoint(self,config, model):
        optimizer = config.optimizer
        if os.path.isfile(config.model.checkpoint):
            print("=> loading checkpoint '{}'".format(config.model.checkpoint))
            checkpoint = torch.load(config.model.checkpoint)
            #self.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint)
            #optimizer.load_state_dict(checkpoint["optimizer"])
            #self.step = checkpoint["steps"]
            #print("=> loaded checkpoint '{}' (epoch {})".format(config.model.checkpoint, checkpoint["epoch"]))
            
        return 0
        
    
    
    
    
    
    
    
    def cal_group_slices(self, train_loader): # deffirent way with original mos(uses original dataset split, not group label other than saving a ordring group array npy, the the faults is that the data set must be loading and saving in order, and can not be disrupted), but they do the same thing
        # cal group config
        group = {}
        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='cal group_config',
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            group_label = deepcopy(batch['group_label'])
            class_label = deepcopy(batch['class_label'])
            # collect all classes labels in each group  ,e.g. group:{'1': [408], '0': [85, 28, 377], '4': [39]}
            for i in range(len(class_label)): # deal in each batch 
                gl = group_label[i].item()
                cl = class_label[i].item()

                try:
                    group[str(gl)]
                except:
                    group[str(gl)] = []

                if cl not in group[str(gl)]:
                    group[str(gl)].append(cl)

        self.classes_per_group = []
        for i in range(len(group)):
            self.classes_per_group.append(max(group[str(i)]) + 1)

    def train_epoch(self, epoch_idx):
        total_loss = 0

        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            group_label = batch['group_label'].cuda()# [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
            class_label = batch['class_label'].cuda()
            
             

            labels = []
            for i in range(len(group_label)): # i: batch size
                label = torch.zeros(self.num_group, dtype=torch.int64) # label for one sample shape 8
                label[group_label[i]] = class_label[i] + 1 # and it's not possible to take the "group" label, cuz ervery class label adding 1# plusing one is to add the group one to the sum label index,label1:[40,  0,  0,  0,  0,  0,  0,  0] label2:[  0, 294,   0,   0,   0,   0,   0,   0] this is the new label that non-real numebr is the new all 1008 classes label, num-zero index in the len 8 array is the group label
                labels.append(label.unsqueeze(0))#labels:[tensor([[40,  0,  0,...  0,  0]]), tensor([[  0, 294,  ... 0,   0]])]
            labels = torch.cat(labels, dim=0).cuda()# shape:[16,8]

            # Update learning-rate, including stop training if over.
            lr = get_lr(self.step, self.train_set_len, self.lr,self.config.dataset.train.batch_size,epoch_num=self.config.optimizer.num_epochs)
            if lr is None:
                break
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

            if self.mixup > 0.0: #!!!
                x, y_a, y_b = mixup_data(data, labels, self.mixup_l)

            logits = self.net(data)

            y_a = y_a.cuda()
            y_b = y_b.cuda()
            if self.mixup > 0.0:

                c = mixup_criterion_group(self.cri, logits, y_a, y_b,
                                          self.mixup_l, self.group_slices) # c:group softmsc loss
            else:
                c = calc_group_softmax_loss(self.cri, logits, labels,
                                            self.group_slices)

            
            soft_label_loss = self.soft_label_loss.get_the_loss(class_label,logits,self.config.dataset.train.batch_size)
            c += soft_label_loss
            
            c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.
            #  # Accumulate grads
            
            (c / self.batch_split).backward()
            
            self.accum_steps += 1

            # accstep = f' ({self.accum_steps}/{self.batch_split})' \
            #     if self.batch_split > 1 else ''
            # print(
            #     f'[step {self.step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})')

            total_loss += c_num 

            # Update params
            # if self.accum_steps == self.batch_split:
            self.optim.step()
            self.optim.zero_grad()

            self.step += 1
            self.accum_steps = 0
            # Sample new mixup ratio for next batch
            self.mixup_l = np.random.beta(self.mixup,
                                          self.mixup) if self.mixup > 0 else 1

        # torch.save(self.net.state_dict(),
        #            os.path.join(self.config.output_dir, 'mos_epoch_latest.ckpt'))

        # step, all_top1 = run_eval(self.net, self.train_loader, self.step, self.group_slices,
        #          self.num_group)

        loss_avg = total_loss / len(train_dataiter)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        # metrics['acc'] = np.mean(all_top1) # the acc used in there is the top1 acc

        return self.net, metrics, self.num_group, self.group_slices
