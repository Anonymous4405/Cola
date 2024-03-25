import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

from openood.evaluation_api import Evaluator
from openood.networks import coc_medium # just a wrapper around the ResNet
from openood.networks import ResNet50
from openood.networks import ResNet18_224x224
from openood.networks.haf import ResNet18_fix

net = ResNet18_fix(num_classes=10)
# path = "/home/gmr/ood/OpenOOD/results/cifar100_resnet18_fix_base_e100_lr0.1_default_test/s5/cifar100_resnet18_fix_base_e100_lr0.1_default/s5/best_epoch86_acc0.7770.ckpt"
path21 = "results/cifar10a1/s5/cifar10_resnet18_fix_haf_e100_lr0.1_alpha3_default/best.ckpt"
net.load_state_dict(torch.load(path21, map_location='cuda:0'))


# net = ResNet50(num_classes=1000)
# net.load_state_dict(torch.load("/home/gmr/ood/OpenOOD/results/pretrained_weights/resnet50_imagenet1k_v1.pth"))
net.cuda()
net.eval()

postprocessor_name = "nuco"

evaluator = Evaluator(
    net,
    id_name="cifar10",
    data_root="./data/",
    config_root="/home/gmr/ood/OpenOOD/configs",
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    postprocessor=None,                    # if you want to use your own postprocessor
    batch_size=256,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=32)  

metrics = evaluator.eval_ood(fsood=False)