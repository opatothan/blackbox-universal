import torch
import numpy as np
import os, argparse
from models import PytorchModel
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data,load_imagenet_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
from paper_model import vgg16, BasicCNN
from gradient_functions import white_box_grad
import generator_blackbox
import torchvision.models as models
import generator_whitebox
import pandas as pd
import random



net = vgg16()
#net = VGG_plain('VGG16', 10, img_width=32)
net = torch.nn.DataParallel(net, device_ids=[0])
load_model(net, r"C:\Users\Paul Hao\Documents\Adversial Deep learning\attackbox-master\model\cifar10_vgg_plain.pth")
net.cuda()
net.eval()
model = net
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)

train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()

testing_data =torch.utils.data.Subset(test_dataset, range(5000))
v, data = generator_blackbox.black_box_generate(amodel, testing_data)
#v = generator.white_box_generate(amodel, dataset)

