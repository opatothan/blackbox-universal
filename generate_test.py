import torch
import numpy as np
import os, argparse
#from models import PytorchModel
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, load_imagenet_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
from paper_model import vgg16, BasicCNN
import torchvision.models as models
    


parser = argparse.ArgumentParser(description="settings for the attack")

parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Dataset to be used, [MNIST, CIFAR10, Imagenet]')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch_size')
parser.add_argument('--test_batch', type=int, default=10,
                    help='test batch number')
parser.add_argument('--model_dir', type=str, required=True, help='model loading directory')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.dataset == "MNIST":
    # net = MNIST()
    net = BasicCNN()
    # net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,args.model_dir)
    net = torch.nn.DataParallel(net, device_ids=[0])
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR10':
    # net = CIFAR10() 
    net = vgg16()
    #net = VGG_plain('VGG16', 10, img_width=32)
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,args.model_dir)

    #device = torch.device("cuda")
    #net = WideResNet().to(device)
    #load_model(net, 'model/cifar10_gpu.pt')

    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data(args.test_batch_size)
elif args.dataset == 'Imagenet':
    net = models.resnet50(pretrained=True) 
    net = torch.nn.DataParallel(net, device_ids=[0])
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet_data(args.test_batch_size)
else:
    print("Unsupport dataset")
    #os.exit(0)
    
net.cuda()
net.eval()

model = net 

generator = generator(model, )