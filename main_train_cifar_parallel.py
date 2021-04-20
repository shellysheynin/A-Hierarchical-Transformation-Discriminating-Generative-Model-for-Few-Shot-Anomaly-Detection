from FewShot_models.manipulate import *
from FewShot_models.training_parallel import *
import FewShot_models.functions as functions
from PIL import Image
import os, sys
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
from anomaly_detection_cifar_parallel import anomaly_detection
import cv2
import itertools
from cifar_loader import download_class_cifar
from mnist_loader import download_class_mnist
from fashionmnist_loader import download_class_FashionMnist

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--dataset', help='cifar/mnist/fashionmnist/mvtec/paris', default='cifar')
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--pos_class', help='normal class', required=True)
    parser.add_argument('--random_images_download', help='random selection of images', default=False)
    parser.add_argument('--num_images', type=int, help='number of images to train on', default=1)
    parser.add_argument('--if_download_class', type=bool, help='do you want to download class', default=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--size_image', type=int, help='size orig image', default=128)
    parser.add_argument('--num_epochs', type=int, help='num epochs', default=1)
    parser.add_argument('--policy', default='')
    parser.add_argument('--index_download', help='index in dataset for starting download', type=int, default=1)
    parser.add_argument('--use_internal_load', help='using another dataset', default=False)
    parser.add_argument('--experiment', help='task to be done', default='stop_signs')
    parser.add_argument('--test_size', help='test size', type=int, default=10000)
    parser.add_argument('--num_transforms', help='54 for rgb, 42 for grayscale', type=int, default=54)
    parser.add_argument('--device_ids', help='gpus ids in format: 0/ 0 1/ 0 1 2..', nargs='+', type=int, default=0)


    opt = parser.parse_args()
    scale = opt.size_image
    pos_class = opt.pos_class
    random_images = opt.random_images_download
    num_images = opt.num_images
    opt.num_transforms=opt.num_transforms
    dataset = opt.dataset

    if opt.if_download_class == True and opt.mode == 'train':
        if dataset == 'cifar':
            opt.num_transforms = 54
            opt.input_name = download_class_cifar(opt)
        elif dataset == 'mnist':
            opt.num_transforms = 42
            opt.input_name = download_class_mnist(opt)
        elif dataset == 'fashionmnist':
            opt.num_transforms = 42
            opt.input_name = download_class_FashionMnist(opt)
        elif dataset == 'paris':
            opt.num_transforms = 54
            pass #TODO
        elif dataset == 'mvtec':
            opt.num_transforms = 54
            pass #TODO


    opt = functions.post_config(opt)
    Gs = []
    Zs = {}
    NoiseAmp = {}
    reals_list = torch.FloatTensor(num_images,1,int(opt.nc_im), int(opt.size_image), int(opt.size_image)).cuda()

    for i in range(num_images):
        real = img.imread("%s/%s_%d.png" % (opt.input_dir, opt.input_name[:-4], i))
        if dataset == 'mnist' or dataset == 'fashionmnist':
            real = torch.from_numpy(real).unsqueeze(2)
            real = real.repeat(1, 1, 3).numpy()
        real = functions.np2torch(real, opt)
        real = real[:, 0:3, :, :]
        functions.adjust_scales2image(real, opt)

    dir2save = functions.generate_dir2save(opt)

    reals = {}
    lst =  np.load("TrainedModels/" + str(opt.input_name)[:-4] +  "/transformations.npy")
    opt.list_transformations = lst
    if opt.mode == 'train':
        train(opt, Gs, Zs, reals, NoiseAmp)
    test_size = 10000
    anomaly_detection(opt.input_name, test_size, opt)