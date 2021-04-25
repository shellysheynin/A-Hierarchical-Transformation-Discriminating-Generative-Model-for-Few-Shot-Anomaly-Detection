from FewShot_models.training_parallel import *
import FewShot_models.functions as functions
import os
import itertools
import torch
from Dataloaders.mvtec_loader_just_rotate import download_class_mvtec
from defect_detection_evaluation import defect_detection


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
    parser.add_argument('--test_size', help='test size', type=int, default=10000)
    parser.add_argument('--num_transforms', help='54 for rgb, 42 for grayscale', type=int, default=4)
    parser.add_argument('--device_ids', help='gpus ids in format: 0/ 0 1/ 0 1 2..', nargs='+', type=int, default=0)
    parser.add_argument('--fraction_defect', help='fraction of patches to consider in each scale',type=float, default=0.1)


    opt = parser.parse_args()
    scale = opt.size_image
    pos_class = opt.pos_class
    random_images = opt.random_images_download
    num_images = opt.num_images
    opt.num_transforms=opt.num_transforms
    dataset = opt.dataset

    if opt.if_download_class == True:
        if dataset == 'mvtec':
            opt.num_transforms = 4
            opt.input_name = download_class_mvtec(opt)
        else:
            print("this file is just for mvtec dataset")
            exit()

    opt = functions.post_config(opt)
    Gs = []
    Zs, NoiseAmp = {}, {}
    reals_list = torch.FloatTensor(num_images, 1, int(opt.nc_im), int(opt.size_image), int(opt.size_image)).cuda()


    for i in range(num_images):

        real = img.imread("%s/%s_%d.png" % (opt.input_dir, opt.input_name[:-4], i))
        real = functions.np2torch(real, opt)
        real = real[:, 0:3, :, :]
        functions.adjust_scales2image(real, opt)

    dir2save = functions.generate_dir2save(opt)
    reals = {}
    genertator = itertools.product((0,),(False,),(0,),(0,),(0, 1, 2, 3))
    lst = list(genertator)
    opt.list_transformations = lst
    print(opt.list_transformations)
    print("num transformations: ", len(lst))

    if opt.mode == 'train':
        train(opt, Gs, Zs, reals, NoiseAmp)
    if dataset == 'mvtec':
        defect_detection(opt.input_name, opt.test_size, opt)
    else:
        print("this file is just for mvtec dataset")
        exit()
