from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import itertools
import random

def download_class_FashionMnist(opt):
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images
    trainset = datasets.FashionMNIST('./FashionMnist_data', download=True, train=True)

    train_data = np.array(trainset.data)
    train_labels = np.array(trainset.targets)
    train_data = train_data[np.where(train_labels == int(pos_class))]

    path = "FashionMnist_test_scale" + str(scale) + "_" + str(pos_class)
    if os.path.exists(path) == False:
        print("path not exists")
        os.mkdir(path)
        print("download data...")

        FashionMnist_testset = datasets.MNIST('./FashionMnist_data', download=True, train=False)
        test_data = np.array(FashionMnist_testset.data)
        test_labels = np.array(FashionMnist_testset.targets)
        test_data = torch.from_numpy(test_data).unsqueeze(3)
        test_data = test_data.repeat(1, 1, 1, 3).numpy()
        test_data = test_data.transpose((0, 3, 1, 2)) / 255

        test_data = np.array(test_data)
        FashionMnist_target_new = np.zeros((test_labels.shape))
        FashionMnist_target_new[test_labels == int(pos_class)] = 1
        FashionMnist_target_new[test_labels != int(pos_class)] = 0

        np.save(path + "/fashionmnist_data_test_" + str(pos_class) + str(scale) + ".npy", test_data)
        np.save(path + "/fashionmnist_labels_test_" + str(pos_class) + str(scale) + ".npy", FashionMnist_target_new)

    def imsave(img, i):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((scale, scale))])
        im = transform(img)
        im.save("Input/Images/FashionMnist_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")


    count_images,step_index = 0,0
    for i in range(len(train_data)):
        t = train_data[i]
        imsave(t, count_images)
        count_images += 1
        if count_images == num_images: step_index +=1
        if step_index == opt.index_download: break
        if count_images == num_images and step_index != opt.index_download: count_images=0
    opt.input_name = "FashionMnist_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,))
    genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3))
    genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,))
    genertator = itertools.chain(genertator0, genertator1, genertator3)
    lst = list(genertator)
    random.shuffle(lst)
    path_transform = "TrainedModels/" + str(opt.input_name)[:-4]
    if os.path.exists(path_transform) == False:
        os.mkdir(path_transform)

    np.save(path_transform + "/transformations.npy", lst)
    return opt.input_name
