from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import itertools
import random

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def download_class_cifar(opt):
    opt.input_name = "cifar_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images
    name_to_label = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3,
                     'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    true_label = name_to_label[pos_class]
    def imsave(img, i):
        # im = Image.fromarray(img.astype(np.uint8)).resize((scale,scale))
        im = Image.fromarray(img.astype(np.uint8)).resize((scale,scale))
        im.save("Input/Images/cifar_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")
        im = cv2.imread("Input/Images/cifar_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")

        H, S, V = cv2.split(cv2.cvtColor((im), cv2.COLOR_RGB2HSV))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_V = clahe.apply(V)
        im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        cv2.imwrite("Input/Images/cifar_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png", im)

    if opt.mode == "train":
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=True)
        train_data = np.array(trainset.data)
        train_labels = np.array(trainset.targets)
        train_data = train_data[np.where(train_labels == true_label)]
        dicty = {}
        if opt.random_images_download == False:
            count_images,step_index = 0,0
            for i in range(len(train_data)):
                t = train_data[i]
                imsave(t, count_images)
                dicty[count_images] = i
                count_images += 1
                if count_images == num_images: step_index +=1
                if step_index == opt.index_download: break
                if count_images == num_images and step_index != opt.index_download: count_images=0
            training_images = list(dicty.values())

        else:
            random_index = random.sample(range(0, len(train_data)), opt.num_images)
            training_images = list(random_index)
            for i in range(len(training_images)):
                index = training_images[i]
                t = train_data[index]
                imsave(t,i)
        print("training imgaes: ", training_images)

        genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,))
        genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3))
        genertator2 = itertools.product((1,), (False, True), (0,), (0,), (0, 1, 2, 3))
        genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,))
        genertator4 = itertools.product((1,), (False,), (1, -1), (0,), (0,))
        genertator5 = itertools.product((1,), (False,), (0,), (1, -1), (0,))
        genertator = itertools.chain(genertator0, genertator1, genertator2, genertator3, genertator4, genertator5)
        lst = list(genertator)
        random.shuffle(lst)
        path_transform = "TrainedModels/" + str(opt.input_name)[:-4]
        if os.path.exists(path_transform) == False:
            os.mkdir(path_transform)
        np.save(path_transform +  "/transformations.npy", lst)

    path = "cifar_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path) == False:
        os.mkdir(path)
    cifar_testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False, download=True)
    test_data = cifar_testset.data
    test_labels = np.array(cifar_testset.targets)
    test_data = test_data.transpose((0, 3, 1, 2)) / 255
    test_data = torch.from_numpy(test_data)
    test_data = np.array(test_data)
    cifar_target_new = np.zeros((test_labels.shape))
    cifar_target_new[test_labels == true_label] = 1
    cifar_target_new[test_labels != true_label] = 0
    np.save(path + "/cifar_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", test_data)
    np.save(path + "/cifar_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", cifar_target_new)

    opt.input_name = "cifar_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    return opt.input_name
