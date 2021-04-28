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

def download_class_paris(opt):
    opt.input_name = "paris_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images


    def imsave(img, i):
        cv2.imwrite("Input/Images/paris_train_numImages_"  + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) +
                    "_indexdown" + str(opt.index_download) + "_" + str(i) + ".png", img)
        im = cv2.imread("Input/Images/paris_train_numImages_"  + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) +
                    "_indexdown" + str(opt.index_download) + "_" + str(i) + ".png")
        H,S,V= cv2.split(cv2.cvtColor((im), cv2.COLOR_RGB2HSV))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_V = clahe.apply(V)
        im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        im = cv2.resize(im, (scale,scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("Input/Images/paris_train_numImages_"  + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) +
                    "_indexdown" + str(opt.index_download) + "_" + str(i) + ".png", im)

    count_images,step_index = 0,0
    dicty = {}
    path_lines_ok = "Paris/lab/" + str(pos_class) + "_ok.txt" # take images from this file
    with open(path_lines_ok, "r") as f:
        lines_ok= f.readlines()
        if opt.random_images_download == False:
            for line in lines_ok:
                path = line.strip("\n")
                t = cv2.imread("Paris/jpg/1/" + str(path) + ".jpg")
                imsave(t, count_images)
                dicty[count_images] = path
                count_images += 1
                if count_images == num_images: step_index +=1
                if step_index == opt.index_download: break
                if count_images == num_images and step_index != opt.index_download: count_images=0
        else:
            random_index = random.sample(range(0, len(lines_ok)), opt.num_images)
            training_images = list(random_index)
            count_images = 0
            for i, line in enumerate(lines_ok):
                if i in training_images:
                    path = line.strip("\n")
                    t = cv2.imread("Paris/jpg/1/" + str(path) + ".jpg")
                    imsave(t, count_images)
                    dicty[count_images] = path
                    count_images += 1
                    if count_images == num_images: step_index += 1
    training_images = list(dicty.values())
    print("training imgaes: ", training_images)


    if opt.mode == "train":

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

        np.save(path_transform + "/transformations.npy", lst)

    path =  "paris_test_scale" + str(scale) +"_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path) == False:
        os.mkdir(path)
    transform = transforms.Compose(
        [torchvision.transforms.Resize((scale, scale)),
         transforms.ToTensor()])
    paris_testset = torchvision.datasets.ImageFolder(root='Paris/jpg',
                                                         transform=transform)
    paris_loader = torch.utils.data.DataLoader(dataset=paris_testset,
                                             batch_size=len(paris_testset),
                                             shuffle=False)
    (paris_data, i) = next(iter(paris_loader))
    paris_data = paris_data.numpy()
    path_lines_ok = "Paris/lab/" + str(pos_class) + "_ok.txt"
    with open(path_lines_ok, "r") as f:
        lines_ok = f.readlines()
    paris_target_new = []
    paris_data_new = []
    for i in range(paris_data.shape[0]):
        sample_fname, _ = paris_loader.dataset.samples[i]
        sample_fname = sample_fname[12:-4]
        if sample_fname in training_images:
            continue
        in_normal_class = False
        for line_ok in lines_ok:
            if sample_fname in line_ok.strip("\n"):
                in_normal_class = True
        paris_data_new.append(paris_data[i])
        if in_normal_class == True:
            paris_target_new.append(int(1))
        else:
            paris_target_new.append(int(0))
    paris_data_new = np.stack(paris_data_new)
    paris_target_new = np.stack(paris_target_new)

    np.save(path + "/" +  "paris_data_test_" + str(pos_class) + str(scale)+  "_" + str(opt.index_download) + ".npy" , paris_data_new)
    np.save(path + "/"+  "paris_labels_test_" + str(pos_class) + str(scale)+  "_" + str(opt.index_download) +".npy" , paris_target_new)

    opt.input_name = "paris_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    opt.size_image = 450


    return opt.input_name
