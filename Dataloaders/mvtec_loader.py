from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import random
import itertools
import tarfile
from tqdm import tqdm
import urllib.request
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
           'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
           'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)



class MVTecDataset(Dataset):
    def __init__(self, root_path='../data', class_name='bottle', is_train=True,
                 resize=128, cropsize=128):
        assert class_name in classes, 'class_name: {}, should be in {}'.format(class_name, classes)
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join('mvtec_anomaly_detection')
        self.download()
        self.x, self.y = self.load_dataset_folder()
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)


        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([1] * len(img_fpath_list))
            else:
                y.extend([0] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)

    def download(self):
        """Download dataset if not exist"""
        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    os.mkdir("mvtec_anomaly_detection")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)



def download_class_mvtec(opt):
    opt.input_name = "mvtec_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images

    def imsave(img, i):
        # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((scale, scale))])
        # im = transform(img)
        img = (img) * 255
        npimg = img.numpy().astype(np.uint8)
        npimg = np.transpose(npimg, (1, 2, 0))
        im = Image.fromarray(npimg)
        # im = Image.fromarray(img.astype(np.uint8)).resize((scale,scale))
        im.save("Input/Images/mvtec_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")
        im = cv2.imread("Input/Images/mvtec_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")
        H, S, V = cv2.split(cv2.cvtColor((im), cv2.COLOR_RGB2HSV))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_V = clahe.apply(V)
        im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        im = cv2.resize(im, (scale,scale), interpolation=cv2.INTER_AREA)

        cv2.imwrite("Input/Images/mvtec_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png", im)

    if opt.mode == "train":
        trainset = MVTecDataset(class_name=opt.pos_class, is_train=True)  # images of size 224, 224
        trainloader = DataLoader(trainset, batch_size=len(trainset), pin_memory=True)
        dataiter = iter(trainloader)
        images, _ = dataiter.next()
        dicty = {}
        if opt.random_images_download == False:
            count_images,step_index = 0,0
            for i in range(images.shape[0]):
                t = images[i]
                imsave(t, count_images)
                dicty[count_images] = i
                count_images += 1
                if count_images == num_images: step_index +=1
                if step_index == opt.index_download: break
                if count_images == num_images and step_index != opt.index_download: count_images=0
            training_images = list(dicty.values())

        else:
            random_index = random.sample(range(0, images.shape[0]), opt.num_images)
            training_images = list(random_index)
            for i in range(len(training_images)):
                index = training_images[i]
                t = images[index]
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
        np.save(path_transform + "/transformations.npy", lst)

    mvtec_testset = MVTecDataset(class_name=pos_class, is_train=False)
    mvtec_loader = DataLoader(mvtec_testset, batch_size=len(mvtec_testset), pin_memory=True)
    (mvtec_data, mvtec_targets) = next(iter(mvtec_loader))
    mvtec_target = mvtec_targets.numpy()
    mvtec_data = mvtec_data.numpy()
    path_test = "mvtec_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path_test) == False:
        os.mkdir(path_test)
    np.save(path_test + "/mvtec_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", mvtec_data)
    np.save(path_test + "/mvtec_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", mvtec_target)

    opt.input_name = "mvtec_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
    + "_indexdown" + str(opt.index_download) + ".png"
    return opt.input_name
