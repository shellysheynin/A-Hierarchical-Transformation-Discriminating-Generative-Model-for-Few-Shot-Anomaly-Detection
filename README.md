# A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection

[Project]() | [Arxiv]() |  
### Official pytorch implementation of the paper: "A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection"

## Abstract 

Anomaly detection, the task of identifying unusual samples in data, 
often relies on a large set of training samples. 
In this work, we consider the setting of few-shot anomaly detection in images, where only a few images are given at training. We devise a hierarchical generative model that captures the multi-scale patch distribution of each training image. We further enhance the representation of our model by using image transformations and optimize scale-specific patch-discriminators to  distinguish between real and fake patches of the image, as well as between different transformations applied to those patches. The anomaly score is obtained by aggregating the patch-based votes of the correct transformation across scales and image regions. We demonstrate the superiority of our method on both the one-shot and few-shot settings, on the datasets of Paris, CIFAR10, MNIST and FashionMNIST as well as in the setting of defect detection on MVTec. In all cases, our method outperforms the recent baseline methods.

![](Images/diagram2.png)


## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train the model on mvtec/paris/cifar/mnist/fashionMnist:

![](Images/paris_results.jpg)
python main_train.py  --num_images 1  --pos_class <normal_class_in_dataset> --index_download <index_of_training_image> --dataset <name_of_dataset>


##  Applications
The model can be also used for defect detection:

<img src="Images/mvtec_results.png" width="500px">

See section 3.3 in our [paper]() for more details.


### Citation
If you use this code for your research, please cite our paper:

```

