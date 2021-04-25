# A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection

[Project]() | [Arxiv]() |  
### Official pytorch implementation of the paper: "A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection"

## Abstract 

Anomaly detection, the task of identifying unusual samples in data, 
often relies on a large set of training samples. 
In this work, we consider the setting of few-shot anomaly detection in images, where only a few images are given at training. We devise a hierarchical generative model that captures the multi-scale patch distribution of each training image. We further enhance the representation of our model by using image transformations and optimize scale-specific patch-discriminators to  distinguish between real and fake patches of the image, as well as between different transformations applied to those patches. The anomaly score is obtained by aggregating the patch-based votes of the correct transformation across scales and image regions. We demonstrate the superiority of our method on both the one-shot and few-shot settings, on the datasets of Paris, CIFAR10, MNIST and FashionMNIST as well as in the setting of defect detection on MVTec. In all cases, our method outperforms the recent baseline methods.

![](imgs/teaser.PNG)


## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train SinGAN model on your own image, put the desired training image under Input/Images, and run

```
python main_train.py --input_name <input_file_name>
```

This will also use the resulting trained model to generate random samples starting from the coarsest scale (n=0).

To run this code on a cpu machine, specify `--not_cuda` when calling `main_train.py`


##  Applications
The model can be also used for defect detection:
 ![](imgs/manipulation.PNG)
See section 3.3 in our [paper]() for more details.

For additional details please see section 3.1 in our [paper](https://arxiv.org/pdf/1905.01164.pdf)

### Citation
If you use this code for your research, please cite our paper:

```

