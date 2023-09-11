# <p align=center> This repo supplements our [Survey on deep learning models for detection of COVID-19](https://link.springer.com/article/10.1007/s00521-023-08683-x), which has been published in Neural Computing and Applications (NCAA).
Authors: [Javad Mozaffari](), [Abdollah Amirkhani](), and [Shahriar B Shokouhi]().
</p>

# <p align=center>`Awesome-COVID19-research </p>
![review](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/2a7e7bf1-4eb6-4a3f-b7c1-f5f73a5d0c18)

## Overview
- [Machine learning vs deep learning](#ml-vs-dl)
- [Datasets](#datasets)
- [VGGNet](#vggnet)
- [ResNet](#resnet)
- [Custom models](#custom-models)
- [DenseNet](#densenet)
- [CapsuleNet](#capsulenet)
- [MobileNet](#mobilenet)
- [EfficientNet](#efficientnet)
- [Citation](#citation)


# ML vs DL
-
![MlvsDl2](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/c1892d25-0e57-41d8-bc9d-bd01f754fed4)


# Datasets 

|  Reference  |   Year  |   Image type   |   No. of cases    |   URL   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|  Cohen et al. [129] | 2020  | X-ray  | COVID-19: 468, non-COVID-19: 84  |https://github.com/ieee8023/covid-chestxray-dataset					
|Bell [130]	|2020|	CT, X-ray|	COVID-19: 101|	https://radiopaedia.org/articles/covid-19-3
|Afshar et al. [87]|	2020|	CT	|COVID-19: 169, non-COVID-19: 136|	https://figshare.com/s/c20215f3d42c98f09ad0
|SIRM COVID-19 database [131]	|2020|	CT|	COVID-19: 115|	https://sirm.org/category/senza-categoria/covid-19
|Chowdhury et al. [132]	|2020	|X-ray|	COVID-19: 423, non-COVID-19: 2686|	https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
|Soares et al. [133]|	2020|	CT|	COVID-19: 1252, non-COVID-19: 1230	|https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
|Zhao et al. [134]|	2020	|CT|	COVID-19: 349|	https://github.com/UCSD-AI4H/COVID-CT
|Wang et al. [135]	|2020	|X-ray	|COVID-19: 589, non-COVID-19: 14414	|https://github.com/lindawangg/COVID-Net
|Sait et al. [136]	|2020	|X-ray|	COVID-19: 1281, non-COVID-19: 7927	|https://data.mendeley.com/datasets/9xkhgts2s6/1
|Winther et al. [137]	|2020|	X-ray|	COVID-19: 243|	https://github.com/ml-workgroup/covid-19-image-repository
|Nair et al. [138]|	2021|	X-ray	|COVID-19: 900|	https://github.com/armiro/COVID-CXNet/tree/master/chest_xray_images/covid19
|Jenssen and Sakinis [139]|	2021|	CT|	COVID-19: 110, Lung segmentation	|http://medicalsegmentation.com/covid19
|Ginneken etl al. [140]	|2006|	X-ray	|Lung segmentation	|https://www.isi.uu.nl/Research/Databases/SCR/index.php
|Qiblawey et al. [141]|	2021	|X-ray|	COVID-19:423, non-COVID-19:278|	https://www.kaggle.com/yazanqiblawey/sars-mers-xray-images-dataset/version/3
|Stirenko et al. [142]|	2018|	X-ray|	Lung segmentation: 566|	https://www.kaggle.com/yoctoman/shcxr-lung-mask
|Alqudah and Qazan [143]|	2020|	X-ray|	COVID: 912, non-COVID: 912|	https://data.mendeley.com/datasets/2fxz4px6d8/4
|Malik et al. [144]|	2020|	X-ray|	COVID: 44, non-COVID: 27|	https://data.mendeley.com/datasets/67dmnmx33v/1
|Walid et al. [145]|	2020|	X-ray, CT|	COVID: 9471, non-COVID: 8128|	https://data.mendeley.com/datasets/8h65ywd2jr/3
|Patel [146]	|2020	|X-ray|	COVID: 575|	https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia


# VGGNet
- [Very deep convolutional networks for large-scale image recognition] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Very+deep+convolutional+networks+for+large-scale+image+recognition&btnG=) [[arXiv]](https://arxiv.org/abs/1409.1556) [[code]](https://pytorch.org/hub/pytorch_vision_vgg/)
![vgg](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/661bed0e-059e-4c25-b8bf-b502b827ce37)

|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Modified Vgg Deep Learning Architecture For Covid-19 Classification Using Bio-Medical Image](https://journals.lww.com/BBRJ/Fulltext/2021/05010/Modified_VGG_Deep_Learning_Architecture_for.8.aspx) <br> | 2021 | X-ray | 98% |  4 class
|<br> [A deep learning based approach for automatic detection of COVID-19 cases using chest X-ray images](https://www.sciencedirect.com/science/article/pii/S1746809421007795) <br> | 2022 | X-ray | 96.6% |  3 class
|<br> [BDCNet: multi-classification convolutional neural network model for classification of COVID-19, pneumonia, and lung cancer from chest radiographs](https://link.springer.com/article/10.1007/s00530-021-00878-3) <br> | 2022 | X-ray | 98.33% | 4 class
|<br> [Coronavirus covid-19 detection by means of explainable deep learning](https://www.nature.com/articles/s41598-023-27697-y) <br> | 2023 | CT |  95% | 2 class

# GoogleNet 
- [Going Deeper With Convolutions] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Going+Deeper+With+Convolutions&btnG=) [[arXiv]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) [[code]](https://pytorch.org/hub/pytorch_vision_googlenet/)
![googlenet](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/c94daa67-7657-47dc-96ab-b989cc4ba098)


|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Detection of COVID-19 by GoogLeNet-COD](https://link.springer.com/chapter/10.1007/978-3-030-60799-9_43) <br> | 2020 | CT | 87.5% |  2 class
|<br> [Deep Learning for Reliable Classification of COVID-19, MERS, and SARS from Chest X-ray Images](https://link.springer.com/article/10.1007/s12559-021-09955-1) <br> | 2022 | X-ray | 98.2% | 3 class
|<br> [Application of CycleGAN and transfer learning techniques for automated detection of COVID-19 using X-ray images](https://www.sciencedirect.com/science/article/pii/S0167865521004128) <br> | 2022 | X-ray | 94.2% | 2 class
|<br> [Application of CycleGAN and transfer learning techniques for automated detection of COVID-19 using X-ray images](https://www.sciencedirect.com/science/article/pii/S0167865521004128) <br> | 2021 | X-ray | 98% |  3 class

# ResNet 
- [Deep Residual Learning for Image Recognition] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+Residual+Learning+for+Image+Recognition&btnG=) [[arXiv]](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) [[code]](https://pytorch.org/hub/pytorch_vision_resnet/)
![googlenet](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/8dc40258-4428-4deb-804e-7f4a94a33e19)


|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Automatic Detection of Coronavirus Disease (COVID-19) Using X-ray Images and Deep Convolutional Neural Networks](https://link.springer.com/article/10.1007/s10044-021-00984-y) <br> | 2021 | X-ray | 99.7% | 2 class
|<br> [Experiments of Federated Learning for COVID-19 Chest X-ray Images](https://arxiv.org/abs/2007.05592) <br> | 2021 | X-ray | 91.26% |  3 class
|<br> [CGENet: A Deep Graph Model for COVID-19 Detection Based on Chest CT](https://www.mdpi.com/2079-7737/11/1/33) <br> | 2022 | CT |97.78% |  2 class
|<br> [Deep learning for COVID-19 detection based on CT images](https://www.nature.com/articles/s41598-021-93832-2)) <br> | 2021 | CT |99.2% |  3 class
|<br> [ResGNet-C: A graph convolutional neural network for detection of COVID-19](https://www.sciencedirect.com/science/article/pii/S0925231220319184) <br> | 2021 | CT |96.62% | 2 class


# Custom models
|  Title  |   Year  |   Image type   |   Accuracy       | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Prediction of COVID-19 - Pneumonia based on Selected Deep Features and One Class Kernel Extreme Learning Machine](https://www.sciencedirect.com/science/article/pii/S0045790620308065) <br> | 2021 | CT |95.1%  | 2 class
|<br> [PSSPNN: PatchShuffle Stochastic Pooling Neural Network for an explainable diagnosis of COVID-19 with multiple-way data augmentation](https://www.hindawi.com/journals/cmmm/2021/6633755/) <br> | 2021 | CT |95.79%  | 4 class
|<br> [A seven-layer convolutional neural network for chest CT based COVID-19 diagnosis using stochastic pooling](https://ieeexplore.ieee.org/abstract/document/9203857) <br> | 2020 | CT |94.03%  | 2 class
|<br> [A five-layer deep convolutional neural network with stochastic pooling for chest CT-based COVID-19 diagnosis](https://link.springer.com/article/10.1007/s00138-020-01128-8) <br> | 2020 | CT |93.64% |  2 class
# DenseNet 
- [Densely Connected Convolutional Networks] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Densely+Connected+Convolutional+Networks&btnG=) [[arXiv]](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) [[code]](https://pytorch.org/hub/pytorch_vision_densenet/)
![densenet](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/bcea2bfc-9c59-418a-aa2b-40bfd2c09f54)

|  Title  |   Year  |   Image type   |   Accuracy   | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [COVIDX-Net: A Framework of Deep Learning Classifiers to Diagnose COVID-19 in X-Ray Images](https://arxiv.org/abs/2003.11055) <br> | 2020 | X-ray |90%  | 2 class
|<br> [Cascaded deep learning classifiers for computer-aided diagnosis of COVID-19 and pneumonia diseases in X-ray scans](https://arxiv.org/abs/2003.11055) <br> | 2021 | X-ray |99.9%  | 2 class
|<br> [DenseNet Convolutional Neural Networks Application for Predicting COVID-19 Using CT Image](https://link.springer.com/article/10.1007/s42979-021-00782-7) <br> | 2021 | CT |92%  | 2 class
|<br> [A Two-Dimensional Sparse Matrix Profile DenseNet for COVID-19 Diagnosis Using Chest CT Images](https://ieeexplore.ieee.org/abstract/document/9268138) <br> | 2020 | CT |80%  | 2 class
|<br> [COVID-19 Diagnosis via DenseNet and Optimization of Transfer Learning Setting](https://link.springer.com/article/10.1007/s12559-020-09776-8) <br> | 2021 | CT |96.3%  | 2 class
|<br> [A novel data augmentation based on Gabor filter and convolutional deep learning for improving the classification of COVID-19 chest X-Ray images](https://www.sciencedirect.com/science/article/pii/S174680942100923X) <br> | 2022 | X-ray |98.5% | 2 class
|<br> [Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1) <br> | 2021 | CT |86% | 2 class
|<br> [Covid-densenet: A deep learning architecture to detect covid-19 from chest radiology images](https://link.springer.com/chapter/10.1007/978-981-19-6634-7_28) <br> | 2023 | X-ray |96.49 %  | 3 class
|<br> [COVID-19 diagnosis using state-of-the-art CNN architecture features and Bayesian Optimization](https://www.sciencedirect.com/science/article/pii/S0010482522000361) <br> | 2022 | X-ray |96.29% | 3 class
|<br> [Diagnosing Covid-19 chest x-rays with a lightweight truncated DenseNet with partial layer freezing and feature fusion](https://www.sciencedirect.com/science/article/pii/S1746809421001804) <br> | 2021 | X-ray |97.99%  | 3 class
|<br> [COVID-CXNet: Detecting COVID-19 in frontal chest X-ray images using deep learning,](https://link.springer.com/article/10.1007/s11042-022-12156-z) <br> | 2022 | X-ray |87.88%  | 3 class
|<br> [COVID-19 classification by CCSHNet with deep fusion using transfer learning and discriminant correlation analysis](https://www.sciencedirect.com/science/article/pii/S1566253520304073) <br> | 2021 | CT |97.4%  | 3 class
# CapsuleNet 
- [Dynamic Routing Between Capsules] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Dynamic+Routing+Between+Capsules&btnG=) [[arXiv]](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html) [[code]](https://github.com/jindongwang/Pytorch-Capsule)
![capsulenet](https://github.com/javadmozaffari/Awesome-COVID19-research/assets/47658906/e751e9fa-217c-4bde-8831-eff99c515deb)

|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [COVID-CAPS: A capsule network-based framework for identification of COVID-19 cases from X-ray images](https://arxiv.org/abs/2003.11055) <br> | 2020 | X-ray |98.3%| 2 class
|<br> [Ct-Caps: Feature Extraction-Based Automated Framework for Covid-19 DiseaseIdentification From Chest Ct Scans Using Capsule Networks](https://ieeexplore.ieee.org/abstract/document/9414214) <br> | 2021 | CT |90.8% | 2 class
|<br> [A lightweight capsule network architecture for detection of COVID-19 from lung CT scans](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.22706) <br> | 2022 | CT |99%  | 2 class
|<br> [Convolutional capsnet: A novel artificial neural network approach to detect COVID-19 disease from X-ray images using capsule networks](https://www.sciencedirect.com/science/article/pii/S0960077920305191) <br> | 2020 | X-ray |97.23% | 2 class
|<br> [Convolutional capsule network for COVID‐19 detection using radiography images](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.22566) <br> | 2021 | X-ray |97%  | 2 class
|<br> [Human-level COVID-19 Diagnosis from Low-dose CT Scans Using a Two-stage Time-distributed Capsule Network](https://www.nature.com/articles/s41598-022-08796-8) <br> | 2021 | CT |94.1%  | 3 class
|<br> [DenseCapsNet: Detection of COVID-19 from X-ray images using a capsule neural network](https://www.sciencedirect.com/science/article/pii/S0010482521001931) <br> | 2021 | X-ray |90.7%  | 3 class
|<br> [The application of fast CapsNet computer vision in detecting Covid-19](https://www.researchgate.net/profile/Raj-Sandu/publication/342699125_The_application_of_fast_CapsNet_computer_vision_in_detecting_Covid-19/links/5f01b753299bf18816037ea0/The-application-of-fast-CapsNet-computer-vision-in-detecting-Covid-19.pdf) <br> | 2020 | CT |92%  | 3 class
# MobileNet
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=MobileNets%3A+Efficient+Convolutional+Neural+Networks+for+Mobile+Vision+Applications&btnG=) [[arXiv]](https://arxiv.org/abs/1704.04861) [[code]](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)

|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Detection of Covid-19 Patients using Chest X-ray images with Convolution Neural Network and Mobile Net](https://ieeexplore.ieee.org/abstract/document/9316100) <br> | 2020 | X-ray |98.6%  | 2 class
|<br> [COVID-19 detection from chest x-ray using MobileNet and residual separable convolution block](https://link.springer.com/article/10.1007/s00500-021-06579-3) <br> | 2022 | X-ray |99.71%  | 2 class
|<br> [Covid-19: automatic detection from X-ray images utilizing transfer learning with convolutional neural networks](https://link.springer.com/article/10.1007/s13246-020-00865-4) <br> | 2020 | X-ray |98.75% |  2 class
|<br> [Extracting Possibly Representative COVID-19 Biomarkers from X-ray Images with Deep Learning Approach and Image Data Related to Pulmonary Diseases](https://link.springer.com/article/10.1007/s40846-020-00529-4) <br> | 2020 | X-ray | 99.18% |  2 class
|<br> [RAM-Net: A Residual Attention MobileNet to Detect COVID-19 Cases from Chest X-Ray Images](https://ieeexplore.ieee.org/abstract/document/9356348) <br> | 2020 | X-ray |95.3% | 3 class
|<br> [KL-MOB Automated Covid-19 Recognition Using a Novel Approach Based on Image Enhancement and a Modified MobileNet CNN](https://peerj.com/articles/cs-694/) <br> | 2021 | X-ray |98.7% | 3 class
|<br> [Fast COVID-19 Detection of Chest X-Ray Images Using Single Shot Detection MobileNet Convolutional Neural Networks](https://jsju.org/index.php/journal/article/view/846) <br> | 2021 | X-ray |94% | 3 class
# EfficientNet 
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks] [[scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=EfficientNet%3A+Rethinking+Model+Scaling+for+Convolutional+Neural+Networks&btnG=) [[arXiv]](https://proceedings.mlr.press/v97/tan19a.html?ref=jina-ai-gmbh.ghost.io) [[code]](https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/)

|  Title  |   Year  |   Image type   |   Accuracy     | Number of classes |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [Towards an effective and efficient deep learning model for COVID-19 patterns detection in X-ray images](https://link.springer.com/article/10.1007/s42600-021-00151-6) <br> | 2021 | X-ray |93.9%  | 2 class
|<br> [COVID-19 detection in CT images with deep learning: A voting-based scheme and cross-datasets analysis](https://www.sciencedirect.com/science/article/pii/S2352914820305773) <br> | 2020 | CT |87.6% | 2 class
|<br> [A complete framework for accurate recognition and prognosis of COVID-19 patients based on deep transfer learning and feature classification approach](https://link.springer.com/article/10.1007/s10462-021-10127-8) <br> | 2020 | X-ray |99.61% | 2 class
|<br> [ECOVNet: An Ensemble of Deep Convolutional Neural Networks Based on EfficientNet to Detect COVID-19 From Chest X-rays](https://arxiv.org/abs/2009.11850) <br> | 2020 | X-ray |96% | 2 class
|<br> [Efficient-CovidNet: Deep Learning Based COVID-19 Detection From Chest X-Ray Images](https://ieeexplore.ieee.org/abstract/document/9398980) <br> | 2020 | X-ray |95% |  3 class
|<br> [Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT scans](https://arxiv.org/abs/2009.06412) <br> | 2020 | X-ray |93.18% | Segmentation

# Citation

If you find the listing and survey useful for your work, please cite the paper:

```
{
title={A survey on deep learning models for detection of COVID-19}, 
      author={Mozaffari, Javad and Amirkhani, Abdollah and Shokouhi B, Shahriar}
      year={2023},
      Journal={Neural Computing and Applications}
}
```
