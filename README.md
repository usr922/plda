## Pixel-Level Domain Adaptation: A New Perspective for Weakly Supervised Semantic Segmentation

### Overview
The Pytorch implementation of _Pixel-Level Domain Adaptation: A New Perspective for Weakly Supervised Semantic Segmentation._


### Prerequisites
- Python 3.8
- pytorch>=1.8.0
- torchvision
- CUDA>=9.0
- pydensecrf from https://github.com/lucasb-eyer/pydensecrf
- others (opencv-python etc.)


### Data

1. Clone this repository.
2. Data preparation.
   Download PASCAL VOC 2012 devkit following instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit. 
   It is suggested to make a soft link toward downloaded dataset. 
   Then download the annotation of VOC 2012 trainaug set (containing 10582 images) from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0 and place them all as ```VOC2012/SegmentationClassAug/xxxxxx.png```. 
   Download the image-level labels ```cls_label.npy``` from https://github.com/YudeWang/SEAM/tree/master/voc12/cls_label.npy and place it into ```voc12/```, or you can generate it by yourself.
3. Download ImageNet pretrained backbones.
   We use ResNet-38 for initial seeds generation and ResNet-101 for segmentation training. 
   Download pretrained ResNet-38 from https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view.
   The ResNet-101 can be downloaded from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth.


 ### Train
 ``bash plda_train.sh``

 ### Pre-trained Models
 Pre-trained models can be found at [onedrive](https://connectpolyu-my.sharepoint.com/:u:/g/personal/23123041r_connect_polyu_hk/EReECi3Jm6JCnyiAs7WCJnAB_wzHmt29F9PbYtvJTc_XVA?e=OCXcCN).
