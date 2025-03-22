## Pixel-Level Domain Adaptation: A New Perspective for Weakly Supervised Semantic Segmentation

### Overview
The Pytorch implementation of _Pixel-Level Domain Adaptation: A New Perspective for Weakly Supervised Semantic Segmentation._

>Recent attention has been devoted to the pursuit of learning semantic segmentation models exclusively from image tags, a paradigm known as image-level Weakly Supervised Semantic Segmentation (WSSS). Existing attempts adopt the Class Activation Maps (CAMs) as priors to mine object regions yet observe the imbalanced activation issue, where only the most discriminative object parts are located. In this paper, we argue that the distribution discrepancy between the discriminative and the non-discriminative parts of objects prevents the model from producing complete and precise pseudo masks as ground truths. For this purpose, we propose a Pixel-Level Domain Adaptation (PLDA) method to encourage the model in learning pixel-wise domain-invariant features. Specifically, a multi-head domain classifier trained adversarially with the feature extraction is introduced to promote the emergence of pixel features that are invariant with respect to the shift between the source (i.e., the discriminative object parts) and the target (\textit{i.e.}, the non-discriminative object parts) domains. In addition, we come up with a Confident Pseudo-Supervision strategy to guarantee the discriminative ability of each pixel for the segmentation task, which serves as a complement to the intra-image domain adversarial training. Our method is conceptually simple, intuitive and can be easily integrated into existing WSSS methods. Taking several strong baseline models as instances, we experimentally demonstrate the effectiveness of our approach under a wide range of settings.


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


### Acknowledgements
We sincerely thank [Yude Wang](https://scholar.google.com/citations?user=5aGpONMAAAAJ&hl=en) for his great work SEAM in CVPR'20. We borrow codes heavly from his repositories [SEAM](https://github.com/YudeWang/SEAM) and [Segmentation-codebase](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc).
We also thank [Seungho Lee](https://scholar.google.com/citations?hl=zh-CN&user=vUM0nAgAAAAJ) for his [EPS](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf) and [jiwoon-ahn](https://github.com/jiwoon-ahn) for his [PSA](https://github.com/jiwoon-ahn/psa) and [IRN](https://github.com/jiwoon-ahn/irn). Without them, we could not finish this work.

### Citation
```
@article{du2024pixel,
  title={Pixel-Level Domain Adaptation: A New Perspective for Enhancing Weakly Supervised Semantic Segmentation},
  author={Du, Ye and Fu, Zehua and Liu, Qingjie},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```
