
import PIL.Image
import random
import numpy as np


class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, sal=None):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        if sal:
           sal = sal.resize(target_shape, resample=PIL.Image.CUBIC)
           return img, sal
        return img


class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, sal=None):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]
        if sal is not None:
            container_sal = np.zeros((self.cropsize, self.cropsize,1), np.float32)
            container_sal[cont_top:cont_top+ch, cont_left:cont_left+cw,0] = \
                sal[img_top:img_top+ch, img_left:img_left+cw]
            return container, container_sal

        return container

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], PIL.Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, PIL.Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = PIL.Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img, sal=None):
        if bool(random.getrandbits(1)):
            #img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img = np.fliplr(img).copy()
            if sal:
                #sal = sal.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                sal = np.fliplr(sal).copy()
                return img, sal 
            return img
        else:
            if sal:
                return img, sal
            return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(tensor, sal=False):
    if sal:
        tensor = np.expand_dims(tensor, axis=0)
    else:
        tensor = np.transpose(tensor, (2, 0, 1))
    return tensor


class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)




def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))




"""
used for coco, from SIPE repo
"""


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img[0], target_scale, order)

def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return [np.fliplr(m) for m in img]
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images, box

def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container

def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)


def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride


def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)


def colorize_score(score_map, exclude_zero=False, normalize=True, by_hue=False):
    import matplotlib.colors
    if by_hue:
        aranged = np.arange(score_map.shape[0]) / (score_map.shape[0])
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

        test = rgb_color[np.argmax(score_map, axis=0)]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test

        if normalize:
            return test / (np.max(test) + 1e-5)
        else:
            return test

    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                     (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                     (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]

        test = VOC_color[np.argmax(score_map, axis=0)%22]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test
        if normalize:
            test /= np.max(test) + 1e-5

        return test


def colorize_displacement(disp):

    import matplotlib.colors
    import math

    a = (np.arctan2(-disp[0], -disp[1]) / math.pi + 1) / 2

    r = np.sqrt(disp[0] ** 2 + disp[1] ** 2)
    s = r / np.max(r)
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

    return rgb_color


def colorize_label(label_map, normalize=True, by_hue=True, exclude_zero=False, outline=False):

    label_map = label_map.astype(np.uint8)

    if by_hue:
        import matplotlib.colors
        sz = np.max(label_map)
        aranged = np.arange(sz) / sz
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
        rgb_color = np.concatenate([np.zeros((1, 3)), rgb_color], axis=0)

        test = rgb_color[label_map]
    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                              (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]
        test = VOC_color[label_map]
        if normalize:
            test /= np.max(test)

    if outline:
        edge = np.greater(np.sum(np.abs(test[:-1, :-1] - test[1:, :-1]), axis=-1) + np.sum(np.abs(test[:-1, :-1] - test[:-1, 1:]), axis=-1), 0)
        edge1 = np.pad(edge, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        edge2 = np.pad(edge, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        edge = np.repeat(np.expand_dims(np.maximum(edge1, edge2), -1), 3, axis=-1)

        test = np.maximum(test, edge)
    return test

