import os
import collections

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from mxnet.gluon.data import Dataset
from mxnet.gluon.loss import Loss
from img_utils import get_mask_fn


class ImageWithMaskDataset(Dataset):
    """
    A dataset for loading images (with masks).
    Based on: mxnet.incubator.apache.org/tutorials/python/data_augmentation_with_masks.html

    :param imgdir: str Path to folder containing the images, relative to root
    :param maskdir: str Path to folder containing the masks/ground truth, relative to root
    :param multisp: bool Flag to enable multispectral image handling
    :param transform_fn : callable, default None A function that takes data and label and transforms them:
    ::
        transform_fn = lambda data, label: (data.astype(np.float32)/255, label)
    """

    def __init__(self, imgdir, maskdir, multisp=False, transform_fn=None):
        self._imgdir = os.path.expanduser(imgdir)
        self._maskdir = os.path.expanduser(maskdir)
        self._transform = transform_fn
        self._multisp = multisp
        self._exts = ['.png'] if not multisp else ['.npy']
        self._geopedia_layers = {'tulip_field_2016': 'ttl1904', 'tulip_field_2017': 'ttl1905'}
        self._list_images(self._imgdir)

    def _list_images(self, root):
        images = collections.defaultdict(dict)
        for filename in sorted(os.listdir(root)):
            name, ext = os.path.splitext(filename)
            mask_flag = "geopedia" in name
            if ext.lower() not in self._exts:
                continue
            if not mask_flag:
                mask_fn = get_mask_fn(filename)
                images[name]["base"] = filename
                images[name]["mask"] = mask_fn
        self._image_list = list(images.values())

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for: " + self._image_list[idx]["mask"]
        base_filepath = os.path.join(self._imgdir, self._image_list[idx]["base"])
        assert 'mask' in self._image_list[idx], "Couldn't find mask image for: " + self._image_list[idx]["base"]
        mask_filepath = os.path.join(self._maskdir, self._image_list[idx]["mask"])

        if self._multisp:
            base = np.load(base_filepath)
        else:
            base = mx.image.imread(base_filepath)

        mask = mx.image.imread(mask_filepath, flag=0)

        if self._transform is not None:
            return self._transform(base, mask, self._multisp)
        else:
            return base, mask

    def __len__(self):
        return len(self._image_list)


class DiceCoeffLoss(Loss):
    """
    Soft dice coefficient loss.
    Based on https://github.com/Lasagne/Recipes/issues/99

    :param pred: (batch size, c, w, h) network output, must sum to 1 over c channel (such as after softmax)
    :param label:(batch size, c, w, h) one hot encoding of ground truth
    :param eps: smoothing factor to avoid division by zero
    :return Loss tensor with shape (batch size)
    """

    def __init__(self, eps=1e-7, _weight=None, _batch_axis=0, **kwards):
        Loss.__init__(self, weight=_weight, batch_axis=_batch_axis, **kwards)
        self.eps = eps

    def hybrid_forward(self, F, label, pred):
        # One-hot encode the label
        label = nd.concatenate([label != 1, label], axis=1)

        axes = tuple(range(2, len(pred.shape)))
        intersect = nd.sum(pred * label, axis=axes)
        denom = nd.sum(pred + label, axis=axes)
        return - (2. * intersect / (denom + self.eps)).mean(axis=1)


class IouMetric(mx.metric.EvalMetric):
    """
    Stores a moving average of the intersection over union metric
    """

    def __init__(self, axis=[2, 3], smooth=1e-7):
        super(IouMetric, self).__init__('IoU')
        self.name = 'IoU'
        self.axis = axis
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, label, pred):
        """
        Implementation of updating metrics
        """
        i = nd.sum((pred == 1) * (label == 1), axis=self.axis)
        u = nd.sum(pred, axis=self.axis) + nd.sum(label, axis=self.axis) - i
        iou = (i + self.smooth) / (u + self.smooth)
        self.sum_metric += nd.sum(iou, axis=0).asscalar()
        self.num_inst += pred.shape[0]

    def get(self):
        """
        Get the current evaluation result.

        :return (name, value): tuple with metric name and value
        """
        value = (self.sum_metric / self.num_inst) if self.num_inst != 0 else float('nan')
        return self.name, value


def transform(base, mask, multispectral, drop=None):
    """
    Performs all the required preformatting of images

    :param base: ndarray representing the base image
    :param mask: ndarray representing the ground truth mask
    :param multispectral: bool indicating wether we are working with multispectral images or not
    :param drop: list of indexes of bands to drop (not use them) for trainining
    :return: base with shape (channels-len(drop), w, h) as float[0,1], mask with shape (w, h) as binary 0/1
    """
    mask = transform_mask(mask)

    # Multispectral images are already saved with the correct order, and in [0,1] floats so no need to modify them
    if not multispectral:
        base = base.astype('float32') / 255
        base = nd.transpose(base, (2, 0, 1))
    elif drop is not None:
        base = np.delete(base, drop, axis=0)

    return base, mask


def transform_mask(mask):
    # Convert types
    mask = mask.astype('float32') / 255

    # Convert mask to binary
    mask = (mask > 0.4).astype('float32')

    # Reshape the tensors so the order is now (channels, w, h)
    return nd.transpose(mask, (2, 0, 1))
