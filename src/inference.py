# Satellite image segmentation
# U-Net inference

import os
import unet
import argparse
import glob
import math
import logging

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import img_utils as imgutils

from skimage.io import imsave
from train_utils import transform_mask


class BatchLoader:
    def __init__(self, folders, ext, batch_size, ctx, multisp):
        """
        Class to load all images from the given folders
        :param folders: list of directories to load images from
        :param ext: string extension of the images to be loaded
        :param batch_size: int number of images per batch
        :param ctx: mx.gpu(n) or mx.cpu()
        :param multisp: boolean (multispectral) If true, images will be read from .npy files,
                        and only the first 3 bands will be used for cloud classification.
        """
        self.ctx = ctx
        self.batch_size = batch_size
        self.filenames = []
        self.multisp = multisp

        if multisp:
            ext = '.npy'

        for folder in folders:
            files = glob.glob(os.path.join(folder, '*' + ext))
            logging.info("Scanned {}, found {} images".format(folder, len(files)))
            self.filenames.extend(files)

        if self.filenames:
            self.channels, self.imgsize, _ = self._read_img(self.filenames[0]).shape

        logging.info("Found a total of {} images".format(len(self.filenames)))

    def __len__(self):
        return len(self.filenames)

    def _preprocess(self, data):
        data = nd.array(data).astype('float32').as_in_context(self.ctx)
        if not self.multisp:
            data = data / 255
            data = nd.transpose(data, (2, 0, 1))
        return data

    def _read_img(self, filename):
        if self.multisp:
            img = np.load(filename)
        else:
            img = mx.image.imread(filename)
        return self._preprocess(img)

    def _load_batch(self, filenames):
        batch = mx.nd.empty((len(filenames), self.channels, self.imgsize, self.imgsize), self.ctx)
        for idx, fn in enumerate(filenames):
            batch[idx] = self._read_img(fn)
        return batch

    def get_batches(self):
        for n in range(int(math.ceil(len(self.filenames)/self.batch_size))):
            if (n + 1) * self.batch_size <= len(self.filenames):
                files_batch = self.filenames[n * self.batch_size:(n + 1) * self.batch_size]
            else:
                files_batch = self.filenames[n * self.batch_size:]

            yield self._load_batch(files_batch), files_batch


def save_batch(filenames, predictions, outdir, suffix='predicted_mask.png'):
    for fn, pred in zip(filenames, predictions):
        base, _ = os.path.splitext(os.path.basename(fn))
        mask_name = base + suffix
        if outdir is not None:
            imsave(os.path.join(outdir, mask_name), pred.asnumpy())
        else:
            imsave(os.path.join(os.path.dirname(fn), mask_name), pred.asnumpy())


def plot_predictions(imgs, preds, filenames, mask_dir, multispectral):
    if mask_dir is not None:
        labels = [os.path.join(mask_dir, imgutils.get_mask_fn(fn)) for fn in filenames]
        labels = [transform_mask(mx.image.imread(fn, flag=0)) for fn in labels]

        for img, pred, label in zip(imgs, preds, labels):
            imgutils.plot_images(img, pred, label=label, multisp=multispectral)

    else:
        for img, pred in zip(imgs, preds):
            imgutils.plot_images(img, pred, multisp=multispectral)


def main():
    parser = argparse.ArgumentParser(description='Script to segment tulip fields from the input images')
    parser.add_argument("params",
                        help="Path to .params file with the saved UNet parameters",
                        metavar='net_params')

    parser.add_argument("dirs",
                        help="Directories containing the image files we'll run inference on. \
                              Separated by whitespaces and relative to the root of the project (tulip-fields/)",
                        metavar="dirs", nargs="*")

    parser.add_argument("--output",
                        help="Path to directory where predicted tulip masks are saved. \
                              If not specified, predictions are saved in the same folder as their corresponding image",
                        dest="out", metavar="output_dir", default=None)

    parser.add_argument("--mask-dir",
                        help="Path to directory containing the ground truth masks. \
                             If not specified, output plots will only contain predictions",
                        dest="masks", default=None)

    parser.add_argument("--batch-size",
                        help="Batch size (default is 8)",
                        dest="batch_size", default=8, type=int)

    parser.add_argument("--ext",
                        help="Extension of the image files to be read (default is .png)",
                        dest="ext", default=".png")

    parser.add_argument("--multispectral",
                        help="Enables 13 band multispectral image mode (default is only RGB)",
                        dest="multisp", action="store_true")

    parser.add_argument("--use-cpu",
                        help="Use CPU instead of GPU",
                        dest="cpu", action="store_true")

    parser.add_argument("--show",
                        help="If set, the predictions for the first N batches are displayed",
                        default=0, metavar="N", type=int)

    args = parser.parse_args()

    ctx = mx.cpu() if args.cpu else mx.gpu(0)

    root = os.path.join(os.path.dirname(__file__), os.pardir)
    img_dirs = [os.path.join(root, d) for d in args.dirs]
    outdir = os.path.join(root, args.out) if args.out else None

    # Instantiate a U-Net and load the saved state
    net = unet.Unet()
    net.load_params(os.path.join(root, args.params), ctx)

    loader = BatchLoader(img_dirs, args.ext, args.batch_size, ctx, args.multisp)

    if len(loader) == 0:
        logging.error('No images found, please specify another directory')
    else:
        for idx, (batch, filenames) in enumerate(loader.get_batches()):
            preds = nd.argmax(net(batch), axis=1)
            save_batch(filenames, preds, outdir)
            if idx < args.show:
                plot_predictions(batch, preds, filenames, args.masks, args.multisp)


if __name__ == "__main__":
    main()
