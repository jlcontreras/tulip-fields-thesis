# ## Satellite image segmentation

import os
import argparse
import unet
import shutil
import time
import logging

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import matplotlib.pyplot as plt

from tqdm import tqdm
from train_utils import ImageWithMaskDataset, DiceCoeffLoss, IouMetric, transform
from msaugmentor import AugmentorDataset


def evaluate(data_iterator, net, ctx):
    """
    Evaluates network performance by calculating IoU

    :param data_iterator: gluon.data.DataLoader containing the test data
    :param net: network to be tested
    :param ctx: context
    :return: float IoU score
    """
    metric = IouMetric()
    for data, labels in data_iterator:
        # To prevent 'ValueError: Too many slices for data with shape ...' in case the last batch is too small
        if data.shape[0] < len(ctx):
            continue

        data = gluon.utils.split_and_load(data, ctx, even_split=False)
        labels = gluon.utils.split_and_load(labels, ctx, even_split=False)

        preds = [nd.argmax(net(b), axis=1, keepdims=True) for b in data]
        for lab, pred in zip(labels, preds):
            metric.update(lab, pred)

    return metric.get()[1]


def train(net, data, loss_fn, trainer, ctx, epochs, checkpoint_dir):
    """
    Function to train the neural network.
    
    Params:
    :param net: network to train
    :param data: dict containing the gluon.data.DataLoader for train and val
    :param loss_fn: loss function to use for training
    :param trainer: gluon.Trainer to use for training
    :param ctx: context where we will operate (GPU or CPU)
    :param epochs: number of epochs to train for
    :param checkpoint_dir: directory where checkpoints are saved every 100 batches
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_iter = data['train']
    val_iter = data['val']
    res = {'train': [], 'val': [], 'names': [], 'time': 0}
    start = time.time()

    for epoch in range(epochs):
        metric = IouMetric()
        for i, (batch, labels) in tqdm(enumerate(train_iter), desc='Epoch {}'.format(epoch)):
            batch_size = batch.shape[0]

            # To prevent 'ValueError: Too many slices for data with shape ...' in case the last batch is too small
            if batch_size < len(ctx):
                continue

            # Split the data and load each part in a gpu
            batch = gluon.utils.split_and_load(batch, ctx, even_split=False)
            labels = gluon.utils.split_and_load(labels, ctx, even_split=False)
            
            with mx.autograd.record():
                outputs = [net(b) for b in batch]
                losses = [loss_fn(lab, out) for lab, out in zip(labels, outputs)]
                preds = [nd.argmax(out, axis=1, keepdims=True) for out in outputs]

            for loss in losses:
                loss.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
            
            #  Keep a moving average of the iou
            for lab, pred in zip(labels, preds):
                metric.update(lab, pred)

            if i != 0 and i % 50 == 0:
                np.save(os.path.join(checkpoint_dir, 'results.npy'), res)

        names, train_iou = metric.get()
        val_iou = evaluate(val_iter, net, ctx)

        model_fn = '{}.params'.format(epoch)

        res['train'].append(train_iou)
        res['val'].append(val_iou)
        res['names'].append(model_fn)

        # Only save model params if results are better than the previous ones
        if val_iou >= max(res['val']):
            net.save_params(os.path.join(checkpoint_dir, model_fn))

        logging.info("Epoch {} | train IoU {} | val IoU {}".format(epoch, train_iou, val_iou))
        metric.reset()

    elapsed = time.time() - start
    res['time'] = elapsed
    logging.info("Elapsed time: {}s".format(elapsed))
    return res


# Execution
def main():
    parser = argparse.ArgumentParser(description="Script to train a U-Net on the images passed as parameters.")
    parser.add_argument("train", metavar="train_images",
                        help="Path to directory containing the training images (relative to project root)")

    parser.add_argument("val", metavar="val_images",
                        help="Path to directory containing the validation images")

    parser.add_argument("masks", metavar="masks",
                        help="Path to directory containing the ground truth masks")

    parser.add_argument("--train-masks",
                        help="Used to define a different directory for training set masks",
                        default=None, dest="train_masks")

    parser.add_argument("--checkpoints", metavar="checkpoints_dir",
                        help="Path to directory where the parameters of the best models will be stored",
                        default="models")

    parser.add_argument("--epochs",
                        help="Number of epochs to train for (default is 50)",
                        type=int, default=50)

    parser.add_argument("--batch-size",
                        help="Specify batch size (default is 8)",
                        type=int, default=8)

    parser.add_argument("--multispectral",
                        help="Enable multispectral mode",
                        action="store_true")

    parser.add_argument("--use-cpu",
                        help="Train on CPU instead of GPU (very slow, not recommended)",
                        dest="cpu", action="store_true")

    parser.add_argument("--gpu-count",
                        help="Number of GPUs to use",
                        type=int, default=1, dest="gpus")

    parser.add_argument("--drop-bands",
                        help="Indexes of bands not to be used, separed by commas",
                        default=None, dest="drop")

    parser.add_argument("--show",
                        help="Plots the learning curve once training is completed",
                        dest="show", action="store_true")

    parser.add_argument("--data-aug",
                        help="If selected, training data is augmented online before being fed to the network",
                        dest="aug", action="store_true")

    args = parser.parse_args()

    ctx = [mx.cpu()] if args.cpu else [mx.gpu(i) for i in range(args.gpus)]
    batch_size = args.batch_size

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    train_dir = os.path.join(root, args.train)
    val_dir = os.path.join(root, args.val)
    mask_dir = os.path.join(root, args.masks)
    checkpoint_dir = os.path.join(root, args.checkpoints)
    tmp = os.path.join(os.path.dirname(__file__), 'tmp')

    train_masks = os.path.join(root, args.train_masks) if args.train_masks is not None else mask_dir

    drop_bands = [int(band) for band in args.drop.split(',')] if args.drop is not None else None

    # Create train and validation DataLoaders from our Datasets
    if args.aug:
        dataset_class = AugmentorDataset
    else:
        dataset_class = ImageWithMaskDataset

    train_ds = dataset_class(train_dir, train_masks, multisp=args.multispectral,
                             transform_fn=lambda b, m, ms: transform(b, m, ms, drop=drop_bands))
    train_iter = gluon.data.DataLoader(train_ds, batch_size, shuffle=True)

    val_ds = ImageWithMaskDataset(val_dir, mask_dir, multisp=args.multispectral,
                                  transform_fn=lambda b, m, ms: transform(b, m, ms, drop=drop_bands))
    val_iter = gluon.data.DataLoader(val_ds, batch_size, shuffle=True)

    if len(train_ds) == 0:
        raise ValueError('The train directory {} does not contain any valid images'.format(train_dir))
    if len(val_ds) == 0:
        raise ValueError('The test directory {} does not contain any valid images'.format(train_dir))

    data = {'train': train_iter, 'val': val_iter}

    # Instantiate a U-Net and train it
    net = unet.Unet()
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()
    loss = DiceCoeffLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 1e-4, 'beta1': 0.9, 'beta2': 0.99})

    epochs = args.epochs

    results = train(net, data, loss, trainer, ctx, epochs, tmp)

    if args.show:
        plt.plot(range(len(results['val'])), results['val'])
        plt.title('Model learning curve')
        plt.xlabel('Epoch')
        plt.ylabel('IoU score')
        plt.show()

    # Find best scoring model
    best = results['val'].index(max(results['val']))
    best_params = os.path.join(tmp, results['names'][best])

    # Copy it to <checkpoints> folder
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.multispectral:
        bands = 'ALL_BANDS' if drop_bands is None else ('ALL_BANDS-' + '-'.join([str(b) for b in drop_bands]))
    else:
        bands = 'RGB'
    save_filename = os.path.join(checkpoint_dir, 'unet_{}.params'.format(bands))
    shutil.copyfile(best_params, save_filename)
    np.save(os.path.join(checkpoint_dir, 'unet_{}_learning_curve_data.npy'.format(bands)), results)
    shutil.rmtree(tmp)

    logging.info('Best model on validation set saved in: {}'.format(save_filename))


if __name__ == '__main__':
    main()
