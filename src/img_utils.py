# Utility functions for plotting segmentatino results

import os

import numpy as np
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('No display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime


def get_mask_fn(fn):
    geopedia_layers = {'tulip_field_2016': 'ttl1904', 'tulip_field_2017': 'ttl1905'}
    patch_id = os.path.basename(fn).split('_')[1]
    year = datetime.strptime(fn.split('_')[3], "%Y%m%d-%H%M%S").year
    return 'tulip_{}_geopedia_{}.png'.format(patch_id, geopedia_layers['tulip_field_{}'.format(year)])


def get_overlay(label, pred):
    """
    Superimposes the prediction and the label.
    :param label: ground truth data
    :param pred: prediction
    :return: 3-channel image containing the overlay
    """
    return np.stack([pred, label * 0.3 + pred * 0.3, label], axis=2)


def plot_images(img, prediction, label=None, multisp=True):
    plt.subplots(figsize=(18, 9))

    try:
        img = img.asnumpy()
        prediction = prediction.asnumpy()
    except:
        pass

    n = 2 if label is None else 3

    # Plot image
    ax1 = plt.subplot(1, n, 1)

    if multisp:
        # Multiply by a factor to make RGB representation look better
        img = img * 2.5
        # Transpose because Bands 1:3 correspond to BGR
        img = np.transpose(img[1:4][[2, 1, 0], :, :], (1, 2, 0))
        ax1.imshow(img)
    else:
        ax1.imshow(np.transpose(img, (1, 2, 0)))
    ax1.set_title("Image")
    ax1.axis('off')

    # Plot predicition
    ax2 = plt.subplot(1, n, 2)
    ax2.imshow(prediction, cmap='gray')
    ax2.set_title("Prediction")
    ax2.axis('off')

    # Plot prediction vs groud truth
    if label is not None:
        try:
            label = label.asnumpy()
        except:
            pass

        if len(label.shape) > 2:
            label = label[0]

        ax3 = plt.subplot(1, n, n)
        ax3.imshow(get_overlay(label, prediction))
        ax3.set_title("Prediction vs ground truth")
        ax3.axis('off')
        ax3.annotate('Prediction (red) and ground truth (blue).\nPink represents the intersection of both.',
                     (0, 0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.show()


def plot_bands(img, mask=None):
    """
    Plots a multispectral image band by band
    :param img: numpy array containing the multispectral image
    :param mask: ground truth data for this image
    """

    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06','B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    plt.subplots(figsize=(15, 9))

    for i in range(img.shape[0]):
        band = img[i]
        ax = plt.subplot(3, 5, i+1)
        ax.set_title(band_names[i])
        ax.axis('off')
        ax.imshow(band, cmap='gray')

    # Plot RGB for reference
    ax = plt.subplot(3, 5, 14)
    ax.set_title('RGB')
    ax.axis('off')
    ax.imshow(np.transpose(img[1:4][[2,1,0], :, :], (1,2,0))*2.5)

    # Plot also ground truth mask
    if mask is not None:
        ax = plt.subplot(3, 5, 15)
        ax.set_title('Tulip fields')
        ax.axis('off')
        ax.imshow(mask[0].asnumpy(), cmap='gray')
