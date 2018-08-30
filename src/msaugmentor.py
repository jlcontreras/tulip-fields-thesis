# Extension to the Augmentor lib to handle multispectral images.
import os
import uuid
import random
import glob
import Augmentor
import logging

import numpy as np
import mxnet as mx

from Augmentor.ImageUtilities import AugmentorImage
from tqdm import tqdm
from PIL import Image
from mxnet.gluon.data import dataset
from concurrent.futures import ThreadPoolExecutor
from img_utils import get_mask_fn
from data_augmentation import setup_pipeline


class RgbPipeline(Augmentor.Pipeline):
    def ground_truth(self, ground_truth_directory):
        """
        Modification of the original method to conform to our mask naming convention.
        ------
        Specifies a directory containing corresponding images that
        constitute respective ground truth images for the images
        in the current pipeline.

        This function will search the directory specified by
        :attr:`ground_truth_directory` and will associate each ground truth
        image with the images in the pipeline by file name.
    
        The relationship between image and ground truth filenames is the following:
        img filename: '<img_dir>/tulip_<PATCH_ID>_wms_<DATE>_<SENTINELHUB_LAYER>.png'
        ground truth filename: '<mask_dir>/tulip_<PATCH_ID>_geopedia_<GEOPEDIA_LAYER>.png'
        
        Typically used to specify a set of ground truth or gold standard
        images that should be augmented alongside the original images
        of a dataset, such as image masks or semantic segmentation ground
        truth images.

        :param ground_truth_directory: A directory containing the
         ground truth images that correspond to the images in the
         current pipeline.
        :type ground_truth_directory: String
        :return: None.
        """
        num_of_ground_truth_images_added = 0

        # Progress bar
        progress_bar = tqdm(total=len(self.augmentor_images), desc="Processing", unit=' Images', leave=False)

        for augmentor_image_idx in range(len(self.augmentor_images)):
            filename = os.path.basename(self.augmentor_images[augmentor_image_idx].image_file_name)
            mask_fn = get_mask_fn(filename)
            ground_truth_image = os.path.join(ground_truth_directory,
                                              mask_fn)
            if os.path.isfile(ground_truth_image):
                self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                num_of_ground_truth_images_added += 1
                progress_bar.update(1)

        progress_bar.close()

        # May not be required after all, check later.
        if num_of_ground_truth_images_added != 0:
            self.process_ground_truth_images = True

        logging.info("{} ground truth image(s) found.".format(num_of_ground_truth_images_added))

    def _execute(self, augmentor_image, save_to_disk=True, multi_threaded=True):
        """
        Modification so that saved images also follow our naming convention.
        ------
        Private method. Used to pass an image through the current pipeline,
        and return the augmented image.

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        if save_to_disk:
            file_id = str(uuid.uuid4())
            basename_split = os.path.basename(augmentor_image.image_path).split('.')[0].split('_')
            basename_split[1] = file_id
            filename = '_'.join(basename_split)
            try:
                for i in range(len(images)):
                    if i == 0:
                        save_name = filename + "." + \
                                    (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        save_name = get_mask_fn(filename)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
            except IOError as e:
                logging.error("Error writing {}, {}".format(file_id, e.message))

        return images[0], images[1]

    def sample(self, n, multi_threaded=True):
            """
            Generate :attr:`n` number of samples from the current pipeline.
            This function samples from the pipeline, using the original images
            defined during instantiation. All images generated by the pipeline
            are by default stored in an ``output`` directory, relative to the
            path defined during the pipeline's instantiation.
            By default, Augmentor will use multi-threading to increase the speed
            of processing the images. However, this may slow down some
            operations if the images are very small. Set :attr:`multi_threaded`
            to ``False`` if slowdown is experienced.
            :param n: The number of new samples to produce.
            :type n: Integer
            :param multi_threaded: Whether to use multi-threading to process the
             images. Defaults to ``True``.
            :type multi_threaded: Boolean
            :return: None
            """
            if len(self.augmentor_images) == 0:
                raise IndexError("There are no images in the pipeline. "
                                 "Add a directory using add_directory(), "
                                 "pointing it to a directory containing images.")

            if len(self.operations) == 0:
                raise IndexError("There are no operations associated with this pipeline.")

            if n == 0:
                augmentor_images = self.augmentor_images
            else:
                augmentor_images = [random.choice(self.augmentor_images) for _ in range(n)]

            if multi_threaded:
                with tqdm(total=len(augmentor_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                    with ThreadPoolExecutor(max_workers=None) as executor:
                        for _ in executor.map(self, augmentor_images):
                            progress_bar.update(1)
            else:
                with tqdm(total=len(augmentor_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                    for augmentor_image in augmentor_images:
                        self._execute(augmentor_image)
                        progress_bar.set_description("Processing {}".format(os.path.basename(augmentor_image.image_path)))
                        progress_bar.update(1)


class MsPipeline(RgbPipeline):
    def __init__(self, **kwargs):
        # Monkey-patch the scan_directory function from Augmentor so it looks for .npy files instead of images
        Augmentor.ImageUtilities.scan_directory = self.scan_directory
        super(MsPipeline, self).__init__(**kwargs)

    @staticmethod
    def scan_directory(source_directory):
        """
        Scan a directory for multispectral images, returning any images found with the
        extension ``.npy``.
        :param source_directory: The directory to scan for images.
        :type source_directory: String
        :return: A list of images found in the :attr:`source_directory`
        """
        file_types = ['*.npy']

        list_of_files = []

        if os.name == "nt":
            for file_type in file_types:
                list_of_files.extend(glob.glob(os.path.join(os.path.abspath(source_directory), file_type)))
        else:
            file_types.extend([str.upper(str(x)) for x in file_types])
            for file_type in file_types:
                list_of_files.extend(glob.glob(os.path.join(os.path.abspath(source_directory), file_type)))

        return list_of_files

    def _check_images(self, abs_output_directory):
        """
        Private method. Used to check and get the dimensions of all of the images
        :param abs_output_directory: the absolute path of the output directory
        :return:
        """
        # Make output directory/directories
        if len(set(self.class_labels)) <= 1:
            if not os.path.exists(abs_output_directory):
                try:
                    os.makedirs(abs_output_directory)
                except IOError:
                    logging.error("Insufficient rights to read or write output directory ({})"
                                  .format(abs_output_directory))
        else:
            for class_label in self.class_labels:
                if not os.path.exists(os.path.join(abs_output_directory, str(class_label[0]))):
                    try:
                        os.makedirs(os.path.join(abs_output_directory, str(class_label[0])))
                    except IOError:
                        logging.error("Insufficient rights to read or write output directory ({})"
                                      .format(abs_output_directory))
        # Check the images, read their dimensions, and remove them if they cannot be read
        for augmentor_image in self.augmentor_images:
            try:
                opened_image = np.load(augmentor_image.image_path)
                self.distinct_dimensions.add(opened_image.shape)
            except IOError as e:
                logging.error("There is a problem with image {} in the source directory: {}"
                              .format(augmentor_image.image_path, e.message))
                self.augmentor_images.remove(augmentor_image)

        logging.info("Initialised with {} image(s) found.\n".format(len(self.augmentor_images)))
        logging.info("Output directory set to {}.".format(abs_output_directory))

    def _execute(self, augmentor_image, save_to_disk=True, multi_threaded=True):
        """
        Modification to handle multispectral images.
        """

        images = []

        # Data is saved with the shape (bands, width, height)
        img = np.load(augmentor_image.image_path)

        # npad is a tuple of (n_before, n_after) for each dimension
        m = (3 - img.shape[0] % 3) % 3
        npad = ((0, m), (0, 0), (0, 0))
        padded_img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
        # PIL's fromarray needs the images to be in uint8
        padded_img = (np.transpose(padded_img, (1, 2, 0)) * 255).astype(np.uint8)

        images.extend([Image.fromarray(a, mode='RGB') for a in np.dsplit(padded_img, padded_img.shape[2]/3)])

        if augmentor_image.ground_truth is not None:
            images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        # Reconstruct the original numpy array
        arrays = [np.array(img) for img in images[:-1]]
        augmented = np.transpose(np.dstack(arrays), (2, 0, 1))
        augmented = (augmented[:img.shape[0], :, :]).astype(np.float32) / 255  # Convert back to float

        mask = images[-1]

        if save_to_disk:
            file_id = str(uuid.uuid4())
            basename_split = os.path.basename(augmentor_image.image_path).split('.')[0].split('_')
            basename_split[1] = file_id
            filename = '_'.join(basename_split)
            try:
                # Save image
                imgname = filename + "." \
                          + (self.save_format if self.save_format else augmentor_image.file_format)
                np.save(os.path.join(augmentor_image.output_directory, imgname), augmented)
                # Save ground truth
                if augmentor_image.ground_truth is not None:
                    maskname = get_mask_fn(filename)
                    mask.save(os.path.join(augmentor_image.output_directory, maskname))
            except IOError as e:
                logging.error("Error writing {}, {}".format(file_id, e.message))

        return augmented, mask


class AugmentorDataset(dataset.Dataset):
    def __init__(self, imgdir, maskdir, multisp=False, transform_fn=None):
        self._multisp = multisp
        self._transform = transform_fn

        if multisp:
            self.pipeline = MsPipeline(source_directory=imgdir,
                                       output_directory=imgdir)
        else:
            self.pipeline = RgbPipeline(source_directory=imgdir,
                                        output_directory=imgdir)
        self.pipeline.ground_truth(maskdir)

        setup_pipeline(self.pipeline)

    def __getitem__(self, idx):
        base, mask = self.pipeline._execute(self.pipeline.augmentor_images[idx], save_to_disk=False)

        base = mx.nd.array(base)
        mask = mx.nd.array(mask)
        mask = mx.nd.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        if self._transform is not None:
            return self._transform(base, mask, self._multisp)
        else:
            return base, mask

    def __len__(self):
        return len(self.pipeline.augmentor_images)
