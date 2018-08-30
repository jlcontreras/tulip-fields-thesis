# Data augmentation script
#
# Uses an extension of augmentor: https://github.com/mdbloice/Augmentor
import os
import argparse
import msaugmentor


def setup_pipeline(pipeline):
    """
    Define transformations to be applied to our images.
    Details of the transformations here:
    https://github.com/mdbloice/Augmentor#main-features
    """
    pipeline.shear(probability=0.4, max_shear_left=15, max_shear_right=15)
    pipeline.flip_left_right(probability=0.5)
    pipeline.flip_top_bottom(probability=0.5)
    pipeline.rotate_random_90(probability=0.75)


def main():
    parser = argparse.ArgumentParser(description='Data augmentation script')
    parser.add_argument("imgs", metavar='image_dir',
                        help="Path to directory containing the images to be augmented (relative to project root)")

    parser.add_argument("masks", metavar='masks_dir',
                        help="Path to directory containing the ground truth images (also to be augmented)")

    parser.add_argument("out", metavar='output_dir',
                        help="Path to output directory (for both augmented images and masks unless --masks-out \
                             is defined)")

    parser.add_argument('-n', metavar='num_samples',
                        help='Number of images to output (default is 20000)',
                        type=int, dest='n', default=20000)

    parser.add_argument('--multispectral',
                        help='Enable multispectral mode',
                        dest='multi', action='store_true')

    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    img_dir = os.path.join(root, args.imgs)

    if args.multi:
        p = msaugmentor.MsPipeline(source_directory=img_dir, output_directory=os.path.join(root, args.out))
    else:
        p = msaugmentor.RgbPipeline(source_directory=img_dir, output_directory=os.path.join(root, args.out))

    setup_pipeline(p)

    p.ground_truth(os.path.join(root, args.masks))

    # Number of images to generate
    p.sample(args.n)


if __name__ == '__main__':
    main()
