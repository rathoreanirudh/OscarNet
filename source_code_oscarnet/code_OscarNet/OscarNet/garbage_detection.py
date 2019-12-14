"""
Mask R-CNN
Train on the SSGarbage dataset and implement color splash effect with a bounding box.

Large chunk of code borrowed from Matterport, Inc.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: Run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 garbage_detection.py train --dataset=/path/to/garbage/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 garbage_detection.py train --dataset=/path/to/garbage/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 garbage_detection.py train --dataset=/path/to/garbage/dataset --weights=imagenet

    # Apply color splash and bounding box to an image
    python3 garbage_detection.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file> --bbox_flag=true/false

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images


# importing to calculate loss
import keras.layers as KL

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class GarbageDetectionConfig(Config):
    """Configuration for training on the SSGarbage dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "garbage"

    # We use a GPU with 8GB memory, which can fit one images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1


    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + garbage

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class GarbageDataset(utils.Dataset):

    def load_garbage(self, dataset_dir, subset):
        """Load a subset of the Garbage dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("garbage", 1, "garbage")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "garbage",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a garbage dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "garbage":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "garbage":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GarbageDataset()
    dataset_train.load_garbage(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GarbageDataset()
    dataset_val.load_garbage(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
         
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        bbox = utils.extract_bboxes(mask)
        class_ids = np.array([1 for i in range(bbox.shape[0])])

        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def get_predicted_mask(model, image_path=None):
    assert image_path
    
    if image_path:
        # Run model detection and generate the color splash effect
        
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        mask = r['masks']
        if mask.shape[-1] > 0:
         
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
     
        return mask
    else:
        print("No image path provided")
        exit()


def bbox_show(image, mask):
    """Apply bbox and show"""
    if mask.shape[-1] > 0:
        # mask = (np.sum(mask, -1, keepdims=True) >= 1)
        bbox = utils.extract_bboxes(mask)
        class_ids = np.array([1 for i in range(bbox.shape[0])])
        visualize.display_instances(image, bbox, mask, class_ids, ['BG', 'GARBAGE'])


def detect_and_color_splash(model, image_path=None, bbox_flag=False):
    assert image_path 

    # Image
    if image_path:
        # Run model detection and generate the color splash effect
        
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        
        if bbox_flag:
            bbox_show(image, r['masks'])
        
        splash = color_splash(image, r['masks'])

        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # file_name = "splash.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    
        print("Saved to ", file_name)
        
    else:
        print("No image path provided")
        exit()


def calculate_mAP(model, image_folder, iou_threshold=0.5):
    """calculate mean average precision (mAP) for dataset"""

    dataset = GarbageDataset()
    GARBAGE_DIR = args.dataset
    dataset.load_garbage(GARBAGE_DIR, image_folder)
    APs = []
    dataset.prepare()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, GarbageDetectionConfig(),
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, GarbageDetectionConfig()), 0)
        results = model.detect([image], verbose=0)
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold)
        APs.append(AP)
    print("mAP: ", np.mean(APs))
    return np.mean(APs)


def calculate_mAP_with_diff_iou_threshold(model, subset):

    # subset = train or val or test
    mAP_list = []
        
    for iou in np.arange(0.5, 1.0, 0.05):
        print("Calculating mAP value for iou = "+str(iou))
        mAP_value = calculate_mAP(model, subset, iou) 
        mAP_list.append(mAP_value)
        # print("mAP value for "+str(iou)+" is "+str(mAP_value))

    return mAP_list



############################################################
#  Training
############################################################



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect garbage.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'metrics'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/garbage/dataset/",
                        help='Directory of the SSGarbage dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--iou', required=False,
                        default=0.5,
                        metavar="mAP for what value of iou",
                        help="iou can range from 0.5 to 0.95 in general")
    parser.add_argument('--subset_folder', required=False,
                        default="train",
                        metavar="subset folder for calculating metrics on",
                        help="sub folder on which metric needs to be calculated")
    parser.add_argument('--bbox_flag', required=False,
                        metavar="show bounding box or not",
                        help='To show the bounding box on splashed images')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image to apply color splash"
    elif args.command == "metrics":
        assert args.dataset, "Argument --dataset is required for calculating metrics"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GarbageDetectionConfig()
    else:
        class InferenceConfig(GarbageDetectionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        if args.bbox_flag.lower() == 'true':
            detect_and_color_splash(model, image_path=args.image, bbox_flag=True)
        else:
           detect_and_color_splash(model, image_path=args.image)
    elif args.command == "metrics":
        if args.iou == "all":
            mAP_list_on_subset_folder = calculate_mAP_with_diff_iou_threshold(model, args.subset_folder)
            # print(mAP_list_on_subset_folder) 
        elif args.iou.replace('.', '1').isdigit():
            iou_threshold = float(args.iou)
            print("Caluclating mAP value for iou = "+str(iou_threshold))
            mAP_value = calculate_mAP(model, args.subset_folder, iou_threshold)
            # print(mAP_value)
        else:
            print("Invalid iou value")
            exit()
              
    else:
        # get_predicted_mask(model, args.image)
        print("'{}' is not recognized. "
              "Use 'train' or 'splash' or 'metrics'".format(args.command))

      
