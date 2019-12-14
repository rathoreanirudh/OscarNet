from time import time
import garbage_detection 
import argparse


if __name__ == '__main__':
    
    prev = time()
    delta = 59

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect garbage.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=garbage_detection.DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--bbox_flag', required=False,
                        metavar="show bounding box or not",
                        help='To show the bounding box on splasged images')

    args = parser.parse_args()

    # Validate arguments
    assert args.image,\
           "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    class InferenceConfig(garbage_detection.GarbageDetectionConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model
    model = garbage_detection.modellib.MaskRCNN(mode="inference", config=config,
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

    # Evaluate
        while True:
            current = time()
            delta += current - prev
            prev = current

            if delta > 60:
                delta = 0
                if args.bbox_flag.lower() == 'true':
                    garbage_detection.detect_and_color_splash(model, image_path=args.image, bbox_flag=True)
                else:
                    garbage_detection.detect_and_color_splash(model, image_path=args.image, bbox_flag=False)

 
