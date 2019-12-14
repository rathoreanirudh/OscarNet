import os
import sys
import math
import argparse
from time import time

# Import local libraries from separate folders OscarNet and monodepth2
ROOTDIR = os.path.abspath("")
MRCNNDIR = ROOTDIR+"/Mask_RCNN"
DEPTHDIR = ROOTDIR+"/monodepth2"

sys.path.append(MRCNNDIR)
sys.path.append(DEPTHDIR)
import get_depth_in_meters as gdim
import code_OscarNet.OscarNet.garbage_detection as get_mask

#image_path = "/home/shivendra/Downloads/himanshu_test/7XRNH2UIXUI6DAESRAMLQIIEU4.jpg"
#weights_path = "/home/shivendra/Downloads/himanshu_test/Mask_RCNN/mask_rcnn_oscarnet.h5"

class GetMaskConfig(get_mask.GarbageDetectionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = GetMaskConfig()


def calculate_depth_matrix(image_path):
    depth_matrix = gdim.return_depth_meter_np_array(image_path)
    depth_matrix = depth_matrix.transpose()
    print("Shape of Depth matrix: " +  str(depth_matrix.shape)) 
    return depth_matrix

def generate_masks(weights_path, image_path):
    model = get_mask.modellib.MaskRCNN(mode="inference", config=config, model_dir=get_mask.DEFAULT_LOGS_DIR)
    model.load_weights(weights_path, by_name=True)
    predicted_masks_matrix = get_mask.get_predicted_mask(model, image_path)
    predicted_masks_matrix = predicted_masks_matrix.transpose(1,0,2)
    print("Shape of Mask matrix: " +  str(predicted_masks_matrix.shape)) 
    return predicted_masks_matrix


def is_mask(predicted_mask_matrix, x, y):
    a, b, c = predicted_mask_matrix.shape
    bool_value = False
    # if x == 1441 and y == 122
    for i in range(c):
        bool_value |= predicted_mask_matrix[x][y][i]
    return bool_value

def calculate_min_depth_point(weights_path, image_path):
    depth_matrix = calculate_depth_matrix(image_path)
    image_size = depth_matrix.shape
    #old_image_size = image_size
    #image_size = image_size[::-1]
    predicted_masks_matrix = generate_masks(weights_path, image_path)
    min_depth_x = 0
    min_depth_y = 0
    min_depth = 100000
    min_depth_points = []
    
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if is_mask(predicted_masks_matrix, i, j):
                depth_ij = depth_matrix[i][j]
                if depth_ij < min_depth:
                    min_depth_x = i
                    min_depth_y = j
                    min_depth_points = []
                    min_depth = depth_ij
                    min_depth_points.append((min_depth_x, min_depth_y))
                elif depth_ij == min_depth:
                    min_depth_points.append((min_depth_x, min_depth_y))

    return min_depth_points, min_depth, image_size

def calculate_locomotion_direction(min_depth_points, min_depth, image_size):
    start_x = (image_size[0]+1)/2
    start_y = image_size[1]

    action = {}
    action['depth'] = min_depth
    min_angle = 180
    for point in min_depth_points:
        print("hi")
        target_x, target_y = point
        if target_x > start_x:
            if target_y == start_y:
                angle = 90.0
            else:
                angle = math.degrees(math.atan((target_x - start_x)/(start_y - target_y)))
            if angle < min_angle:
                min_angle = angle
                action['direction'] = 'right'
                action['angle'] = min_angle
                action['x'] = target_x
                action['y'] = target_y
        elif target_x < start_x:
            if target_y == start_y:
                angle = 90.0
            else:
                angle = math.degrees(math.atan((start_x - target_x)/(start_y - target_y)))
            if angle < min_angle:
                min_angle = angle
                action['direction'] = 'left'
                action['angle'] = min_angle
                action['x'] = target_x
                action['y'] = target_y
        elif target_x == start_x:
            angle = 0
            if target_y == start_y:
                min_angle = angle
                action['direction'] = 'pick'
                action['angle'] = min_angle
                action['x'] = target_x
                action['y'] = target_y
            else:
                min_angle = angle
                action['direction'] = 'straight'
                action['angle'] = min_angle
                action['x'] = target_x
                action['y'] = target_y

    return action
            
def find_action(weights_path, image_path):
    # depth_matrix = calculate_depth_matrix(image_path)
    # predicted_masks_matrix = generate_masks(weights_path, image_path)
    min_depth_points, min_depth, image_size = calculate_min_depth_point(weights_path, image_path)
    action = calculate_locomotion_direction(min_depth_points, min_depth, image_size)
    return action

prev = time()
delta = 59
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate commands for the bot')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=True,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    print("Recommended action")
    
    # pick up an image to get the direction
    while True:
        current = time()
        delta += current - prev
        
        if delta > 60:
            delta = 0
            print(find_action(args.weights, args.image))
