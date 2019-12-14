import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

import argparse


# setting up network and loading weights

def setup_network(model_name="mono_640x192"):

    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()
    
    return encoder, depth_decoder, loaded_dict_enc


def load_image(loaded_dict_enc, image_path=""):

    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    #print("Input Image Size" + str(input_image.size)) 
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    
    return input_image, input_image_pytorch, original_width, original_height



def predict_depth(encoder, depth_decoder, pytorch_image):
    
    with torch.no_grad():
        features = encoder(pytorch_image)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    return disp


def generate_depth_map(input_image, disp, original_width, original_height, display_parity_flag="false"):
    if display_parity_flag.lower() == "true":
        diplay_parity_flag = True
    elif display_parity_flag.lower() == "false":
        display_parity_flag = False
    else:
        print("Illegal Display Disparity flag")
        exit()
    
    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)


    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    if display_parity_flag:
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(input_image)
        plt.title("Input", fontsize=22)
        plt.axis('off')

        plt.subplot(212)
        plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
        plt.title("Disparity prediction", fontsize=22)
        plt.axis('off')

    return disp_resized_np



def generate_depth_meters(disp_resized_np, original_width, original_height, baseline=0.54, focal=707.0493):
    
    disp_resized_np_after_rescaling = np.multiply(disp_resized_np, original_width)
    prod = baseline*focal
    depth_in_meters = np.divide(prod, disp_resized_np_after_rescaling)
    
    return depth_in_meters


def return_depth_meter_np_array(image_path, baseline=0.54, focal=707.0493, display_disparity_flag="false"):
    
    encoder, depth_decoder, loaded_dict_enc = setup_network()   # taking the default model
    input_image, input_image_pytorch, original_width, original_height = load_image(loaded_dict_enc, image_path)
    disp = predict_depth(encoder, depth_decoder, input_image_pytorch)
    disp_np = generate_depth_map(input_image, disp, original_width, original_height, display_disparity_flag)
    depth_in_meters = generate_depth_meters(disp_np, original_width, original_height, baseline, focal)
    
    return depth_in_meters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Detect depth in an image.')
    parser.add_argument('--image', required=True,
                        metavar="/path/to/image",
                        help="Path to image")
    parser.add_argument('--baseline', required=False,
                        default=0.54,
                        metavar="baseline value of the trained model",
                        help='baseline value of the trained model')
    parser.add_argument('--focal', required=False,
                        default=707.0493,
                        metavar="focal value of the camera",
                        help='focal value of the camera')
    parser.add_argument('--display_disparity_flag', required=False,
                        default="false",
                        metavar="Display image or not",
                        help='Display image or not')


    args = parser.parse_args()

    
    encoder, depth_decoder, loaded_dict_enc = setup_network()   # taking the default model
    input_image, input_image_pytorch, original_width, original_height = load_image(loaded_dict_enc, args.image)
    disp = predict_depth(encoder, depth_decoder, input_image_pytorch)
    disp_np = generate_depth_map(input_image, disp, original_width, original_height, args.display_disparity_flag)
    depth_in_meters = generate_depth_meters(disp_np, original_width, original_height, args.baseline, args.focal)
    # print(depth_in_meters)
    print(return_depth_meter_np_array(args.image))
