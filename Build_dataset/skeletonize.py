import cv2
import numpy as np
from rembg import remove
from PIL import Image
from skimage.morphology import skeletonize
from skimage.io import imsave, imread
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import os
import glob

def binarize_transparent(img_pil, threshold=0):
    #https://stackoverflow.com/questions/61918194/how-to-make-a-binary-mask-out-of-an-image-with-a-transparent-background
    #https://learnopencv.com/opencv-threshold-python-cpp/
    
    # load image with alpha channel
    #img = cv2.imread('object.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)

    # extract alpha channel
    alpha = img[:, :, 3]

    # threshold alpha channel
    alpha = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)[1]

    return alpha

def opening_closing(img_cv2):
    kernel_open = np.ones((21,21),np.uint8)
    kernel_close = np.ones((15,15),np.uint8)

    opening = cv2.morphologyEx(img_cv2, cv2.MORPH_OPEN, kernel_open)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)

    return opening, closing

def skeleton_pipeline(img_pil):
    # Removing the background
    rembg_img = remove(img_pil)

    # Binarize
    binary_img = binarize_transparent(rembg_img)

    # Opening and closing
    opening, closing = opening_closing(binary_img)

    # Skeletonize
    return skeletonize(closing)
  
def dev_skeleton_pipeline(ROOT_DIR_GIS, input_path, save_steps=False):

    img_nr = input_path.split(".")[-2][-1]
    
    # Open the image
    input = Image.open(os.path.join(ROOT_DIR_GIS,input_path))
    
    # Removing the background
    rembg_img = remove(input)

    # Binarize
    binary_img = binarize_transparent(rembg_img)

    # Opening and closing
    opening, closing = opening_closing(binary_img)

    # Save intermediate steps images
    if save_steps:
        # Store path of the output image in the variable output_path
        output_folder = os.path.join(ROOT_DIR_GIS,'photos','tree_steps')
        output_path_bg = os.path.join(output_folder, f'tree{img_nr}_rembg.png')
        output_path_bin = os.path.join(output_folder, f'tree{img_nr}_binary.png')
        output_path_closing = os.path.join(output_folder, f'tree{img_nr}_closing.png')
        
        cv2.imwrite(output_path_closing, closing)
        cv2.imwrite(output_path_bin, binary_img)
        rembg_img.save(output_path_bg)
    else:
        output_folder = os.path.join(ROOT_DIR_GIS,'photos','tree_skeleton')

    # Skeletonize
    skeleton = skeletonize(closing)
    output_path_skel = os.path.join(output_folder, f'tree{img_nr}_skel.png')
    imsave(output_path_skel, img_as_ubyte(skeleton), check_contrast=False)

    return f'Image {img_nr} processed and results saved'