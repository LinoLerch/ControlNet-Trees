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
    
  
def run_skeleton_pipeline(ROOT_DIR_GIS, input_path, save_steps=False):

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

def get_files_folder(path):
    image_list = []
    for filename in glob.glob(os.path.join(path,'*')):
        im=filename #Image.open(filename)
        image_list.append(im)
    return image_list

def append_img_to_dataset(img_path, img_nr, output_folder):
    # Open the image
    input = Image.open(img_path)
    
    # Removing the background
    rembg_img = remove(input)

    # Binarize
    binary_img = binarize_transparent(rembg_img)

    # Opening and closing
    opening, closing = opening_closing(binary_img)

    # Skeletonize
    skeleton = skeletonize(closing)
    
    # Save both the original image and the skeletonized image
    output_path = os.path.join(output_folder, "images", f'{img_nr}.png')
    output_path_skel = os.path.join(output_folder, "conditioning_images", f'{img_nr}.png')
    input.save(output_path)
    imsave(output_path_skel, img_as_ubyte(skeleton), check_contrast=False)

def build_dataset(input_folder, output_folder):
    output_folder_images = os.path.join(output_folder, "images")
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_images):
        os.makedirs(output_folder_images)
        os.makedirs(os.path.join(output_folder, "conditioning_images"))
    
    # Get all the images in the input folder
    image_list = get_files_folder(input_folder)

    next_img_nr = len(os.listdir(output_folder_images))
    
    # Run the skeleton pipeline for each image
    for img_path in image_list:
        try:
            append_img_to_dataset(img_path, next_img_nr, output_folder)
            next_img_nr += 1
        except Exception as error:
            print(f'An error occurred in image {img_path}: {error}')
    
    return f'Dataset expanded to {next_img_nr} images'
