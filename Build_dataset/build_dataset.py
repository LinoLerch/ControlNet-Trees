import os
import glob
import json
import cv2
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize
from PIL import Image

from skeletonize import skeleton_pipeline
from CaptionPipeline import CaptionPipeline

def get_files_folder(path):
    return [file for file in glob.glob(os.path.join(path,'*'))]
    

def append_img_to_dataset(img, img_nr, output_folder):
    # Skeletonize the image
    skeleton = skeleton_pipeline(img)
    
    # Save both the original image and the skeletonized image
    output_path = os.path.join(output_folder, "images", f'{img_nr}.png')
    output_path_skel = os.path.join(output_folder, "conditioning_images", f'{img_nr}.png')
    input.save(output_path)
    imsave(output_path_skel, img_as_ubyte(skeleton), check_contrast=False)


def build_dataset_from_folder(input_folder, DATASET_DIR):
    """
    Build a dataset from a folder with subfolders of images. \n
    Images are skeletonized and added to the dataset along with the original image. \n
    If the dataset folder already contains images, the new images are appended.

    The input_folder structure should be as follows:
    - input_folder
        - folder1
            - image1.jpg
            - image2.jpg
            - ...
        - folder2
            - image1.jpg
            - image2.jpg
            - ...
    """

    DS_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    # Create output folder if it does not exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(DS_IMAGES_DIR):
        os.makedirs(DS_IMAGES_DIR)
        os.makedirs(os.path.join(DATASET_DIR, "conditioning_images"))

    # Get the number of images already in the dataset
    nr_imgs = len(os.listdir(DS_IMAGES_DIR))
    print(f'Buiding / extending dataset (current size: {nr_imgs} images)')

    folder_list = get_files_folder(input_folder)
    nr_folders = len(folder_list)

    for (i,folder) in enumerate(folder_list):
        # Get all the images in the input folder
        image_list = get_files_folder(folder)
        next_img_nr = len(os.listdir(DS_IMAGES_DIR))
        
        # Run the skeleton pipeline for each image
        for img_path in image_list:
            try:
                append_img_to_dataset(Image.open(img_path), next_img_nr, DATASET_DIR)
                next_img_nr += 1
            except Exception as error:
                print(f'An error occurred in image {img_path}: {error}')
        
        folder_name = folder.split("\\")[-1]
        print(f'Folder {i+1} of {nr_folders} \'{folder_name}\' added to Dataset (new size {next_img_nr} images)')

    create_jsonl_data_file(DATASET_DIR)

    nr_of_images = len(os.listdir(DS_IMAGES_DIR))
    print(f'Dataset created with {nr_of_images} images')


def build_dataset_using_segmentation_labels(input_folder, segmentation_folder, DATASET_DIR):
    """
    Build a dataset from a folder of images and a folder with respective segmentation labels. \n
    (assuming the segmentation labels have the same filename as the images)
    """
    
    DS_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    next_img_nr = len(os.listdir(DS_IMAGES_DIR))
    print(f'Buiding / extending dataset (using segmentation labels) (current size: {next_img_nr} images)')
    image_list = get_files_folder(input_folder)
    
    for img_path in image_list:
        try:
            # Open the image and the segmentation label	
            input = cv2.imread(img_path)
            input = image_resize(input, 512)
            # Get filename and change extension to .png
            filename = os.path.basename(img_path).replace(".jpg", ".png")
            segmentation = cv2.imread(os.path.join(segmentation_folder, filename), cv2.IMREAD_GRAYSCALE)
            segmentation = image_resize(segmentation, 512)
            # thresh = 1
            # im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

            skeleton = skeletonize(segmentation)
            
            # Save both the original image and the skeletonized image
            output_path = os.path.join(DATASET_DIR, "images", f'{next_img_nr}.png')
            output_path_skel = os.path.join(DATASET_DIR, "conditioning_images", f'{next_img_nr}.png')
            cv2.imwrite(output_path, input)
            imsave(output_path_skel, img_as_ubyte(skeleton), check_contrast=False)

            next_img_nr += 1
        except Exception as error:
            print(f'An error occurred in image {img_path}: {error}')
        
    create_jsonl_data_file(DATASET_DIR)

    nr_of_images = len(os.listdir(DS_IMAGES_DIR))
    print(f'Dataset created with {nr_of_images} images')


def create_jsonl_data_file(DATASET_DIR):
    # Create jsonl data file with text (prompt), image and conditioning image (skeleton) 
    nr_of_images = len(os.listdir(os.path.join(DATASET_DIR, "images")))
    metadata = [{"text": "", "image": f"images/{i}.png", "conditioning_image": f"conditioning_images/{i}.png"} for i in range(nr_of_images)]

    # add metadata.jsonl file to this folder
    with open(os.path.join(DATASET_DIR,"train.jsonl"), 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")


def generate_img_captions(DATASET_DIR):
    """
    Generate captions as training prompts for all images in the dataset and add them to the jsonl data file.

    Image captions are generated using GIT: A Generative Image-to-text Transformer for Vision and Language.
    """
    
    # Load the dataset
    dataset = load_dataset(path = DATASET_DIR, data_dir="images", split="train")
    # Load the model
    pipe = pipeline(model="microsoft/git-base-coco", pipeline_class=CaptionPipeline ,device=0, max_new_tokens=20)

    # Generate captions for all images in the dataset
    caption_list = []
    for out in pipe(KeyDataset(dataset, "image")):
        caption = out[0]["generated_text"]
        filename = out[0]["filename"]
        metadata = {"text": caption, "image": f"images/{filename}", "conditioning_image": f"conditioning_images/{filename}"}
        caption_list.append(metadata)

    # add captions to jsonl data file
    assert len(caption_list) == len(dataset), "Number of captions does not match number of images"

    with open(os.path.join(DATASET_DIR,"train.jsonl"), 'w') as f:
        for item in caption_list:
            f.write(json.dumps(item) + "\n")


def image_resize(image, sidelength, inter = cv2.INTER_AREA):
    """
    Resize an image to size sidelength for the smaller side while maintaining the aspect ratio.
    """
    (h, w) = image.shape[:2]

    if h > w:
        width = sidelength
        height = int(h * (width / w))
    else:
        height = sidelength
        width = int(w * (height / h))

    dim = (width, height)
    
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)