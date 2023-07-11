import os
import glob
import json
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from datasets import Image as DatasetsImage
from skimage.io import imsave, imread
from skimage.util import img_as_ubyte
from PIL import Image

from skeletonize import skeleton_pipeline
from CaptionPipeline import CaptionPipeline

def get_files_folder(path):
    return [file for file in glob.glob(os.path.join(path,'*'))]
    

def append_img_to_dataset(img_path, img_nr, output_folder):
    # Open the image
    input = Image.open(img_path)
    skeleton = skeleton_pipeline(input)
    
    # Save both the original image and the skeletonized image
    output_path = os.path.join(output_folder, "images", f'{img_nr}.png')
    output_path_skel = os.path.join(output_folder, "conditioning_images", f'{img_nr}.png')
    input.save(output_path)
    imsave(output_path_skel, img_as_ubyte(skeleton), check_contrast=False)


def build_dataset_from_folder(input_folder, DATASET_DIR):
    DS_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    # Create output folder if it does not exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(DS_IMAGES_DIR):
        os.makedirs(DS_IMAGES_DIR)
        os.makedirs(os.path.join(DATASET_DIR, "conditioning_images"))

    folder_list = get_files_folder(input_folder)

    for folder in folder_list:
        # Get all the images in the input folder
        image_list = get_files_folder(folder)

        next_img_nr = len(os.listdir(DS_IMAGES_DIR))
        
        # Run the skeleton pipeline for each image
        for img_path in image_list:
            try:
                append_img_to_dataset(img_path, next_img_nr, DATASET_DIR)
                next_img_nr += 1
            except Exception as error:
                print(f'An error occurred in image {img_path}: {error}')
        
        folder_name = folder.split("\\")[-1]
        print(f'Folder \'{folder_name}\' added to Dataset (new size {next_img_nr} images)')

    # Create jsonl data file 
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
