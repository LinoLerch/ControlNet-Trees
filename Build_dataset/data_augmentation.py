import cv2
import os
import torchvision.transforms as T
from PIL import Image
from build_dataset import get_files_folder, append_img_to_dataset, image_resize
import pandas as pd   
 

def augment_image_folder(input_jsonl, DATASET_DIR):
    """
    Augments all images in a folder and saves them to another folder.

    Args:
        input_jsonl (str): Path to the jsonl file containing the image paths.
        DATASET_DIR (str): Path to the folder where the augmented images should be saved.
    """
    # Define the transformations
    colorT = T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=.05)
    flipT = T.RandomHorizontalFlip(p=1)
    perspectiveT = T.RandomPerspective(distortion_scale=0.4, p=1)
    posterT = T.RandomPosterize(4, p=1)

    customT = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=.05),
        T.RandomHorizontalFlip(p=0.75),
        T.RandomPerspective(distortion_scale=0.3, p=0.5),
        T.RandomPosterize(6, p=0.5)
    ])

    list_transformations = [colorT, flipT, perspectiveT, posterT, customT]

    # Create output folder if it does not exist
    DS_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(DS_IMAGES_DIR):
        os.makedirs(DS_IMAGES_DIR)
        os.makedirs(os.path.join(DATASET_DIR, "conditioning_images"))

    # Get the list and number of images to augment
    images_df = pd.read_json(path_or_buf=input_jsonl, lines=True)
    
    len_images = len(images_df)
    next_img_nr = len_images
    img_dir = os.path.dirname(input_jsonl)
    
    output_jsonl = os.path.join(DATASET_DIR,"train.jsonl")
    new_json = []

    # Loop over all images in images_df
    i = 0
    for img_line in images_df.itertuples():
        try:
            img_path = os.path.join(img_dir, img_line.image)
            img = Image.open(img_path).convert("RGB")
            img = image_resize(img, 512)
            prompt = img_line.text
            # Apply all transformations, skeletonize, save
            for t in list_transformations:
                imgT = t(img)
                append_img_to_dataset(imgT, next_img_nr, DATASET_DIR)
                new_json.append({"text": prompt, "image": f"images/{next_img_nr}.png", "conditioning_image": f"conditioning_images/{next_img_nr}.png"})
                next_img_nr += 1
            i += 1
        except Exception as error:
            print(f'An error occurred in image {img_path}: {error}')
        
        # Print progress and write to jsonl file
        if i % 250 == 0:
            print(f'Transformed {i} of {len_images} images')
            write_jsonl_file(new_json, images_df, output_jsonl)
    
    # Write the jsonl file at the end
    write_jsonl_file(new_json, images_df, output_jsonl)
    print(f'Transformed {len_images} of {len_images} images')

def write_jsonl_file(new_json, images_df, output_jsonl):
    new_df = pd.DataFrame(new_json, columns=images_df.columns)
    images_df = pd.concat([images_df, new_df])
    images_df.to_json(path_or_buf=output_jsonl, orient="records", lines=True)
