# ControlNet-Trees
Creating realistic tree images with Stable Diffusion and ControlNet

## Setup
- git clone https://github.com/ohyicong/Google-Image-Scraper.git
- pip install -r requirements.txt

## Skeletonizing (Pipeline for condition images)
1. Remove background
2. Binarize
3. Opening and Closing (to improve binary image)
4. Skeletonize