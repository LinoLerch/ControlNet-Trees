# ControlNet-Trees
Training a custom [ControlNet](https://github.com/lllyasviel/ControlNet) to generate realistic tree images with [Stable Diffusion](https://github.com/CompVis/stable-diffusion) conditioned by hand-drawn-like tree skeletons

<img src="https://github.com/LinoLerch/ControlNet-Trees/assets/113920231/3e238764-e8ee-4db5-aabe-dd9e2e1e32ca" alt="drawing" width="600"/>

## Setup
- pip install -r requirements.txt

## Abstract
Text-to-image models such as Stable Diffusion made it possible to generate realistic images
from a simple text prompt. The ControlNet architecture extends this by adding conditional
images that allow to specify the layout of the desired image. In our project, we trained a
custom ControlNet to generate realistic tree images conditioned by a tree skeleton that
indicates shape and position. We started by building the dataset obtaining ground truth tree
images from a tree dataset and web scraping, plus expanding it through data augmentation.
We implemented a skeleton pipeline to turn these images into hand-drawn-like tree skeletons.
The text prompts were generated using an image captioning model. Our ControlNet model
was trained over several consecutive runs, first on the small then on the augmented dataset.
We defined four validation inputs to track the progress of the model during training and
compare the results with the baseline, the ControlNet scribble model. The results indicate
that our model does not follow the condition image as strictly as the ControlNet scribble
model. However, our model fixes problems of the baseline by avoiding leafless branches
and often producing more natural and photorealistic tree images.

## Build Dataset
see [Build_dataset/README.md](Build_dataset/README.md)

## Training
see [Training/README.md](Training/README.md)

## Results
![grafik](https://github.com/LinoLerch/ControlNet-Trees/assets/113920231/12be017c-0785-4371-bc86-87f649822ccc)

