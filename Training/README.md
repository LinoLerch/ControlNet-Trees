# Custom ControlNet Training
based on the diffusers [GitHub ControlNet training example](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) and [Blogpost](https://huggingface.co/blog/train-your-controlnet#3-training-the-model)

### Steps to run the custom ControlNet training
1. git clone https://github.com/huggingface/diffusers
2. Copy the files `train.sh`, `val_image_1.png`,... , `val_image_4.png` to the diffusers/examples/controlnet folder
4. Set permissions to run the train.sh script: `chmod +x train.sh`
3. Add the following to the requirements.txt file to install Huggingface diffusers from source `git+https://github.com/huggingface/diffusers.git`
4. Copy the Dataset Loading script `treedataset.py` into the Tree dataset dir (the file name of the script has to be the identical with the directory name of the dataset + `.py`)
5. Set the ROOT_DIR path of the Tree dataset in `treedataset.py`
6. Submit job (see example below)

### Submitting job on cluster
```
submit ./train.sh --pytorch --name controlnet-trees --gpus "4090:1" --requirements requirements.txt --apt-install git --max-time 7-0
```

### Continue Training on pretrained ControlNet
add in train.sh:
- `--controlnet_model_name_or_path="model_out/cn1"`
