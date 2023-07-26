export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="model_out"

accelerate launch train_controlnet.py --mixed_precision="fp16" \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir="/mnt/hdd/linol/treedataset" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./val_image_1.png" "./val_image_2.png" \
 --validation_prompt "a lone tree in the middle of a field" "a large tree with a blue sky." \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --mixed_precision="fp16" \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet_trees" \
 --checkpointing_steps=5000 \
 --validation_steps=5000
