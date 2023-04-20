import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


from utils.helper import prepare_image, get_torch_prompts_labels

sys.path.append("..")

val_grounding_file = 'val_grounding.json'
batch_size = 5
max_prompt_size = 3
sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
save_predictions = "./predictions/"
thresh = 150


def get_data():
    with open(val_grounding_file) as f:
        data = json.load(f)

    return data


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

data = get_data()
keys = data.keys()

for i in range(0,len(keys),batch_size):
    valid_indices = [j for j in range(i, min(i+batch_size, len(keys)))]
    batched_input = list()
    for idx in valid_indices:
        temp = dict()
        image = cv2.imread('./val/'+keys[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xy_coo_array, xy_label_array = get_torch_prompts_labels(data[keys[idx]]["answer_grounding"], max_prompt_size, device)
        temp = {
            'image': prepare_image(image, resize_transform, sam),
            'point_coords': resize_transform.apply_coords_torch(xy_coo_array, image.shape[:2]),
            'point_labels': xy_label_array,
            'original_size': image.shape[:2]
        }
        batched_input.append(temp)

    batched_output = sam(batched_input, multimask_output=False)
    for idx,obj in (valid_indices,batched_output):
        mask = obj['masks']
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        fn = lambda x : 255 if x > thresh else 0
        re_image = Image.fromarray((mask_image * 255).astype(np.uint8)).convert('L').point(fn, mode='1')
        re_image.save(save_predictions + keys[idx])



