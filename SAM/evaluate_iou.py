import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img
from tqdm import tqdm
import json

val_grounding_file = 'val_grounding.json'
pred_folder = '/gdrive/MyDrive/Research_ICV/predictions'
gt_folder = './binary_masks_png/val/'

def get_data():
    with open(val_grounding_file) as f:
        data = json.load(f)

    return data

def binarize_image(point):
    return 0 if point == 0 else 1
def bw_image(point):
    return 255 if point > 50 else 0

data = get_data()
iou = 0
for key in data:
    image = img.imread(gt_folder + key).astype('uint8')
    pred_image = img.imread(pred_folder + key).astype('uint8')
    image = np.vectorize(binarize_image)(image)
    pred_image = np.vectorize(bw_image)(pred_image)
    pred_image = np.vectorize(binarize_image)(pred_image)
    SMOOTH = 1e-6
    intersection = (image & pred_image).sum()
    union = (image | pred_image).sum()
    
    iou += (intersection + SMOOTH) / (union + SMOOTH)


print(iou/len(data.keys))

