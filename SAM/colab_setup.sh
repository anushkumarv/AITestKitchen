#! /bin/sh

# install python dependencies 
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx

# Create dirs 
mkdir pretrained_models
mkdir predictions

# Get SAM pre_trained model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth ./pretrained_models/.

# Get validation images from answer grounding shared task
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/VizWiz_grounding/annotations.zip
unzip -q val.zip 
unzip -q annotations.zip


# Extract ground truth masks
python ./utils/create_binary_masks.py