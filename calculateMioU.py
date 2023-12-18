import torch
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as F
from torchmetrics import JaccardIndex
from torchvision import transforms 
import matplotlib.pyplot as plt

import segm.utils.torch as ptu
from segm.data.utils import STATS
from segm.model.factory import load_model
from segm.model.utils import inference
import nazirCityscapes as nazir
import numpy as np


def create_seg_map(img, model, variant, transform, device, normalization):
    # Convert image to tensor and process it
        timg = transform(img).float()
        timg = F.normalize(timg, normalization["mean"], normalization["std"])
        timg = timg.to(device).unsqueeze(0)
        
        # Run inference, make predictions
        im_meta = dict(flip=False)
        logits = inference (
        model,
        [timg],
        [im_meta],
        ori_shape=timg.shape[2:4],
        window_size=variant["inference_kwargs"]["window_size"],
        window_stride=variant["inference_kwargs"]["window_stride"],
        batch_size=1,
        )

        # Store the segmentation map 
        seg_map = logits.argmax(0, keepdim=True)

        return seg_map


# Set device
ptu.set_gpu_mode(True)
mydevice = ptu.device

# Load the data
test_data = datasets.Cityscapes('/datasets/cityscapes/', split='val', mode='fine', target_type='semantic')

# Load and prepare the model
model_path = "/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/seg_tiny_mask/model.pth"
model_dir = Path(model_path).parent
model, variant = load_model(model_path)
model.to(mydevice)
model.eval()

# Normalization variables
normalization_name = variant["dataset_kwargs"]["normalization"]
normalization = STATS[normalization_name]

# transform ToTensor
to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

# Start evaluation
print("Starting test!")
print("-------------------")



with torch.no_grad():
    
    jaccard = JaccardIndex(task='multiclass', num_classes=20, ignore_index=19).to(mydevice)

    for idx , element in enumerate(tqdm(test_data)):
        if idx == 2:
             break
        
        img, smnt = element

        # Convert ground truth mask to tensor and process it
        smnt = nazir.Cityscapes.encode_target(smnt)
        smnt = to_tensor(smnt)
        smnt = smnt.to(mydevice)
        
        seg_map = create_seg_map(img, model, variant, to_tensor, mydevice, normalization)
        
        # Update the Jaccard Values
        jaccard.update(seg_map, smnt)


    # Calculate Mean IoU values       
    miou_value = jaccard.compute()
    print (f"miou value: {miou_value * 100}")

    print("-------------------")
    print("Test ended!")

