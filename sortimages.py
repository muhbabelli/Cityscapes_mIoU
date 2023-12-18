import torch
from torchvision import datasets
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as F
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import transforms 
import matplotlib.pyplot as plt

import segm.utils.torch as ptu
from segm.data.utils import STATS
from segm.model.factory import load_model
from segm.model.utils import inference
import nazirCityscapes as nazir
import numpy as np
from PIL import Image
import os
#import pickle


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

        logits = logits.cpu()
        # Store the segmentation map 
        seg_map = logits.argmax(0, keepdim=True).to(mydevice)
        
        del logits

        return seg_map


def create_save_overlayed_imgs(img, seg_map, path, special_class = None):
    color_map = {
    0: [128, 64, 128],    # 0 : road
    1: [244, 35, 232],   # 1 : sidewalk
    2: [70, 70, 70],   # 2 : building
    3: [102, 102, 156],   # 3 : wall
    4: [190, 153, 153],  # 4 : fence
    5: [153, 153, 153],  # 5 : pole
    6: [250, 170, 30],  # 6 : traffic light
    7: [220, 220, 0],  # 7 : traffic sign
    8: [107, 142, 35],   # 8 : vegetation
    9: [152, 251, 152],   # 9 : terrain
    10: [70, 130, 180],   # 10 : sky
    11: [220, 20, 60],  # 11 : person
    12: [255, 0, 0],  # 12 : rider
    13: [0, 0, 142],  # 13 : car
    14: [0, 0, 70],  # 14 : truck
    15: [0, 60, 100],    # 15 : bus
    16: [0, 80, 100],    # 16 : train
    17: [0, 0, 230],    # 17 : motorcycle
    18: [119, 11, 32],  # 18 : bicycle
    }
    # Create an RGB image with labeled colors
    color_map[special_class] = [0, 204, 204]

    seg_map_np = seg_map.cpu().numpy()
    colored_map = np.zeros((seg_map_np.shape[1], seg_map_np.shape[2], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_map[seg_map_np[0] == label] = color

    # Convert the numpy array to a PIL Image and save it
    predicted_image = Image.fromarray(colored_map)
    predicted_image = predicted_image.resize(img.size, Image.NEAREST)

    # Create an Overlayed Image with original and predicted images
    alpha = 0.35
    blended_image = Image.blend(img.convert("RGBA"), predicted_image.convert("RGBA"), alpha)
    blended_image.save(path, "PNG")

class_names = {
    0: 'road',
    1: 'sidewalk',
    2:'building',
    3: 'wall',
    4:'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    }

class Images:
    def __init__(self, overall_iou, img, seg_map, smnt, iou_per_class, index=0):
        self.overall_iou = overall_iou
        self.img = img
        self.seg_map = seg_map
        self.smnt = smnt
        self.iou_per_class = iou_per_class

        """d = {
            "img_path": img,
            "seg_map":
        }"""

        """np.save(d, "th")"""


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
images_data = {}
jaccard_per_class = MulticlassJaccardIndex(num_classes=20, average='none', ignore_index= 19).to(mydevice)
with torch.no_grad():
    for idx , element in enumerate(tqdm(test_data)):
                
        img, smnt = element

        # Convert ground truth mask to tensor and process it
        smnt = nazir.Cityscapes.encode_target(smnt)
        smnt = to_tensor(smnt)
        smnt = smnt.to(mydevice)
        
        seg_map = create_seg_map(img, model, variant, to_tensor, mydevice, normalization)
        
        # find the mIoU value per image
        jaccard = JaccardIndex(task='multiclass', num_classes=20, ignore_index=19).to(mydevice)
        jaccard.update(seg_map, smnt)
        miou_value = jaccard.compute()
  

        # calculate the iou of each class for each image
        classes_iou_per_img = MulticlassJaccardIndex(num_classes=20, average='none', ignore_index= 19).to(mydevice)
        classes_iou_per_img.update(seg_map, smnt)
        classes_iou_per_img = (classes_iou_per_img.compute()) * 100

        
        # Find mIoU value for each class
        jaccard_per_class.update(seg_map, smnt)

        # store info of each image in images_data
        overall_iou_per_img = miou_value.item() * 100

        image = Images(overall_iou_per_img, img, seg_map, smnt, classes_iou_per_img)

        images_data[idx] = image

        torch.cuda.empty_cache()


# Compute the mIoU for each class and put them in classes_dict_sorted sorted by mIoU score
iou_per_class = (jaccard_per_class.compute()) * 100

classes_dict = {}

for num in range (0,19):
    classes_dict[num] = iou_per_class[num].item()

classes_dict_sorted = dict(sorted(classes_dict.items(), key=lambda x:x[1]))

# find classes with IoU lower than 80 and extract images containing that class
# Store these images in class_img_dict. key: class, value = image object 
class_img_dict = {}

for class_ in classes_dict_sorted:
    if classes_dict_sorted[class_] < 80:    
        class_img_dict[class_] = [] # create an empty list to store the images in.
        for image in images_data:
            if torch.any(images_data[image].smnt == class_):
                class_img_dict[class_].append(images_data[image]) # add the image to the ,initially empty, list that is the value of the class key.
        
        if class_img_dict[class_] == []:
            class_img_dict.pop(class_)
            continue

        class_img_dict[class_] = sorted(class_img_dict[class_], key=lambda x:x.iou_per_class[class_]) # Sort the iou values from lowest to highest



# Save the worst 10 images of each class under a subdirectory for that class.
for key in class_img_dict:
    img_count = 1
    directory = f"class_{key}/"
    parent_dir = "/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/pics/" 
    dir_path = os.path.join(parent_dir, directory)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for image in class_img_dict[key]:
        if img_count > 10: # Save only the worst 10 images
            break
        image_filename = f"overlayed_{img_count}.png"
        image_path = os.path.join(dir_path, image_filename)
        create_save_overlayed_imgs(image.img,image.seg_map, image_path, special_class = key)
        print(f"IMAGE SAVED ! IoU of class {key} in image {img_count}: {image.iou_per_class[key]}")
        img_count += 1

    print("--------------")



# Save the LOWEST OVERALL IOU 5 Overlayed images to visualize the prediction and the original image
worst_directory = "worst_images/"
parent_dir = "/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/pics/" 
worst_dir_path = os.path.join(parent_dir, worst_directory)
if not os.path.exists(worst_dir_path):
    os.mkdir(worst_dir_path)

# Save the BEST OVERALL IOU 5 Overlayed images to visualize the prediction and the original image
best_directory = "best_images/"
parent_dir = "/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/pics/" 
best_dir_path = os.path.join(parent_dir, best_directory)
if not os.path.exists(best_dir_path):
    os.mkdir(best_dir_path)

sorted_images_data = dict(sorted(images_data.items(), key=lambda x:x[1].overall_iou)) # Sort the iou values from lowest to highest
count = 1
for idx in sorted_images_data:
    
    img = sorted_images_data[idx].img
    seg_map = sorted_images_data[idx].seg_map
    smnt = sorted_images_data[idx].smnt    
    
    image_filename = f"overlayed_{count}.png"

    if count <= 10:
        # Store the image in "worst_images"
        image_path = os.path.join(worst_dir_path, image_filename)
        create_save_overlayed_imgs(img, seg_map, image_path)
    
    elif count >= len(sorted_images_data) - 5 : 
        # Store the image in "best_images"
        image_path = os.path.join(best_dir_path, image_filename)
        create_save_overlayed_imgs(img, seg_map, image_path)
        
    count += 1
 
# ------------------------------------------------------

# Plot IoU-Class graph of 5 lowest classes 
classes = [class_names[x] for x in list(classes_dict_sorted.keys())]
fig, ax = plt.subplots()
colors = ['red', 'blue', 'orange']
bars = ax.bar(classes[:5], list(classes_dict_sorted.values())[:5], color = colors)

for bar, value in zip(bars, list(classes_dict_sorted.values())[:5]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')

ax.set_title('Lowest IoU')
ax.set_xlabel('Class')
ax.set_ylabel('IoU')
plt.savefig('/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/graphs/IoU_Class_lowest.png')


# Plot IoU-Class graph of 5 highest classes 
fig, ax = plt.subplots()
colors = ['red', 'blue', 'orange']
bars = ax.bar(classes[-5:], list(classes_dict_sorted.values())[-5:], color = colors)

for bar, value in zip(bars, list(classes_dict_sorted.values())[-5:]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')

ax.set_title('Highest IoU')
ax.set_xlabel('Class')
ax.set_ylabel('IoU')
plt.savefig('/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/graphs/IoU_Class_highest.png')

# Plot IoU-Class graph of all classes
fig, ax = plt.subplots()
colors = ['red', 'blue', 'orange']
bars = ax.bar(classes, list(classes_dict_sorted.values()), color = colors)

for bar, value in zip(bars, list(classes_dict_sorted.values())):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')

ax.set_title('Class - IoU')
ax.set_xlabel('Class')
ax.set_ylabel('IoU')
plt.savefig('/kuacc/users/mbabelli22/myworkfolder/CityScapes_inference/segmenter/graphs/IoU_all_classes.png')
