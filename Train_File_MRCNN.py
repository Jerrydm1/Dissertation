import os
import sys
import cv2
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances

# This is the Root directory containing the work for this project
DIR_ROOT = "D:\\env_with_tensorflow1.14\\all_maskrcnn\\gleason_mask_detector"

# Importing Mask RCNN
sys.path.append(DIR_ROOT)  # Finding the local version of the library

# This is the path to the trained weights file
PATH_COCO_WEIGHTS = os.path.join(DIR_ROOT, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
LOGS_DIR_DEFAULT = os.path.join(DIR_ROOT, "logs")

# Training configuration on the custom  dataset.
class CustomConfig(Config):   
    OBJECT_NAME = "object" # Giving the configuration a recognizable name    
    IMAGES_PER_GPU = 1

    # Number of classes to be classified (background inclusive)
    NUM_CLASSES = 4 + 1  # (Benign and Gleason pattern 3, Gleason pattern 4, Gleason pattern 5) + background class

    # Number of training steps per each epoch
    STEPS_PER_EPOCH = 100

    # Skip detections if < 90% confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

#  DATASET FOR THIS PROJECT

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_path, train_path):
        #dataset_path: This is the root directory containing the dataset.
        #train_path: train_path to load: train or val
        
        # Specifying and adding classes to the dataset. 

        self.add_class("object", 1, "Gleason_pattern_3")  #Gleason_pattern_3
        self.add_class("object", 2, "Gleason_pattern_4")
        self.add_class("object", 3, "Gleason_pattern_5")
        self.add_class("object", 4, "Benign")
        train_path = "train"
        dataset_path = os.path.join(dataset_path, train_path)

        # Loading the annotationfilevalues (We are interested in the x and y coordinates of each annotated image region)
        
        annotation_train = json.load(open('D://env_with_tensorflow1.14//all_maskrcnn//gleason_mask_detector//dataset//train//via_project_17Jul2021_12h52m_json.json'))
        
        annotationfilevalues = list(train_annotation.values())
        annotationfilevalues = [a for a in annotationfilevalues if a['regions']]
        
        # Add images and get the x, y coordinates of polygons for each object instance
        for a in annotationfilevalues:            
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            object_names_dict = {"Gleason_pattern_3": 1,"Gleason_pattern_4": 2,"Gleason_pattern_5": 3,"Benign": 4}     
            object_num_ids = [object_names_dict[a] for a in objects]
     
            # Reading the images to determine its size. 
            # load_mask() needs the image size to convert polygons to masks.
            
            print("ids",object_num_ids)
            image_path = os.path.join(dataset_path, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                object_num_ids=object_num_ids
                )
    # Generating instance masks and class IDs for each image

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Converting the polygons to bitmap masks of shape (height, width, instance_count)
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        object_num_ids = info['object_num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        	mask[rr, cc, i] = 1        
        object_num_ids = np.array(object_num_ids, dtype=np.int32)
        return mask, object_num_ids
    
    # Returning image path
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
            
# MODEL TRAINING

def train(model):
    dataset_train = CustomDataset()
    # Dataset for training.
    dataset_train.load_custom("D://env_with_tensorflow1.14//gleason_mask_detector//dataset", "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    # Dataset for validation
    dataset_val.load_custom("D://env_with_tensorflow1.14//gleason_mask_detector//dataset", "val")
    dataset_val.prepare()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=LOGS_DIR_DEFAULT)

path_weights = PATH_COCO_WEIGHTS
# Function for downloading COCO weights file
if not os.path.exists(path_weights):
  utils.download_trained_weights(path_weights)
# Function for loading the downloaded COCO weights file
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)