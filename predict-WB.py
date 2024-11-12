import os
import pandas as pd
import cv2
#rom google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import seaborn as sns
import random
import numpy as np
import math
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Flatten, Dense, Input, MaxPooling2D, Add, Reshape, concatenate, AveragePooling2D, Multiply, GlobalAveragePooling2D, UpSampling2D, MaxPool2D,Softmax
from tensorflow.keras.activations import softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

def patches1(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 1), step=patch_size)
  return patches

from PIL import Image
import numpy as np

from PIL import Image
import numpy as np

def prediction_pillow(img, model, patch_size=256):
    # Load the image using Pillow
    #img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize((1024, 1024),resample=Image.NEAREST)

    img_np = np.array(img).astype("float32") / 255.0

    # Convert Pillow image to numpy array
    #img_np = np.array(img)
    #img_np = img_np.astype("float32") / 255.0

    # Assuming patches1 and unpatchify functions are defined elsewhere and compatible with numpy arrays
    img_patches = patches1(img_np, patch_size) 

    nsy = []
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)

    # Model prediction
    pred_img = (model.predict(nsy) > 0.5).astype(np.uint8)

    # Reshape and unpatchify
    pred_img = np.reshape(pred_img, (4, 4, 1, patch_size, patch_size, 1))
    pred_img = unpatchify(pred_img, img_np.shape[:2] + (1,))
    print(pred_img.shape)

    # Normalize the predicted mask
    pred_img = np.clip(pred_img, 0, 1)  # Ensure values are in [0, 1] range
    pred_img = pred_img * (255.0 / pred_img.max())  # Scale to [0, 255] range

        # Convert the result to uint8 before converting back to an image
    pred_img_uint8 = pred_img.astype(np.uint8)
    print(pred_img_uint8.shape)

    # Convert numpy array back to Pillow Image
    pred_img_pil = Image.fromarray(np.squeeze(pred_img_uint8))

    return pred_img_pil
    
import tensorflow as tf
with tf.device('/gpu:2'):
    APB= tf.keras.models.load_model('Models/White-Background-model.h5',compile= False)

from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Suppose that 'prediction' is your custom prediction function
# which takes an image as input and returns a segmented image

with tf.device('/GPU:2'):
    original_images_dir = 'Images_test/'
    masks_save_dir = 'Predicted_test_White/'
    original = sorted(os.listdir(original_images_dir))

    for image_name in original:
    
        print(image_name)
        image_path = os.path.join(original_images_dir, image_name)

        # Load the original image and its ICC profile
        original_image = Image.open(image_path)
        icc_profile = original_image.info.get('icc_profile', '')

        w, h = original_image.size


        predicted_mask = prediction_pillow(original_image, APB)

        
        # Redimensionner le masque prédit pour correspondre à l'image original
        resized_mask_pil = predicted_mask.resize((w, h))


        resized_mask_pil = np.array(resized_mask_pil.convert("L"))

        original_image_np = np.array(original_image)


        # Apply the binary mask
        original_image_np[np.where(resized_mask_pil == 0)] = [255, 255, 255]

        # Convert the numpy array back to PIL image
        modified_image = Image.fromarray(np.uint8(original_image_np))

        # Save the modified image with the original ICC profile
        name = os.path.splitext(image_name)[0] + '.jpg'
        modified_image.save(os.path.join(masks_save_dir, name), icc_profile=icc_profile)