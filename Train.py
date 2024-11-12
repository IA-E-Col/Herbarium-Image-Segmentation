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


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
#%%
GT = []
#Noisy = []
files = os.listdir('/pools/apollon/ummisco/data/db/ecolplus/datasets/segmentation/dataset_stage_seg_white/dataset_eng_ps_px_Org+ps')
for img in files:
    GT.append('/pools/apollon/ummisco/data/db/ecolplus/datasets/segmentation/dataset_stage_seg_white/dataset_eng_ps_px_Org+ps' +'/' +img)

#%%
Noisy = []

files = os.listdir('/pools/apollon/ummisco/data/db/ecolplus/datasets/segmentation/dataset_stage_seg/dataset_org')
for img in files:
    Noisy.append('/pools/apollon/ummisco/data/db/ecolplus/datasets/segmentation/dataset_stage_seg/dataset_org' + '/' + img)
    
df = pd.DataFrame()
df['Ground Truth Images'] = GT
df['Noisy Images'] = Noisy
df.head()
df.shape

#%%
X = df['Noisy Images']
y = df['Ground Truth Images']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 1), step=patch_size)
  return patches

def patches1(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

with tf.device('/gpu:1'):
    X_train_patches = []
    y_train_patches = []
    for i in range(len(X_train)):
        path = X_train.iloc[i]
        img_nsy = cv2.imread(path)
        img_nsy = cv2.cvtColor(img_nsy, cv2.COLOR_BGR2RGB)
        img_nsy = cv2.resize(img_nsy, (1024, 1024))  # resizing the X_train images
        patches_nsy = patches1(img_nsy, 256)

        path = y_train.iloc[i]
        img_gt = cv2.imread(path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, (1024, 1024))  # resizing the y_train images
        patches_gt = patches(img_gt, 256)

        rows = patches_nsy.shape[0]
        cols = patches_nsy.shape[1]
        for j in range(rows):
            for k in range(cols):
                X_train_patches.append(patches_nsy[j][k][0])
                y_train_patches.append(patches_gt[j][k][0])

    X_train = np.array(X_train_patches)
    y_train = np.array(y_train_patches)

with tf.device('/gpu:1'):
    X_test_patches = []
    y_test_patches = []
    for i in range(len(X_test)):
        path = X_test.iloc[i]
        img_nsy = cv2.imread(path)
        img_nsy = cv2.cvtColor(img_nsy, cv2.COLOR_BGR2RGB)
        img_nsy = cv2.resize(img_nsy, (1024, 1024))  # resizing the X_test images
        patches_nsy = patches1(img_nsy, 256)

        path = y_test.iloc[i]
        img_gt = cv2.imread(path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, (1024, 1024))  # resizing the y_test images
        patches_gt = patches(img_gt, 256)

        rows = patches_nsy.shape[0]
        cols = patches_nsy.shape[1]
        for j in range(rows):
            for k in range(cols):
                X_test_patches.append(patches_nsy[j][k][0])
                y_test_patches.append(patches_gt[j][k][0])

    X_test = np.array(X_test_patches)
    y_test = np.array(y_test_patches)
    
X_train = X_train.astype("float32") / 255.0
y_train = y_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_test = y_test.astype("float32") / 255.0


with tf.device('/gpu:1'):

    seed=24
    from keras.preprocessing.image import ImageDataGenerator

    img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect',
                         preprocessing_function = lambda x: np.where(x > 0.9, 0, 1).astype(x.dtype)) #Binarize the output again. 

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(X_train, augment=True, seed=seed)

    image_generator = image_data_generator.flow(X_train, seed=seed)
    valid_img_generator = image_data_generator.flow(X_test, seed=seed)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)

    def my_image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)

    my_generator = my_image_mask_generator(image_generator, mask_generator)

    validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


    x = image_generator.next()
    y = mask_generator.next()

    batch_size = 32
    steps_per_epoch = 3*(len(X_train))//batch_size

import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#BACKBONE = 'resnet101'
##BACKBONE =  tf.keras.models.load_model('/export/sas4/ummisco/home/hsklab/data_segmentation/models/model_322_0.0982_0.9406.h5',compile=False)
#preprocess_input = sm.get_preprocessing(BACKBONE)

#my_generator = preprocess_input(my_generator)
#validation_datagen = preprocess_input(validation_datagen)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K

def jacard_coef(y_true, y_pred):
    y_true_f =K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)




def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function



#model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model=tf.keras.models.load_model('/export/sas4/ummisco/home/hsklab/data_segmentation/modelsAPB101/model_135_0.0604_0.9625.h5',compile=False)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Custom callback to save the best models based on a specific metric
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Custom callback to save the best models based on a specific metric
class SaveBestModels(Callback):
    def __init__(self, base_path, monitor_metric='val_loss',monitor_metric1='val_iou_score'):
        super(SaveBestModels, self).__init__()
        self.base_path = base_path
        self.monitor_metric = monitor_metric
        self.monitor_metric1 = monitor_metric1
        self.best_metric_value = float('inf')  # Initialize with a large value for loss metrics
        self.best_models = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric_value = logs.get(self.monitor_metric)
        current_metric_value1 = logs.get(self.monitor_metric1)
        
        if current_metric_value is None:
            return
        if current_metric_value1 is None:
            return
        
        if current_metric_value < self.best_metric_value:
            self.best_metric_value = current_metric_value
            self.best_models.append(epoch)
            model_path = self.base_path.format(epoch=epoch, metric_value=current_metric_value,metric_value1=current_metric_value1)
            self.model.save(model_path)
            print(f"\nSaved model with {self.monitor_metric}: {current_metric_value} to {model_path}")
            print(f"\nSaved model with {self.monitor_metric1}: {current_metric_value1} to {model_path}")



# Define the directory and filename pattern for saving the models
model_save_path = "/export/sas4/ummisco/home/hsklab/data_segmentation/modelsAPB101/model_{epoch:02d}_{metric_value:.4f}_{metric_value1:.4f}.h5"


# Create the SaveBestModels callback
save_best_models_callback = SaveBestModels(model_save_path, monitor_metric='val_loss',monitor_metric1='val_iou_score')

callbacks=[save_best_models_callback]

with tf.device('/gpu:1'):
    history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=150,callbacks=callbacks)
    model.save('/export/sas4/ummisco/home/hsklab/data_segmentation/modelsAPB101/UNET101_APB.h5')
