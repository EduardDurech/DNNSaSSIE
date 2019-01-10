import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf


dat_set = "C:/Users/BORG-AO-Ref/Documents/CommercialSaSSIE/train/dat/"
dir_titles = next(os.walk(dat_set))[1]
set_len = 0

res_im = 512
cur_data = "deep"
cur_trgt = "OPL"

ext = (cur_data+".bmp", cur_data+".tif")

X1_train = np.zeros((10, res_im, res_im))
X2_train = np.zeros((10, res_im, res_im))
X3_train = np.zeros((10, res_im, res_im))
X4_train = np.zeros((10, res_im, res_im))
X5_train = np.zeros((10, res_im, res_im))
X6_train = np.zeros((10, res_im, res_im))
X7_train = np.zeros((10, res_im, res_im))
X8_train = np.zeros((10, res_im, res_im))
X9_train = np.zeros((10, res_im, res_im))
X10_train = np.zeros((10, res_im, res_im))
Y_train = np.zeros((len(dir_titles), res_im, res_im))

Xn_train = [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train, X9_train, X10_train]

for ind, path in tqdm(enumerate(dir_titles), total=len(dir_titles)):
	i = 0
	for picture in next(os.walk(dat_set+"/"+path))[2]:
		if picture.endswith(ext):
			im = Image.open(dat_set+"/"+path+"/"+picture)
			imResize = im.resize((res_im,res_im))
			xn = Xn_train[i]
			xn[ind] = imResize
			i += 1

		if picture.startswith("final_"+cur_trgt):
			im = Image.open(dat_set+"/"+path+"/"+picture)
			imResize = im.resize((res_im,res_im))
			Y_train[ind] = imResize
print("Finished resizing")

"""
X1_train = X1_train / 255.0
X2_train = X2_train / 255.0
X3_train = X3_train / 255.0
X4_train = X4_train / 255.0
X5_train = X5_train / 255.0
X6_train = X6_train / 255.0
X7_train = X7_train / 255.0
X8_train = X8_train / 255.0
X9_train = X9_train / 255.0
X10_train = X10_train / 255.0
Y_train = Y_train / 255.0
"""

inputs = Input((10, 512, 512))
lm = Lambda(lambda x: x / 255)(inputs)


c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (lm)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2), padding = 'same') (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2), padding = 'same') (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2), padding = 'same') (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D((2, 2), padding = 'same') (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
p5 = MaxPooling2D((2, 2), padding = 'same') (c5)

tl5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p5)
tl5 = Dropout(0.3) (tl5)
tl5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (tl5)

u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (tl5)
u5 = concatenate([u5, c5])
bl5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u5)
bl5 = Dropout(0.3) (bl5)
bl5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (bl5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (bl5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(262144, activation=tf.nn.selu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(262144, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
model.train_on_batch(np.array(Xn_train), Y_train)