#  EfficientNetB03



"""
CircularCrop used in preprocessing comes from : https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy?scriptVersionId=20340219
Phase2 classifies DR stage between
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

gp-split-phase2-data is a concatination of (Eyepacs(train and test datasets)+ Messidor 1 and 2 + Aptos2019 + IDRID)(ONLY CLASS 1,2,3 AND 4 DR (CLASS 0 IS IN PHASE 1)) all Circular Cropped then resized to 320,
shuffled then split in to train and test with 0.10 ratio
"""

# To have reproducible results and compare them
nr_seed = 2019
import tensorflow as tf

#### REQUIREMENTS
#install keras_efficientnets
#install tensorflow 1.14.0
#install keras 2.4.0

# import libraries

import gc
import warnings
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import cv2
import numpy as np

from keras import backend as K
from keras import layers
from keras_efficientnets import EfficientNetB3
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score, accuracy_score

warnings.filterwarnings("ignore")

print(tf.__version__)
print(cv2.__version__)

# Image size
im_size = 320
# Batch size
BATCH_SIZE = 32
# Bucket Number
bucket_num = 9

#this is only a small part of the preprocessing done. the images already went through circular crop and resize
def preprocess_image_old(image, desired_size=im_size):
    img = cv2.addWeighted(image,4,cv2.GaussianBlur(image, (0,0), desired_size/40) ,-4 ,128)#blend two images
    
    return img



#TRAINING DATA CIRCULARY CROPPED AND RESIZED TO 320 X 320 CAN BE FOUND IN THIS NOTEBOOK: https://www.kaggle.com/code/sarahali232/gp-split-phase2-data-to-train-test
x_npz = np.load("../input/gp-split-phase2-data-to-train-test/X_train.npz")
x_trains = x_npz['arr_0']
x_npz = np.load('../input/gp-split-phase2-data-to-train-test/Y_train.npz')
y_train = x_npz['arr_0']

size = round(len(x_trains) / 3)
counts1 = 0
counts2 = 0

x_train1 = np.empty((size, im_size, im_size, 3), dtype=np.uint8)
x_train2 = np.empty((size, im_size, im_size, 3), dtype=np.uint8)
x_train3 = np.empty((size, im_size, im_size, 3), dtype=np.uint8)

for j in range(len(x_trains)):
    if (j < size):
        x_train1[j, :, :, :] = preprocess_image_old(x_trains[j], desired_size=im_size)
    elif (j >= size and j < size * 2):
        x_train2[counts1, :, :, :] = preprocess_image_old(x_trains[j], desired_size=im_size)
        counts1 += 1
    else:
        x_train3[counts2, :, :, :] = preprocess_image_old(x_trains[j], desired_size=im_size)
        counts2 += 1

x_train1 = np.array(x_train2)
x_train2 = np.array(x_train2)
x_train3 = np.array(x_train3)

y_train = np.array(y_train)

print(y_train[0])
print(x_train1.shape)
print(x_train2.shape)
print(x_train3.shape)

print(y_train.shape)

del x_npz
del x_trains
gc.collect()


# TEST DATA CAN BE FOUND IN THIS NOTEBOOK OUTPUT https://www.kaggle.com/code/sarahali232/gp-split-phase2-data-to-train-test

x_npz = np.load("../input/gp-split-phase2-data-to-train-test/X_test.npz")
x_vals = x_npz['arr_0']
x_npz = np.load('../input/gp-split-phase2-data-to-train-test/Y_test.npz')
y_val = x_npz['arr_0']

x_val= np.empty((len(x_vals), im_size, im_size, 3), dtype=np.uint8)
for j in range(len(x_vals)):
    x_val[j, :, :, :] = preprocess_image_old(x_vals[j],desired_size=im_size)

x_val =np.array(x_val)
y_val =np.array(y_val)

print(x_val.shape)
print(y_val.shape)
print(y_val[0])

del x_vals
del x_npz
gc.collect()

#  Display Sample
def display_samples(x,y, columns=4, rows=3):
    fig = plt.figure(figsize=(5 * columns, 4 * rows))
    
    for i in range(columns * rows):
        img = preprocess_image_old(x[i])
        fig.add_subplot(rows, columns, i + 1)
        plt.title(np.sum(y[i]))
        plt.imshow(img)

    plt.tight_layout()

display_samples(x_train1,y_train)

print(y_val[0]) #class 1 0 0 0 dr of degree 1 , class 1 1 0 0 degree 2, and so on
print(y_val[1]) 


#  Creating keras callback for QWK

class Metrics(Callback):

   def on_epoch_end(self, epoch, logs={}): # logs contains the loss value, and all the metrics at the end of a batch or epoch. Example includes the loss and mean absolute error.
        X_val, y_val = self.validation_data[:2] #self.validation_data[0]:x val #self.validation_data[1]:y val
        y_val = y_val.sum(axis=1)  - 1
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1
        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


#  Data Generator
def create_datagen():
    return ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range= 0.3,
        width_shift_range= 0.1,
        height_shift_range = 0.1,
        fill_mode='constant',
        cval=0
    )

#  Model: EfficientNetB3
efficient = EfficientNetB3(
    include_top=False,
    input_shape=(im_size,im_size,3),
)

def build_model():
    model = Sequential()
    model.add(efficient)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0001,decay=1e-6),
        metrics=['accuracy']
    )
    
    return model


#saved model can be found in output of this notebook: https://www.kaggle.com/code/sarahali232/gp-phase2-efficientnet-multi-datasets
model = build_model()
model.load_weights('../input/gp-phase2-efficientnet-multi-datasets/model.h5')
model.summary()

bucket_num = 3
div = round(y_train.shape[0]/bucket_num)
print(div)

#  Training & Evaluation
df_init = {
    'val_loss': [0.0],
    'val_acc': [0.0],
    'loss': [0.0], 
    'acc': [0.0],
    'bucket': [0.0]

}
results = pd.DataFrame(df_init)

# I found that changing the nr. of epochs for each bucket helped in terms of performances
epochs = [5,5,5,5,5,10,10,10,15,15]
kappa_metrics = Metrics()
kappa_metrics.val_kappas = []



#Progressive Training
for i in range(bucket_num):

    if (i == 0):
        data_generator = create_datagen().flow(x_train1, y_train[0:size], batch_size=BATCH_SIZE)
    elif (i == 1):
        data_generator = create_datagen().flow(x_train2, y_train[size:size * 2], batch_size=BATCH_SIZE)
    else:
        data_generator = create_datagen().flow(x_train3, y_train[size * 2:size * 3], batch_size=BATCH_SIZE)

    history = model.fit_generator(
        data_generator,
        steps_per_epoch=len(x_train3) / BATCH_SIZE,
        epochs=epochs[i],
        validation_data=(x_val, y_val),
        callbacks=[kappa_metrics]
    )

    dic = history.history
    df_model = pd.DataFrame(dic)
    df_model['bucket'] = i

    results = results.append(df_model)

    del data_generator
    if (i == 0):
        del x_train1
    elif (i == 1):
        del x_train2
    else:
        del x_train3
    gc.collect()


results = results.iloc[1:]
results['kappa'] = kappa_metrics.val_kappas
results = results.reset_index()
results = results.rename(index=str, columns={"index": "epoch"})
results

results[['loss', 'val_loss']].plot()
results[['acc', 'val_acc']].plot()
results[['kappa']].plot()
results.to_csv('model_results.csv',index=False)

#  Find best threshold
model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

def compute_score_inv(threshold):
    y1 = y_val_pred > threshold
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)

best_threshold = simplex['x'][0]

y1 = y_val_pred > best_threshold
y1 = y1.astype(int).sum(axis=1) - 1
y2 = y_val.sum(axis=1) - 1
score = cohen_kappa_score(y1, y2, weights='quadratic')
print('Threshold: {}'.format(best_threshold))
print('Validation QWK score with best_threshold: {}'.format(score))

y1 = y_val_pred > .5
y1 = y1.astype(int).sum(axis=1) - 1
score = cohen_kappa_score(y1, y2, weights='quadratic')
print('Validation QWK score with .5 threshold: {}'.format(score))

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y2,y1)
print(conf_mat)

