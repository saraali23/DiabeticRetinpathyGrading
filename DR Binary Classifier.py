#  EfficientNetB03



"""
CircularCrop used in preprocessing comes from : https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy?scriptVersionId=20340219

Phase1 classifies DR between
1 - DR [1,0]
2 - No DR [1,1]
gp-split-phase1-data-to-train-test file is a concatination of (Eyepacs(train and test datasets)+ Messidor 1 and 2 + Aptos2019 + IDRID)(ONLY CLASS 1,2,3 AND 4 DR (CLASS 0 IS IN PHASE 1))
all Circular Cropped then resized to 320, shuffled then split into 10 buckets , 9 for training and 1 for validation
"""

# To have reproducible results and compare them
nr_seed = 2019
import numpy as np 
np.random.seed(nr_seed)
import tensorflow as tf
tf.set_random_seed(nr_seed)

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




# TEST DATA
x_npz = np.load("../input/gp-split-phase1-data-to-train-test/x-bucket9.npz")
x_vals = x_npz['arr_0']
x_npz = np.load('../input/gp-split-phase1-data-to-train-test/y-bucket9.npz')
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

display_samples(x_val,y_val)

print(y_val[0]) #class [1 0] no dr , class [1 1] dr exist
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
    model.add(layers.Dense(2, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0001,decay=1e-6),
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.load_weights('../input/efficientnetb3-trained-on-dr-datasets/model (1).h5')
model.summary()

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
epochs = [5,5,5,5,5,10,10,10,15,115]
kappa_metrics = Metrics()
kappa_metrics.val_kappas = []



#Progressive Training
for i in range(bucket_num):
    
    x_npz = np.load('../input/gp-split-phase1-data-to-train-test/x-bucket'+str(i)+'.npz')
    x = x_npz['arr_0']
    x_npz = np.load('../input/gp-split-phase1-data-to-train-test/y-bucket'+str(i)+'.npz')
    y_train = x_npz['arr_0']
    
    x_train= np.empty((len(x), im_size, im_size, 3), dtype=np.uint8)
    for j in range(len(x)):
        x_train[j, :, :, :] = preprocess_image_old(x[j],desired_size=im_size)

    x_train =np.array(x_train)
    y_train =np.array(y_train)
    del x
    del x_npz
    gc.collect()
   
    print("done")
    data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE)
    
    history = model.fit_generator(
                data_generator,
                steps_per_epoch=len(x_train)/ BATCH_SIZE,
                epochs=epochs[i],
                validation_data=(x_val, y_val),
                callbacks=[kappa_metrics]
            )

    dic = history.history
    df_model = pd.DataFrame(dic)
    df_model['bucket'] = i

    results = results.append(df_model)
    
    del data_generator   
    del x_train
    del y_train
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

