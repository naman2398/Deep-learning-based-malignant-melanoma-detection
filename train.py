import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, auc, roc_auc_score, roc_curve
import sklearn
import scipy
import tensorflow as tf
from tqdm import tqdm
from keras.preprocessing import image
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from efficientnet.tfkeras import EfficientNetB7 as effnetb7
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.base_models import dense_net_model, res_net_model, eff_net_model, inception_v3_model, inception_resnet_v2_model


# CONSTANTS
np.random.seed(2019)
tf.random.set_seed(2019)
TEST_SIZE = 0.25
SEED = 2019
BATCH_SIZE = 8


#LOADING DATASET AND LABELS
train_df = pd.read_csv('../input/trainLabels.csv')

x_train = np.load('./npy_files/train_img.npy')
y_train_multi = np.load("./npy_files/labels.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=TEST_SIZE,
    random_state=SEED
)


# IMAGE DATA GEN FOR DATA AUGMENTATIONS
def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)


# METRICS(MORE CAN BE ADDED USING CALLBACKS)
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
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
            self.model.save('effnetb7_model.h5')

        return

# Initialize the different models as base_models
base_model = inception_resnet_v2_model


def extra_layers(base_model):
    x = base_model.output
    batch_normal = BatchNormalization()(x)
    global_avg_pooling = GlobalAveragePooling2D()(batch_normal)
    drop_out = Dropout(0.5)(global_avg_pooling)
    dense1 = Dense(1024, activation='relu')(drop_out)
    dense2 = Dense(5, activation = 'sigmoid')(dense1)
    model = Model(inputs = base_model.input, outputs = dense2)
    return model

model = extra_layers(base_model)

for layer in model.layers:
    layer.trainable = True
    
# model.summary()

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving. 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',
                              verbose=1)

kappa_metrics = Metrics()

# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy','AUC'])

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=60,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, reduce_lr]
)

model.save('./weight_1.h5')

# print("history.history.keys()")

with open('./history.json', 'w') as fp:
    json.dump(str(history.history), fp)
