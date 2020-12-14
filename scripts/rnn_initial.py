from itertools import chain
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D, LSTM, TimeDistributed, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt

from data import Data

LABEL_INDEX = {
    'ap': 0,
    'bs': 1,
    'mid': 2,
    'oap': 3,
    'obs': 4,
}

keras_app = tf.keras.applications.mobilenet
keras_model = tf.keras.applications.mobilenet.MobileNet
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1,
    fill_mode="nearest",
    preprocessing_function=keras_app.preprocess_input)

class SliceDataGenerator(keras.utils.Sequence):
    def __init__(self, data: Data, datasets=None, batch_size=32, target_size=(224, 224), slices_per_sample=25, shuffle=True, image_data_generator=None):
        self.data = data
        self.datasets = datasets
        self.batch_size = batch_size
        self.target_size=  target_size
        self.slices_per_sample = slices_per_sample
        self.shuffle = shuffle
        self.datagen = image_data_generator
        
        self.n_classes = 5
        self.label_indices = { 'ap': 0,
                            'bs': 1,
                            'mid': 2,
                            'oap': 3,
                            'obs': 4,
                           }

        self.samples = dict()
        self.max_slices = 0
        
        if datasets is None:
            datasets = list(data.data.keys())
        if isinstance(datasets, str):
            datasets = [datasets]
        
        # All plural variables are dicts
        for dataset in datasets:
            for patient, slices in data.data[dataset].items():
                for s, images in slices.items():
                    key = "{dataset}_{patient}_{slice}".format(dataset=dataset, patient=patient, slice=s)
                    self.samples[key] = images
                    if len(images) > self.max_slices:
                        self.max_slices = len(images)
        
        if slices_per_sample < self.max_slices:
            raise ValueError("There are some samples that contain more than {} slices ({})".format(
                slices_per_sample, self.max_slices))
        
        unlabeled = []
        self.images_by_label = [0] * len(self.label_indices)
        for slices in self.samples.values():
            for s in slices.values():
                label = self.data.labels.get(s, None)
                if label is None:
                    unlabeled.append(s)
                else:
                    index = self.label_indices[label]
                    self.images_by_label[index] += 1
        if unlabeled:
            raise ValueError("{} unlabeled slice(s): {}...".format(len(unlabeled), str(unlabeled)[:200]))
        
        self.n_batches = math.ceil(len(self.samples) / batch_size)
        
        self._refresh_sample_keys()
    
    def _refresh_sample_keys(self):
        self.sample_keys = sorted(list(self.samples.keys()))
        if self.shuffle:
            np.random.shuffle(self.sample_keys)
        
    def _get_sample_key_batch(self, index):
        return self.sample_keys[index * self.batch_size:(index + 1) * self.batch_size]

    def _load_and_preprocess_image(self, path, standardize=False):
        img = load_img(path, color_mode="grayscale", target_size=(224, 224))
        x = img_to_array(img, data_format="channels_last")
        params = datagen.get_random_transform(x.shape)
        x = x / 255
        x = datagen.apply_transform(x, params)
        x = np.concatenate([x, x, x], axis=2)
        if standardize:
            x = datagen.standardize(x)
        return x
    
    def get_class_weight(self):
        counts = np.array(self.images_by_label)
        weights = counts.sum() / counts / len(counts)
        weights = { i: weight for i, weight in enumerate(weights.tolist())}
        return weights
        
    def __getitem__(self, index):
        """Get `index`th batch
        """
        keys = self._get_sample_key_batch(index)
        batch_size = len(keys)
        x = np.zeros((batch_size, self.slices_per_sample) + self.target_size + (3,))
        y = np.zeros((batch_size, self.slices_per_sample) + (self.n_classes,))
        for i, key in enumerate(keys):
            items = sorted(list(self.samples[key].items()))
            for j, (slice_index, sid) in enumerate(items):
                path = self.data.paths[sid]
                image = self._load_and_preprocess_image(path, standardize=True)
                x[i][j] = image
                label = self.data.labels[sid]
                label_index = self.label_indices[label]
                y[i][j][label_index] = 1
        
        return x, y
    
    def __len__(self):
        return self.n_batches
    
    def on_epoch_end(self):
        self._refresh_sample_keys()

data = Data()

backbone = keras_model(include_top=False, pooling='avg', weights='imagenet', input_shape=(224, 224, 3))
backbone.trainable = False

rnn_model = Sequential()
rnn_model.add(TimeDistributed(backbone))
rnn_model.add(LSTM(512, input_shape=(25, 2048), return_sequences=True))
rnn_model.add(Dropout(0.5))
rnn_model.add(LSTM(512, input_shape=(25, 2048), return_sequences=True))
rnn_model.add(Dropout(0.5))
rnn_model.add(TimeDistributed(Dense(5, activation="softmax")))

print("START LSTM TRAINING")

rnn_model.layers[0].trainable = False
rnn_model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

sequence = SliceDataGenerator(data, "KAG", batch_size=2)
rnn_history = rnn_model.fit(sequence, epochs=300)

print("START FINE-TUNING")

rnn_model.layers[0].trainable = True
rnn_model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
rnn_history = rnn_model.fit(sequence, epochs=300)
