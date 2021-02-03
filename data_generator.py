import math
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing.image import img_to_array

from data import Data

LABEL_INDEX = {
    'ap': 0,
    'bs': 1,
    'mid': 2,
    'oap': 3,
    'obs': 4,
}


class PhaseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data: Data, datasets=None, batch_size=32, target_size=(224, 224),
                 slices_per_sample=25, shuffle=True, image_data_generator=None):
        self.data = data
        self.datasets = datasets
        self.batch_size = batch_size
        self.target_size = target_size
        self.slices_per_sample = slices_per_sample
        self.shuffle = shuffle
        self.datagen = image_data_generator

        self.n_classes = 5
        self.label_indices = LABEL_INDEX

        self.samples = dict()
        self.max_slices = 1

        if datasets is None:
            datasets = list(data.data.keys())
        if isinstance(datasets, str):
            datasets = [datasets]

        # All plural variables are dicts
        for dataset in datasets:
            for patient, phases in data.data[dataset].items():
                for phase, slices in phases.items():
                    key = "{dataset}_{patient:06d}_{slice:02d}".format(dataset=dataset, patient=patient, slice=phase)
                    self.samples[key] = slices
                    if len(slices) > self.max_slices:
                        self.max_slices = len(slices)

        if slices_per_sample < self.max_slices:
            raise ValueError("There are some samples that contain more than {} slices ({})".format(
                slices_per_sample, self.max_slices))

        unlabeled = []
        self.images_by_label = [0] * len(self.label_indices)
        for phases in self.samples.values():
            for phase in phases.values():
                label = self.data.labels.get(phase, None)
                if label is None:
                    unlabeled.append(phase)
                else:
                    index = self.label_indices[label]
                    self.images_by_label[index] += 1
        if unlabeled:
            raise ValueError("{} unlabeled slice(s): {}...".format(len(unlabeled), str(unlabeled)[:200]))

        self.n_batches = math.ceil(len(self.samples) / batch_size)

        self._refresh_sample_keys()

    def _refresh_sample_keys(self):
        self.sample_keys = list(sorted(self.samples.keys()))
        if self.shuffle:
            np.random.shuffle(self.sample_keys)

    def _get_sample_key_batch(self, index):
        return self.sample_keys[index * self.batch_size:(index + 1) * self.batch_size]

    def _load_and_preprocess_image(self, path, standardize=False):
        img = Image.open(path)
        img = img.resize(self.target_size, Image.NEAREST)
        # img = load_img(path, color_mode="rgb", target_size=self.target_size)
        x = img_to_array(img, data_format="channels_last")
        x = x / 65536 * 255  # color processing (for current png formats)
        if self.datagen:
            params = self.datagen.get_random_transform(x.shape)
            x = self.datagen.apply_transform(x, params)
            if standardize:
                x = self.datagen.standardize(x)
        return x

    def get_slice_index(self) -> List[List[str]]:
        index = []
        keys = list(sorted(self.samples.keys()))
        items = sorted(self.samples.items())  # KAG_000001_00: {}
        for _, slices in items:
            slice_list = list(sorted(slices.values()))
            index.append(slice_list)
        return index

    def get_class_weight(self):
        counts = np.array(self.images_by_label)
        weights = counts.sum() / counts / len(counts)
        weights = {i: weight for i, weight in enumerate(weights.tolist())}
        return weights

    def __getitem__(self, index):  # get batch
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


class SliceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data: Data, datasets=None, batch_size=32,
                 target_size=(224, 224), shuffle=True, image_data_generator=None):
        self.data = data
        self.datasets = datasets
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.datagen = image_data_generator

        self.n_classes = 5
        self.label_indices = LABEL_INDEX

        self.slices = list()

        if datasets is None:
            datasets = list(data.data.keys())
        if isinstance(datasets, str):
            datasets = [datasets]

        # All plural variables are dicts
        for dataset in datasets:
            for patient, phases in data.data[dataset].items():
                for phase, slices in phases.items():
                    sids = [item[1] for item in slices.items()]
                    self.slices.extend(sids)
        self.slices.sort()

        unlabeled = []
        self.images_by_label = [0] * len(self.label_indices)
        for sid in self.slices:
            label = self.data.labels.get(sid, None)
            if label is None:
                unlabeled.append(sid)
            else:
                index = self.label_indices[label]
                self.images_by_label[index] += 1
        if unlabeled:
            raise ValueError("{} unlabeled slice(s): {}...".format(len(unlabeled), str(unlabeled)[:200]))

        self.n_batches = math.ceil(len(self.slices) / batch_size)
        self._refresh_slice_order()

    def _refresh_slice_order(self):
        self.slices = list(sorted(self.slices))
        if self.shuffle:
            np.random.shuffle(self.slices)

    def _get_slice_batch(self, index):
        return self.slices[index * self.batch_size:(index + 1) * self.batch_size]

    def _load_and_preprocess_image(self, path, standardize=False):
        img = Image.open(path)
        img = img.resize(self.target_size, Image.NEAREST)
        # img = load_img(path, color_mode="rgb", target_size=self.target_size)
        x = img_to_array(img, data_format="channels_last")
        x = x / 65536 * 255
        if self.datagen:
            params = self.datagen.get_random_transform(x.shape)
            x = self.datagen.apply_transform(x, params)
            if standardize:
                x = self.datagen.standardize(x)
        return x

    def get_slice_list(self):
        return list(sorted(self.slices))

    def get_class_weight(self):
        counts = np.array(self.images_by_label)
        weights = counts.sum() / counts / len(counts)
        weights = {i: weight for i, weight in enumerate(weights.tolist())}
        return weights

    def __getitem__(self, index):
        """Get `index`th batch
        """
        slices = self._get_slice_batch(index)
        batch_size = len(slices)
        x = np.zeros((batch_size,) + self.target_size + (3,))
        y = np.zeros((batch_size,) + (self.n_classes,))
        for i, sid in enumerate(slices):
            path = self.data.paths[sid]
            image = self._load_and_preprocess_image(path, standardize=True)
            x[i] = image
            label = self.data.labels[sid]
            label_index = self.label_indices[label]
            y[i][label_index] = 1

        assert (y.sum() == len(slices))

        return x, y

    def __len__(self):
        return self.n_batches

    def on_epoch_end(self):
        self._refresh_slice_order()
