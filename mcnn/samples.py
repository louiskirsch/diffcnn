from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd
import numpy as np

import random


class Dataset:

    @abstractmethod
    def data_generator(self, dataset: str, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def target_classes_count(self):
        raise NotImplementedError()

    @property
    def sample_length(self) -> int:
        return 0

    @property
    def test_sample_count(self) -> int:
        return 0

    @property
    def train_sample_count(self) -> int:
        return 0


class AugmentedDataset(Dataset):

    def __init__(self, orig_dataset: Dataset, scale: bool = False, noise: bool = True):
        self.orig_dataset = orig_dataset
        self.scale = scale
        self.noise = noise

    @property
    def target_classes_count(self):
        return self.orig_dataset.target_classes_count

    @property
    def sample_length(self):
        return self.orig_dataset.sample_length

    @staticmethod
    def _add_noise(x: np.ndarray) -> np.ndarray:
        min_values = np.min(x, axis=1)
        max_values = np.max(x, axis=1)
        ranges = max_values - min_values
        scale = np.tile(ranges / 100, (x.shape[1], 1)).T
        noise = np.random.normal(loc=0, scale=scale, size=x.shape)
        return x + noise

    @staticmethod
    def _scale(x: np.ndarray) -> np.ndarray:
        means = np.mean(x, axis=1)
        means = np.expand_dims(means, axis=1)
        centered = x - means
        scale = np.random.uniform(0.5, 1.5, (x.shape[0], 1))
        scaled = centered * scale
        return scaled + means

    def data_generator(self, dataset: str, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for x, y in self.orig_dataset.data_generator(dataset=dataset, batch_size=batch_size):
            if self.noise:
                x = self._add_noise(x)
            if self.scale:
                x = self._scale(x)
            yield x, y


class HorizontalDataset(Dataset):

    def __init__(self, dataset_train_path: Path, dataset_test_path: Path, z_normalize: bool = False):
        self.df_train = pd.read_csv(dataset_train_path, header=None)
        self.df_test = pd.read_csv(dataset_test_path, header=None)
        self.target_classes = self.df_train[0].unique()
        self._target_classes_count = len(self.target_classes)
        self._normalize_classes(self.df_train)
        self._normalize_classes(self.df_test)
        if z_normalize:
            self._z_normalize()

    def _z_normalize(self):
        train_values = self.df_train.ix[:, 1:]
        test_values = self.df_test.ix[:, 1:]
        train_mean = np.mean(train_values.values)
        train_std = np.std(train_values.values)
        self.df_train.ix[:, 1:] = (train_values - train_mean) / train_std
        self.df_test.ix[:, 1:] = (test_values - train_mean) / train_std

    def _normalize_classes(self, df: pd.DataFrame):
        class_to_id = dict((c, id) for id, c in enumerate(self.target_classes))
        df[0].replace(class_to_id, inplace=True)

    @property
    def test_data(self):
        x = self.df_test.ix[:, 1:]
        y = self.df_test.ix[:, 0]
        return x, y

    @property
    def sample_length(self):
        return self.df_train.shape[1] - 1

    @property
    def test_sample_count(self) -> int:
        return self.df_test.shape[0]

    @property
    def train_sample_count(self) -> int:
        return self.df_train.shape[0]

    @property
    def target_classes_count(self):
        return self._target_classes_count

    def data_generator(self, dataset: str, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        dataset = self.df_train if dataset == 'train' else self.df_test
        sample_count = dataset.shape[0]
        permutation = np.random.permutation(sample_count)
        for step in range(int(np.ceil(sample_count / batch_size))):
            offset_start = step * batch_size
            offset_end = min(offset_start + batch_size, sample_count)
            x = dataset.ix[permutation[offset_start:offset_end], 1:]
            y = dataset.ix[permutation[offset_start:offset_end], 0]
            yield x, y


class PercentalSplitDataset(Dataset):

    def __init__(self, dataset_path: Path, target_name: str, split_threshold: float):
        self.df = pd.read_csv(dataset_path)
        self.target = self.df[target_name]
        self.split_threshold = split_threshold

        self.target_classes = self.target.unique()
        self._target_classes_count = len(self.target_classes)

        self.data_class_grouped = [self.df[self.target == cls] for cls in self.target_classes]

    @property
    def target_classes_count(self):
        return self._target_classes_count

    def _random_training_index(self, row_count: int, sample_length: int) -> int:
        return random.randrange(int(self.split_threshold * (row_count - sample_length)))

    def _random_test_index(self, row_count: int, sample_length: int) -> int:
        return random.randrange(int(self.split_threshold * row_count), row_count - sample_length)

    def _row_count_for_class(self, cls_index: int) -> int:
        return self.data_class_grouped[cls_index].shape[0]

    def _z_normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def _generate_sample(self, feature_name: str, random_index_fnc,
                         sample_length: int) -> Tuple[np.ndarray, np.ndarray]:
        cls_index = random.randrange(self._target_classes_count)
        row_count = self._row_count_for_class(cls_index)
        start_index = random_index_fnc(row_count, sample_length)
        x = self.data_class_grouped[cls_index][feature_name][start_index:start_index + sample_length]
        x = self._z_normalize(x)
        y = cls_index
        return x, y

    @staticmethod
    def _batch_samples_from_callback(sample_callback, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch = [sample_callback() for _ in range(batch_size)]
        x = np.stack((x for x, y in batch), axis=0)
        y = np.stack((y for x, y in batch), axis=0)
        return x, y

    def _generate_batch(self, feature_name: str, batch_size: int, random_index_fnc,
                        sample_length: int) -> Tuple[np.ndarray, np.ndarray]:
        gen_sample = lambda: self._generate_sample(feature_name, random_index_fnc, sample_length)
        return self._batch_samples_from_callback(gen_sample, batch_size)

    def data_generator(self, dataset: str, batch_size: int, sample_length: int = None, feature_name: str = None,
                       loop: bool = False, **kwargs) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        index_fnc = self._random_training_index if dataset == 'train' else self._random_test_index
        while True:
            yield self._generate_batch(feature_name, batch_size, index_fnc, sample_length)
