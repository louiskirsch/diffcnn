from abc import abstractmethod, abstractproperty
from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd
import numpy as np

import random
import itertools


def batch_samples_from_callback(sample_callback, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    batch = [sample_callback() for _ in range(batch_size)]
    x = np.stack((x for x, y in batch), axis=0)
    y = np.stack((y for x, y in batch), axis=0)
    return x, y


def batch_generator(sample_iterator, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    while True:
        batch = list(itertools.islice(sample_iterator, batch_size))
        if len(batch) == 0:
            return
        x = np.stack((x for x, y in batch), axis=0)
        y = np.stack((y for x, y in batch), axis=0)
        yield x, y


class Dataset:

    @abstractmethod
    def data_generator(self, dataset: str, batch_size: int, feature_name: str = None,
                       sample_length: int = None, loop: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @abstractproperty
    @property
    def target_classes_count(self):
        raise NotImplementedError()

    @property
    def test_sample_count(self) -> int:
        return -1

    @property
    def train_sample_count(self) -> int:
        return -1


class HorizontalDataset(Dataset):

    def __init__(self, dataset_train_path: Path, dataset_test_path: Path):
        self.df_train = pd.read_csv(dataset_train_path, header=None)
        self.df_test = pd.read_csv(dataset_test_path, header=None)
        self.target_classes = self.df_train[0].unique()
        self.class_to_id = dict((c, id) for id, c in enumerate(self.target_classes))
        self._target_classes_count = len(self.target_classes)

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

    def _generate_samples(self, dataset: pd.DataFrame, loop: bool) -> Tuple[np.ndarray, int]:
        row_count = dataset.shape[0]
        while True:
            for row_index in np.random.permutation(row_count):
                x = dataset.iloc[row_index, 1:].values
                y = self.class_to_id[dataset.iloc[row_index, 0]]
                yield x, y
            if not loop:
                return

    def _generate_samples_with_offset(self, dataset: pd.DataFrame, sample_length: int,
                                      loop: bool) -> Tuple[np.ndarray, int]:
        row_count = dataset.shape[0]
        offset_count = self.sample_length - sample_length + 1
        samples_spec = list(itertools.product(range(row_count), range(offset_count)))
        while True:
            random.shuffle(samples_spec)
            for row_index, offset in samples_spec:
                start = 1 + offset
                end = start + sample_length
                x = dataset.iloc[row_index, start:end].values
                y = self.class_to_id[dataset.iloc[row_index, 0]]
                yield x, y
            if not loop:
                return

    def data_generator(self, dataset: str, batch_size: int, sample_length: int = None,
                       loop: bool = False, **kwargs) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        dataset = self.df_train if dataset == 'train' else self.df_test
        if sample_length is None or self.sample_length == sample_length:
            yield from batch_generator(self._generate_samples(dataset, loop), batch_size)
        else:
            yield from batch_generator(self._generate_samples_with_offset(dataset, sample_length, loop), batch_size)


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

    def _generate_batch(self, feature_name: str, batch_size: int, random_index_fnc,
                        sample_length: int) -> Tuple[np.ndarray, np.ndarray]:
        gen_sample = lambda: self._generate_sample(feature_name, random_index_fnc, sample_length)
        return batch_samples_from_callback(gen_sample, batch_size)

    def data_generator(self, dataset: str, batch_size: int, sample_length: int = None, feature_name: str = None,
                       loop: bool = False, **kwargs) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        index_fnc = self._random_training_index if dataset == 'train' else self._random_test_index
        while True:
            yield self._generate_batch(feature_name, batch_size, index_fnc, sample_length)
