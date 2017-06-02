from pathlib import Path
import logging
import yaml

import mcnn.operations as operations
from mcnn.model import McnnModel, McnnConfiguration, Model, AutoCnnModel, MutatingCnnModel
from mcnn.samples import HorizontalDataset, Dataset


def create_mcnn(dataset: Dataset) -> Model:
    sample_length = 128
    pooling_factor = sample_length // 32
    local_filter_width = sample_length // 32
    # full_filter_width = pooling_factor // 4
    full_filter_width = pooling_factor
    cfg = McnnConfiguration(downsample_strides=[2, 4, 8, 16],
                            smoothing_window_sizes=[8, 16, 32, 64],
                            pooling_factor=pooling_factor,
                            channel_count=256,
                            local_filter_width=local_filter_width,
                            full_filter_width=full_filter_width,
                            layer_size=256,
                            full_pool_size=4)
    model = McnnModel(batch_size=64,
                      num_classes=dataset.target_classes_count,
                      learning_rate=1e-3,
                      sample_length=sample_length,
                      mcnn_configuration=cfg)
    return model


def create_auto_cnn(dataset: HorizontalDataset) -> Model:
    model = AutoCnnModel(batch_size=64,
                         num_classes=dataset.target_classes_count,
                         learning_rate=1e-3,
                         sample_length=dataset.sample_length,
                         filter_width=4)
    return model


def create_mutating_cnn(dataset: HorizontalDataset, checkpoint_dir: Path) -> MutatingCnnModel:
    model = MutatingCnnModel(batch_size=64,
                             num_classes=dataset.target_classes_count,
                             learning_rate=1e-4,
                             sample_length=dataset.sample_length,
                             checkpoint_dir=checkpoint_dir)
    return model


def train(model: Model, dataset: Dataset, checkpoint_dir: Path, log_dir: Path):
    operations.train(model,
                     dataset,
                     step_count=500,
                     checkpoint_dir=checkpoint_dir,
                     log_dir=log_dir,
                     steps_per_checkpoint=100,
                     feature_name='')


def train_and_mutate(model: MutatingCnnModel, dataset: Dataset, checkpoint_dir: Path, log_dir: Path):
    operations.train_and_mutate(model,
                                dataset,
                                step_count=10000,
                                checkpoint_dir=checkpoint_dir,
                                log_dir=log_dir,
                                steps_per_checkpoint=1000,
                                feature_name='')


def visualize(model: Model, dataset: Dataset, checkpoint_dir: Path):
    operations.deconv(model,
                      dataset,
                      sample_count=1000,
                      checkpoint_dir=checkpoint_dir,
                      feature_name='')


def main():
    logging.basicConfig(level=logging.INFO)
    root = Path.home() / 'data' / 'UCR_TS_Archive_2015'

    accuracies = dict()

    #for dataset_path in root.iterdir():
    for dataset_path in [root / 'Adiac']:
        dataset_name = dataset_path.name
        checkpoint_dir = Path('checkpoints') / dataset_name
        log_dir_train = Path('logs') / (dataset_name + '_train')
        log_dir_test = Path('logs') / (dataset_name + '_test')

        dataset = HorizontalDataset(root / dataset_name / (dataset_name + '_TRAIN'),
                                    root / dataset_name / (dataset_name + '_TEST'))

        model = create_mutating_cnn(dataset, checkpoint_dir)
        train_and_mutate(model, dataset, checkpoint_dir, log_dir_train)
        accuracy = operations.evaluate(model, dataset, checkpoint_dir, log_dir_test, feature_name='')
        accuracies[dataset_name] = accuracy
    with open('accuracies.yml', 'w') as outfile:
        yaml.dump(accuracies, outfile, default_flow_style=False)


if __name__ == '__main__':
    main()